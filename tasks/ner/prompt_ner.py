from typing import List, Optional
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import torch
from utils import *
from ner_data import prompt_sample_MNER
from ner_metric import *
from transformers import MBartTokenizer, MBartForConditionalGeneration, MBartConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import MT5EncoderModel, T5Tokenizer
import torch.nn as nn
import traceback
import random

class PromptNER:
    """
    This is the the class inheriting Prompt class
    member variables:
            input_text (dict): one sample to be processed. For example:
                {"sentence": "I have been in New York for two years",
                 "span": "two years"}
            templates (list[str]): ["[sentence] ||| [span] is [answer]."], where ||| is used to tell which part is fed to encoder or decoder if needed.
            answers (list[str]): ["location entity","person entity","organization entity", "non-entity"]
            answer_to_label (dict): {"location entity":"LOC", "person entity":"PERSON","organization entity":"ORG", "non-entity":"O"}
    """

    def __init__(self, args):
        self.args = args
        if self.args.eval_model_class == 'bart':
            self.tokenizer = BartTokenizer.from_pretrained(self.args.eval_tokenizer)
            self.model = BartForConditionalGeneration.from_pretrained(self.args.eval_model)
            self.max_length = 1024

        elif self.args.eval_model_class =='t5':
            self.tokenizer = T5Tokenizer.from_pretrained(self.args.eval_tokenizer)
            self.model = T5ForConditionalGeneration.from_pretrained(self.args.eval_model)
            self.max_length = 1024
            # when generating, we will use the logits of right-most token to predict the next token
            # so the padding should be on the left
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token = self.tokenizer.eos_token  # to avoid an error
        elif self.args.eval_model_class == 'mbart':
            self.tokenizer = MBartTokenizer.from_pretrained(self.args.eval_tokenizer)
            self.model = MBartForConditionalGeneration.from_pretrained(self.args.eval_model)
            self.max_length = 1024
        elif self.args.eval_model_class =='mt5':
            self.tokenizer = T5Tokenizer.from_pretrained(self.args.eval_tokenizer)
            self.model = MT5ForConditionalGeneration.from_pretrained(self.args.eval_model)
            self.max_length = 1024
            # when generating, we will use the logits of right-most token to predict the next token
            # so the padding should be on the left
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token = self.tokenizer.eos_token  # to avoid an error

        self.model.eval()
        self.model.config.use_cache = False
        self.model.cuda()
        # self.model.cpu()

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)


    def prompting(self,):
        """
        :param args:
        :param kwargs:
        :return: a list of prompts, which consist of a tuple of string:
            (1) prompt_cell_encoder that will be fed into BART encoder;
            (2) prompt_cell_decoder that will be fed into BART decoder;
            Caveat: this may be unnecessary if we used left-to-right pretrained models. Let's think more later.
        """
        # read data...

        input_texts, lang_nclass, lang_idx2label = prompt_sample_MNER(
            args=self.args,
            data_name=args.data_name,
            path_temp=args.path_temp,
            dir=args.dir,
            language=args.language,
            column_no=1,
            delimiter='\t',
            ratio_neg=1.5,
            dtype='test',
            max_span_width=6)

        self.args.n_class = lang_nclass[self.args.language]
        self.args.batch_size = 5 * self.args.n_class
        idx2label = lang_idx2label[self.args.language]
        label2idx = {}
        for idx, label in idx2label.items():
            label2idx[label] = idx

        scores = self.answer_scoring(input_texts, label2idx,batch_size=self.args.batch_size)

        # print args...
        text = '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])
        print()
        print('args value: ')
        print(text)
        if self.args.temp_form == 'mask_prob':
            f1, p, r, correct_preds, total_preds, total_correct = self.evaluate(input_texts, scores, self.args.n_class,
                                                                                kw_idx2label)
        else:
            f1, p, r, correct_preds, total_preds, total_correct = self.evaluate(input_texts, scores,self.args.n_class,idx2label)


        # return pred_ress
        print('f1, p, r, correct_preds, total_preds, total_correct: ')
        print(f1, p, r, correct_preds, total_preds, total_correct)

        return [f1, p, r, correct_preds, total_preds, total_correct]


    def answer_scoring(self, input_texts,label2idx,batch_size):
        """ Score a batch of examples
            batch_size equal to the number of the classes.
            pad_token_id=1, Padding token id.
            bos_token_id=0, Beginning of stream token id.
            eos_token_id=2, End of stream token id.
        """
        score_list = []
        for i in range(0, len(input_texts), batch_size):
            input_list = input_texts[i:i+batch_size]
            src_list = []
            tgt_list = []
            for input in input_list:
                # src_list.append(input["sentence"].lower())
                # tgt_list.append(input["target"].lower())
                if self.args.temp_form == 'generate' or self.args.temp_form == 'multi_ans':
                    src_list.append(input["sentence"])
                    tgt_list.append(input["target"])
                # if self.args.temp_form == 'mask':
                #     src_list.append(input["sent2"])
                #     tgt_list.append(input["target2"])
                # if self.args.temp_form =='mask_fill':
                #     src_list.append(input["sent4"])
                #     tgt_list.append(input["target4"])
                # if self.args.temp_form =='t5mask':
                #     src_list.append(input["sent3"])
                #     tgt_list.append(input["target3"])
                # if self.args.temp_form =='mask_prob':
                #     src_list.append(input["sent1"])
                #     tgt_list.append(input["target1"])
            # print('src_list: ', src_list)
            # print('tgt_list: ', tgt_list)
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )

                    src_tokens = encoded_src['input_ids'].cuda()
                    src_mask = encoded_src['attention_mask'].cuda()

                    tgt_tokens = encoded_tgt['input_ids'].cuda()
                    tgt_mask = encoded_tgt['attention_mask'].cuda()
                    tgt_len = tgt_mask.sum(dim=1).cuda()

                    if self.args.score_fun == 'bartScore':
                        # begin{bartScore score...}
                        output = self.model(
                            input_ids=src_tokens,
                            attention_mask=src_mask,
                            labels=tgt_tokens
                        )
                        logits = output.logits.view(-1, self.model.config.vocab_size)
                        loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                        loss = loss.view(tgt_tokens.shape[0], -1)
                        loss = loss.sum(dim=1) / tgt_len
                        curr_score_list = [-x.item() for x in loss]
                        score_list += curr_score_list
                        # end{bartScore score...}
                    elif self.args.score_fun =='TNS':
                        # begin{templatedNER score...}
                        tgt_tokens[:, 0] = 2
                        output = self.model(
                            input_ids=src_tokens,
                            decoder_input_ids=tgt_tokens[:, :tgt_tokens.shape[1] - 2]
                            # decoder_input_ids=tgt_tokens
                        )
                        # begin{original TemplateNER-score}
                        output_length_list = [0] * 5 * 1
                        for i in range(len(tgt_list) // 5):
                            base_length = ((self.tokenizer(tgt_list[i * 5+1], return_tensors='pt', padding=True, truncation=True)[
                                'input_ids']).shape)[1] - 4
                            output_length_list[i * 5:i * 5 + 5] = [base_length] * 5
                            O_idx =label2idx['O']
                            output_length_list[i * 5 + O_idx] += 1
                        score = [1] * tgt_tokens.shape[0]
                        for i in range(tgt_tokens.shape[1] - 3):
                            logits = output[0][:, i, :]
                            logits = logits.softmax(dim=1)
                            logits = logits.cpu().numpy()
                            for j in range(0, tgt_tokens.shape[0]):
                                if i < output_length_list[j]:
                                    score[j] = score[j] * logits[j][int(tgt_tokens[j][i + 1])]
                        score_list += score

                        # end{original TemplateNER-score}
                    elif self.args.score_fun =='Mask_prob':
                        # begin{templatedNER score...}
                        tgt_tokens[:, 0] = 2
                        output = self.model(
                            input_ids=src_tokens,
                            attention_mask=src_mask,
                            decoder_input_ids=tgt_tokens
                        ) #tgt_tokens[:, :tgt_tokens.shape[1] - 2]: for PLOM: lack id = [2,1]; for 'O': lack id = [2,479]


                        score = [1] * tgt_tokens.shape[0]
                        for i in range(tgt_tokens.shape[1]):
                            logits = output[0][:, i, :]
                            logits = logits.softmax(dim=1)
                            logits = logits.cpu().numpy()
                            for j in range(0, tgt_tokens.shape[0]):
                                # keep the probility of the mask word...
                                token = self.tokenizer.convert_ids_to_tokens([tgt_tokens[j][-2]])
                                score[j] = logits[j][int(tgt_tokens[j][-3])]
                        score_list += score


                    elif self.args.score_fun =='TNSAT':
                        # begin{templatedNER score...}
                        tgt_tokens[:, 0] = 2
                        output = self.model(
                            input_ids=src_tokens,

                            decoder_input_ids=tgt_tokens
                        )
                        score = [1] * tgt_tokens.shape[0]
                        for i in range(tgt_tokens.shape[1]):
                            logits = output[0][:, i, :]
                            logits = logits.softmax(dim=1)
                            logits = logits.cpu().numpy()
                            for j in range(0, tgt_tokens.shape[0]):
                                if i < tgt_len[j]-3:
                                    score[j] = score[j] * logits[j][int(tgt_tokens[j][i + 1])]
                        score_list += score

                        ''' 
                        # consider length [8,8,8,8,9]; 479-->'.'
                        #   tensor([[863,   591,  1889,    16,    10,  2259, 10014,   479],
                      #             [863,   591,  1889,    16,    10,   621, 10014,   479],
                      #             [863,   591,  1889,    16,    41,  1651, 10014,   479],
                      #             [863,   591,  1889,    16,    41,    97, 10014,   479],
                      #             [863,   591,  1889,    16,    45,    10,  1440, 10014,   479]],
                        '''

                    elif self.args.score_fun == 'mask_fill':
                        # print('mask-fill')
                        # begin{mask filling and templatedNER score...}
                        logits = self.model(src_tokens).logits
                        masked_index = src_tokens.eq(self.tokenizer.mask_token_id)
                        scores = []
                        for i,(logit, mask_bool,tgt_token) in enumerate(zip(logits, masked_index,tgt_tokens)):
                            probs = logit[mask_bool].softmax(dim=1)
                            probs = probs.cpu().numpy()
                            score = []
                            for j in range(tgt_len[i]-2):
                                prob = probs[0][tgt_token[j+1]]
                                score.append(prob)
                            score_avg = score[0]
                            scores.append(score_avg)
                        score_list +=scores
                    else:
                        print('Error!')

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
            # break
        return score_list


    def evaluate(self, input_texts, scores,n_class,idx2label):
        print('batch_size: ',self.args.batch_size)
        print('n_class: ',n_class)
        print('idx2label: ',idx2label)
        pred_ress = []

        for i in range(0, len(input_texts), n_class):
            input_list = input_texts[i:i+n_class]
            score_list = scores[i:i+n_class]
            # print('input_list: ',len(input_list))
            # print('score_list: ',score_list)

            max_score = max(score_list)
            label_idx = score_list.index(max_score)
            pred_label = idx2label[label_idx]
            score = score_list[label_idx]

            # print('score: ',score)
            # print('pred_label: ',pred_label)
            # print()

            span = input_list[0]['span']
            true_tag = input_list[0]['true_tag']
            sent_span_idx = input_list[0]['sent_span_idx']

            pred_res = {'span': span,
                        'sent_span_idx': sent_span_idx,
                        'true_tag': true_tag,
                        'pred_tag': pred_label,
                        'score': str(max_score)}
            pred_ress.append(pred_res)

        random_int = '%08d' % (random.randint(0, 100000000))
        print('random_int: ',random_int)
        fn_write = self.args.fn_write_dir+self.args.data_name+'_'+self.args.model_name+ '_'+random_int+'.json'
        write_res(pred_ress, fn_write)

        f1, p, r, correct_preds, total_preds, total_correct = evaluate_metric(pred_ress)
        return f1, p, r, correct_preds, total_preds, total_correct




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning tagger using neural networks')
    parser.add_argument('--data_name', default='conll03',
                        help='dataset name')
    parser.add_argument('--path_temp', default='../data/ner/multilingual_NER_template.xlsx',
                        help='dataset name')
    parser.add_argument('--dir', default="../data/ner/panx-test/",  #
                        help='dataset path')
    parser.add_argument('--language', default="en",  #
                        help='languages considered.')

    parser.add_argument('--data_path', default='data/ner/conll2003txt/test.txt', #
                        help='dataset path')
    parser.add_argument('--use_small_data', type=str2bool, default=False,  choices=[True, False],
                        help='consider 10% dataset')
    parser.add_argument('--eval_model_class', default='bart', choices=['bart', 't5','mbart'],
                        help='evaluate model')
    parser.add_argument('--eval_model', default='facebook/bart-base',
                        help='evaluate model')
    parser.add_argument('--eval_tokenizer', default='facebook/bart-base',
                        help='evaluate tokenizer')
    parser.add_argument('--score_fun', default='bartScore', choices=['bartScore', 'TNS','TNSAT','Mask_prob','mask_fill'],
                        help='evaluate tokenizer')
    parser.add_argument('--temp_form', default='generate', choices=['generate', 'mask','t5mask','mask_prob','mask_fill','multi_ans'],
                        help='...')
    parser.add_argument('--fn_write_dir', default='results/ner/',  #
                        help='dataset path')
    parser.add_argument('--model_name', default='XX',  #
                        help='dataset path')


    parser.add_argument('--column_no', type=int, default=-1,
                        help='NER column')
    parser.add_argument('--delimiter', default='\t',
                        help='NER column')
    parser.add_argument('--ratio_neg', type=int, default=1.5,
                        help='NER column')
    # parser.add_argument('--n_class', type=int, default=5,
    #                     help='the nmber of NER classes..')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='the nmber of NER classes..')

    parser.add_argument('--dtype', default='test',
                        help='choice is [train, test]')
    parser.add_argument('--max_span_width', type=int, default=6,
                        help='NER column')
    args = parser.parse_args()

    pner = PromptNER(args)
    res = pner.prompting()




