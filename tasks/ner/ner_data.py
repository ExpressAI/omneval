from ner_metric import *
from utils import *
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
import pickle
import os
import pandas as pd

def read_MNER_template(file_path):
    df = pd.read_excel(file_path)
    data = pd.DataFrame(df)
    langs = data['language']
    continus_OrNot = data['continue']

    O_ans = data['O']
    LOC_ans = data['LOC']
    PER_ans = data['PER']
    ORG_ans = data['ORG']
    MISC_ans = data['MISC']
    lang_label2ans = {}
    lang_continue = {}
    for lang, O_an, LOC_an, PER_an, ORG_an, MISC_an, conti in zip(langs,O_ans,LOC_ans,PER_ans,ORG_ans,MISC_ans,continus_OrNot):
        if lang not in lang_label2ans:
            lang_label2ans[lang] ={}
        # if lang not in ['zh']:
        if 'O' not in lang_label2ans[lang]:
            lang_label2ans[lang]["O"] = O_an
        if 'LOC' not in lang_label2ans[lang]:
            lang_label2ans[lang]["LOC"] = LOC_an
        if 'PER' not in lang_label2ans[lang]:
            lang_label2ans[lang]["PER"] = PER_an
        if 'ORG' not in lang_label2ans[lang]:
            lang_label2ans[lang]["ORG"] = ORG_an

        if lang not in lang_continue:
            lang_continue[lang] = conti
        # if 'O' not in lang2ans[lang]:
        #     lang2ans[lang]["O"] = ' '+' '.join(O_an.split(' ')[1:])[:-1]+' '+O_an[-1]
        # if 'LOC' not in lang2ans[lang]:
        #     lang2ans[lang]["LOC"] = ' '+' '.join(LOC_an.split(' ')[1:])[:-1]+' '+LOC_an[-1]
        # if 'PER' not in lang2ans[lang]:
        #     lang2ans[lang]["PER"] = ' '+' '.join(PER_an.split(' ')[1:])[:-1]+' '+PER_an[-1]
        # if 'ORG' not in lang2ans[lang]:
        #     lang2ans[lang]["ORG"] = ' '+' '.join(ORG_an.split(' ')[1:])[:-1]+' '+ORG_an[-1]
        # if 'MISC' not in lang2ans[lang]:
        #     lang2ans[lang]["MISC"] = ' '.join(MISC_an.split(' ')[1:])[:-1]+' '+MISC_an[-1]

    # for k,v in lang_label2ans.items():
    #     print("k: ",k)
    #     print('v: ',v)

    return lang_label2ans,lang_continue

def label2template_MNER(args, file_path, data_name):
    lang_label2ans,lang_continue = read_MNER_template(file_path)
    lang_idx2label = {}
    lang_nclass = {}
    for lang, lab2ans in lang_label2ans.items():
        if lang not in lang_idx2label:
            lang_idx2label[lang] = {}
        for i,(lab, ans) in enumerate(lab2ans.items()):
            lang_idx2label[lang][i] = lab

        if lang not in lang_nclass:
            lang_nclass[lang] = {}
        lang_nclass[lang] = len(lab2ans)



    return lang_label2ans, lang_idx2label,lang_nclass,lang_continue





def label2template(args,data_name):
    # define the template for each tag and tag id for a given dataset.
    # templates = []
    idx2label = []
    label2temp=[]
    t5temp = {}
    label2vocab = {}
    if data_name=='conll03':
        templates = [" is not an entity .", " is a location name .", " is a person name .", " is an organization name .",
                               " is an other name ."]
        idx2label = {0:'O',1: 'LOC', 2: 'PER', 3: 'ORG', 4: 'MISC'}
        label2temp = {'O':" is not a named entity .",    # " is not an entity ."
                        'LOC':" is a location entity .",   # " is a location name ."
                        'PER':" is a person entity .",      # " is a person name ."
                        'ORG':" is an organization entity .",   # " is an organization name"
                        'MISC':" is an other entity .", # " is another name"
                          }

        label2temp_mask = {'O': "not",  # " is not an entity ."
                      'LOC': "location",  # " is a location name ."
                      'PER': "person",  # " is a person name ."
                      'ORG': "company",  # " is an organization name"
                      'MISC': "product",  # " is another name"
                      }

        # label2temp = {'O': " is not an entity name .",  # " is not an entity ."
        #               'LOC': " is a location name .",  # " is a location name ."
        #               'PER': " is a person name .",  # " is a person name ."
        #               'ORG': " is an organization name .",  # " is an organization name"
        #               'MISC': " is an other name .",  # " is another name"
        #               }
        t5temp = {'O': "not entity name",  # " is not an entity ."
                      'LOC': "a location name",  # " is a location name ."
                      'PER': "a person name",  # " is a person name ."
                      'ORG': "an organization name",  # " is an organization name"
                      'MISC': "an other name",  # " is another name"
                      }
        # label2vocab = {'O': ["other"],  # " is not an entity ."
        #               'LOC': ["location","park","geopolitical", "road", "railway", "highway", "transit","island","mountain"],  # " is a location name ."
        #               'PER': ["person", "artist", "author", "director","athlete","scholar", "scientist","politician","actor","soldier"],  # " is a person name ."
        #               'ORG': ["organization","media", "newspaper","government", "agency","education","show", "sports", "league","religion","company","team","party"],  # " is an organization name"
        #               'MISC': ["product", "software","food", "game","weapon",
        #                        "event","sportsevent","war","disaster","election","protest",
        #                        "art", "music", "writtenart", "film","painting","broadcastprogram",
        #                        "biologything", "chemicalthing", "livingthing", "astronomything",
        #                        "god", "law", "award", "disease", "medical",
        #                        "language", "currency", "educationaldegree",
        #                        ],  # " is another name"
        #               }

        # label2vocab = {'O': ["other"],  # " is not an entity ."
        #                'LOC': ["location", "park", "geopolitical", "road", "city"],  # " is a location name ."
        #                'PER': ["person", "artist", "author", "director", "athlete",
        #                        "scientist", "politician", "actor", "soldier","scholar"],  # " is a person name ."
        #                'ORG': ["organization", "newspaper", "government", "show","sports",
        #                        "league", "religion", "company", "team", "party"],
        #                # " is an organization name"
        #                'MISC': ["product", "food", "game", "weapon", "event",
        #                         "war", "art", "music", "film", "broadcastprogram",
        #                         # "law", "award",
        #                         #  "disease", "language",
        #                         ],  # " is another name"
        #                }

        # # for the other 4 models
        # label2vocab = {'O': ["other"],  # " is not an entity ."
        #                'LOC': ["location", "park", "geopolitical", "road", "city"],  # " is a location name ."
        #                'PER': ["person", "artist", "author", "director", "athlete",
        #                        "scientist", "politician", "actor"],  # " is a person name ."
        #                'ORG': ["organization", "government",
        #                        "league", "company", "team", ],
        #                # " is an organization name"
        #                'MISC': ["product", "food", "game", "weapon", "event",
        #                         "war", "art", "music", "film",
        #                         # "law", "award",
        #                         #  "disease", "language",
        #                         ],  # " is another name"
        #                }

        # for cnn-based plm
        label2vocab = {'O': ["other"],  # " is not an entity ."
                       'LOC': ["location", "park", "geopolitical", "road", "city"],  # " is a location name ."
                       'PER': ["person", "artist", "author", "athlete",
                               "scientist", "actor"],  # " is a person name ."
                       'ORG': ["organization", "government",
                               "league", "company", "team", ],
                       # " is an organization name"
                       'MISC': ["product", "event",
                                "war", "art", "music", "film",
                                # "law", "award",
                                #  "disease", "language",
                                ],  # " is another name"
                       }
        label2multiAns = {'O0': " is not a named entity .",  # " is not an entity ."
                         'O1': " is not an entity .",
                         'O2': " is not an entity name .",
                         'O3': " is a non entity name .",
                         'O4': " is a non entity.",

                         'LOC0': " is a location name .",
                         'LOC1': " is a park name .",
                         'LOC2': " is a geopolitical name .",
                         'LOC3': " is a road name .",
                         'LOC4': " is a city name .",

                         'PER0': " is a person name .",
                         'PER1': " is a artist name .",
                         'PER2': " is a author name .",
                         'PER3': " is a athlete name .",
                         'PER4': " is a scientist name .",
                         # 'PER5': " is a actor name .",

                         'ORG0': " is a organization name .",
                         'ORG1': " is a government name .",
                         'ORG2': " is a league name .",
                         'ORG3': " is a company name .",
                         'ORG4': " is a team name .",

                         'MISC0': " is a product name .",
                         'MISC1': " is a event name .",
                         'MISC2': " is a war name .",
                         'MISC3': " is a art name .",
                         # 'MISC4': " is a music name .",
                         'MISC4': " is a film name .",
                       }

        label2temp = {'O': " is not a named entity .",  # " is not an entity ."
                      'LOC': " is a location entity .",  # " is a location name ."
                      'PER': " is a person entity .",  # " is a person name ."
                      'ORG': " is an organization entity .",  # " is an organization name"
                      'MISC': " is an other entity .",  # " is another name"
                      }




        # templates = [" is a location entity .", " is a person entity .", " is an organization entity .",
        #              " is an other entity .", " is not a named entity ."]
        # idx2label = {0: 'LOC', 1: 'PER', 2: 'ORG', 3: 'MISC', 4: 'O'}
        # label2temp = {'LOC': " is a location entity .",
        #               'PER': " is a person entity .",
        #               'ORG': " is an organization entity .",
        #               'MISC': " is an other entity .",
        #               'O': " is not a named entity ."}


    elif 'note' in data_name:
        # https://catalog.ldc.upenn.edu/docs/LDC2011T03/OntoNotes-Release-4.0.pdf
        # templates = [" is a location entity .",
        #              " is a person entity .",
        #              " is an organization entity .",
        #              # " is an other entity .",
        #              " is not a named entity ."]
        # idx2label = {0: 'LOC',
        #              1: 'PER',
        #              2: 'ORG',
        #              # 3: 'MISC',
        #              4: 'O'
        #              }
        label2temp = {'LOC': " is a location entity .",
                      'PER': " is a person entity .",
                      'ORG': " is an organization entity .",
                      'TIME': " is a time entity .",
                      'PRODUCT': " is a product entity .",
                      'LAW': " is a law entity .",
                      'CARDINAL': " is a cardinal numeral entity .",
                      'WORK_OF_ART': " is a work of art entity .",
                      'FAC': " is a facility entity .",
                      'PERCENT': " is a percentage numeral entity .",
                      'MONEY': " is a money entity .",
                      'NORP': " is a nationality entity .",
                      'LANGUAGE': " is a language entity .",
                      'EVENT': " is an event entity .",
                      'GPE': " is a geography entity .",
                      'ORDINAL': " is an ordinal entity .",
                      'DATE': " is a date entity .",
                      'QUANTITY': " is a quantity entity .",
                      # 'MISC': " is an other entity .",
                      'O': " is not a named entity ."}

    idx2label = {}
    for i,label in enumerate(label2temp.keys()):
        idx2label[i] = label

    n_kword = 0
    kw_idx2label = {}
    for lab, kwords in label2vocab.items():
        n_kword2 = len(kwords)
        for i in range(n_kword, n_kword + n_kword2):
            kw_idx2label[i] = lab
        n_kword += n_kword2

    if 'mask' in args.temp_form:
        label2temp = label2temp_mask

    if args.temp_form == 'multi_ans':
        label2temp = label2multiAns
        n_class = len(label2multiAns)

        idx2label = {}
        for i,(lab, ans) in enumerate(label2multiAns.items()):
            if len(lab)>3:
                idx2label[i] =lab[:-1]
            else:
                idx2label[i] = lab[0] # for 'O'


    print('kw_idx2label: ',kw_idx2label)
    print('label2temp: ',label2temp)
    print('idx2label: ',idx2label)

    n_class = len(idx2label)
    if args.temp_form =='mask_prob':
        n_class = len(kw_idx2label)
    print('n_class: ',n_class)
    print('label2temp: ',label2temp)
    print('idx2label: ', idx2label)
    return idx2label,label2temp,n_class,t5temp,label2vocab,kw_idx2label


def read_data(fn, column_no=-1, delimiter=' '):
    # read token and tag for a CoNLL-format dataset
    word_sequences = list()
    tag_sequences = list()
    with codecs.open(fn, 'r', 'utf-8') as f:
        lines = f.readlines()
    curr_words = list()
    curr_tags = list()
    for k in range(len(lines)):
        line = lines[k].strip()
        if len(line) == 0 or line.startswith('-DOCSTART-'):  # new sentence or new document
            if len(curr_words) > 0:
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)
                curr_words = list()
                curr_tags = list()
            continue

        strings = line.split(delimiter)
        word = strings[0].strip()
        tag = strings[column_no].strip()  # be default, we take the last tag
        curr_words.append(word)
        curr_tags.append(tag)
        if k == len(lines) - 1:
            word_sequences.append(curr_words)
            tag_sequences.append(curr_tags)

    return word_sequences, tag_sequences

def enuID2listID(enu_ids):
    list_ids = []
    for enu_id in enu_ids:
        sid, eid = enu_id
        list_id = (sid, eid+1)
        list_ids.append(list_id)

    return list_ids


def get_pos_span_idxs(chunks):
    pos_span_idxs = []
    for chunk in chunks:
        chunk_type, sid, eid = chunk
        pos_span_idx = (sid, eid)
        pos_span_idxs.append(pos_span_idx)

    return pos_span_idxs


def id2label_test_MNER(args,
                        wseq,
                        chunks,
                        consider_span_idxs,
                        lang,
                        lang_label2ans,
                        lang_continue,
                        # label2temp,
                        pos_span_idxs,
                        sent_id,
                        ):
    # print('current consider lang: ', lang)
    # print('test....')
    lang_discrete = []
    for lang1, conti in lang_continue.items():
        # print('lang1, conti: ',lang1, conti)
        if conti ==0:
            lang_discrete.append(lang1)
    # print('lang_label2ans: ',lang_label2ans.keys())
    # print('lang: ',lang)
    span_forms = []
    span_prompts = []
    label2ans = lang_label2ans[lang]
    # sent = ' '.join(wseq)
    # print('lang_discrete: ',lang_discrete)
    if lang in lang_discrete:
        sent = ''.join(wseq)
    else:
        sent = ' '.join(wseq)

    for chunk in chunks:
        chunk_type, sid, eid = chunk
        span = wseq[sid:eid]
        span_form = (span, chunk_type, sid, eid)
        span_forms.append(span_form)
        new_span_idx = (sent_id, sid, eid)

        # for postive sample
        # if args.temp_form !='mask_prob':
        if lang in lang_discrete:
            span = ''.join(span)
        else:
            span = ' '.join(span)

        for tag, temp in label2ans.items():
            target = temp.replace('XX', span)
            span_prompt = {"sentence": sent,
                           "span": span,
                           "sent_span_idx": new_span_idx,
                           "true_tag": chunk_type,
                           "consi_tag": tag,
                           "target": target}
            # print('span_prompt: ',span_prompt)
            span_prompts.append(span_prompt)


    # negative span.
    for span_idx in consider_span_idxs:
        if span_idx not in pos_span_idxs:
            chunk_type = 'O'
            sid, eid = span_idx
            span = wseq[sid:eid]
            span_form = (span, chunk_type, sid, eid)
            span_forms.append(span_form)
            new_span_idx = (sent_id,sid, eid)
            if lang in lang_discrete:
                span = ''.join(span)
            else:
                span = ' '.join(span)

            # for negative tag
            for tag, temp in label2ans.items():

                target = temp.replace('XX', span)
                span_prompt = {"sentence": sent,
                               "span": span,
                               "sent_span_idx": new_span_idx,
                               "true_tag": chunk_type,
                               "consi_tag": tag,
                               "target": target}
                span_prompts.append(span_prompt)



    # return span_forms, select_span_prompts
    return span_forms,span_prompts


def prompt_sample_MNER(args,data_name, path_temp,dir,language, column_no=-1, delimiter=' ', ratio_neg=1.5, dtype='',max_span_width=8):

    # lang_label2ans, lang_idx2label, lang_nclass,continus_OrNot = {}, {}, {},{}
    lang_label2ans, lang_idx2label, lang_nclass, lang_continue = label2template_MNER(args,path_temp, data_name)



    span_prompts = {}
    # if os.path.isdir(fn): # dir == multilingual ner
    #     files = os.listdir(fn)
    #     for file in files:
    #         if 'test-' not in file or '_full' not in file:
    #             continue
    #         print('keep file name: ',file)
    #         path = fn + '/' + file
    #         lang = file.split('-')[-1].split('_')[0]
    if os.path.isdir(dir):
        # for language in languages:
        file = 'test-'+language+'_full.tsv'

        path = dir+file
        lang = language

        print('current considered lang: ', lang)
        print('file-path: ', path)

        # if lang not in span_prompts:
        #     span_prompts[lang] = [ ]
        span_prompts = []

        word_seqs, tag_seqs = read_data(path, column_no=column_no, delimiter=delimiter)

        # begin{select 1/10 samples}
        if args.use_small_data:
            wt = list(zip(word_seqs, tag_seqs))
            random.shuffle(wt)
            word_seqs1, tag_seqs1 = zip(*wt)
            word_seqs, tag_seqs = word_seqs1[:10], tag_seqs1[:10]
        # end{select 1/10 samples}

        print('test on %d samples: '%len(word_seqs))
        print()




        for k, (wseq, tseq) in enumerate(zip(word_seqs, tag_seqs)):
            sent_id = k
            all_span_idxs = enumerate_spans(wseq, max_span_width=max_span_width, offset=0)
            all_span_idxs = enuID2listID(all_span_idxs)
            chunks = get_chunks(tseq)
            pos_span_idxs = get_pos_span_idxs(chunks)

            if dtype == 'test':
                consider_span_idxs = all_span_idxs  # all the negative span is considered for test/eval phrase,
                span_forms, span_prompt = id2label_test_MNER(args,
                                                        wseq,
                                                        chunks,
                                                        consider_span_idxs,
                                                        lang,
                                                        lang_label2ans,
                                                        lang_continue,
                                                        # label2temp,
                                                        pos_span_idxs,
                                                        sent_id)
                span_prompts += span_prompt
    print('lanbel description: ')
    print('lang_label2ans: ', lang_label2ans[language])
    print()
    print('idx2label: ')
    print(lang_idx2label[language])
    print()
    print('lang_nclass: ', lang_nclass[language])
    print()

    for i, span_prompt in enumerate(span_prompts):
        print(span_prompt)
        if i > 5:
            break
    print()
    # print('fn: ', span_prompts)
    return span_prompts, lang_nclass, lang_idx2label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning tagger using neural networks')
    parser.add_argument('--data_name', default='panx',
                        help='dataset name')
    parser.add_argument('--path_temp', default='../data/ner/multilingual_NER_template.xlsx',
                        help='dataset name')
    parser.add_argument('--dir', default="../data/ner/panx-test/",  #
                        help='dataset path')
    parser.add_argument('--language', default="en",  #
                        help='languages considered.')

    parser.add_argument('--column_no', type=int, default=1,
                        help='NER column')
    parser.add_argument('--delimiter', default='\t',
                        help='NER column')
    parser.add_argument('--ratio_neg', type=int, default=1.5,
                        help='NER column')
    parser.add_argument('--n_class', type=int, default=4,
                        help='the nmber of NER classes..')
    parser.add_argument('--dtype', default='test',
                        help='choice is [train, test]')
    parser.add_argument('--max_span_width', type=int, default=6,
                        help='NER column')
    args = parser.parse_args()

    # print('args.languages:',args.languages)
    args=''
    span_prompts, lang_nclass, lang_idx2label = prompt_sample_MNER(args=args,
                                                                   data_name=args.data_name,
                                                                   path_temp=args.path_temp,
                                                                   dir=args.dir,
                                                                   language=args.language,
                                                                   column_no=1,
                                                                   delimiter='\t',
                                                                   ratio_neg=1.5,
                                                                   dtype='test',
                                                                   max_span_width=6)

    if os.path.isdir(args.dir): # dir == multilingual ner
        files = os.listdir(args.dir)
        for file in files:
            if 'test-' not in file or '_full' not in file:
                continue
            print('keep file name: ',file)
            path = fn + '/' + file
            lang = file.split('-')[-1].split('_')[0]
            print('language: ',lang)





    #
    # data_name = 'conll03'
    # fn = '../data/ner/conll2003txt/test_sample.txt'
    #
    # # data_name = 'notenw'
    # # fn = '../data/ner/notenw/test.txt'
    # # prompt_sample(data_name, fn, column_no=3, delimiter=' ', ratio_neg=1.5, dtype='test', max_span_width=6)
    #
    # file_path = '../data/ner/multilingual_NER_template.xlsx'
    # # read_MNER_template(file_path)
    # data_name = 'conll03'
    # args =''
    # # label2template_MNER(args, file_path, data_name)
    #
    # dir = "../data/ner/panx-test/"
    # languages = ['en','zh']
    # # print('fn: ',span_prompts[lang] )
