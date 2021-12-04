from omneval.registry import register_metrics
from typing import List
from eaas import Config
from eaas import Client
import pdb

class BaseMetrics:
    def __init__(self):
        pass

    def compute(self, predictions, references):
        raise NotImplementedError


class EaasMetrics(BaseMetrics):
    def __init__(self, metrics, config=Config(), client=Client()):
        super(EaasMetrics).__init__()
        self.config = config
        self.client = client
        self.metrics = metrics
        self.client.load_config(self.config)

    def compute(self, predictions, references, source=None):
        if source is None:
            inputs = [{'source': '', 'references': ref if isinstance(ref, list) else [ref], 'hypothesis': pred}
                      for pred, ref in zip(predictions, references)]
        else:
            inputs = [{'source': src, 'references': ref if isinstance(ref, list) else [ref], 'hypothesis': pred}
                      for pred, ref, src in zip(predictions, references, source)]
        res = self.client.score(inputs, task="sum", metrics=self.metrics, lang='en')
        return self.postprocess(res)

    def postprocess(self, res):
        return res


@register_metrics('rouge1')
class Rouge1(EaasMetrics):
    def __init__(self, metrics=('rouge1',), config=Config(), client=Client()):
        super().__init__(metrics, config, client)

    def postprocess(self, res):
        return {'rouge1': res['corpus_level']['corpus_rouge_1']}


@register_metrics('rouge2')
class Rouge2(EaasMetrics):
    def __init__(self, metrics=('rouge2',), config=Config(), client=Client()):
        super(Rouge2).__init__(metrics, config, client)

    def postprocess(self, res):
        return {'rouge2': res['corpus_level']['corpus_rouge_2']}

@register_metrics('f1-ner')
class NERF1(BaseMetrics):
    def compute(self, predictions, references):
        pred_ress = []
        for i in range(len(references)):
            pred = {
                'span_idx': predictions['span_idx'][i],
                'true_tag': references[i],
                'pred_tag': predictions['predictions'][i],
                'score': predictions['score'][i],
                'sentence_id': predictions['sentence_id'][i]
            }
            pred_ress.append(pred)
        f1, p, r, correct_preds, total_preds, total_correct = evaluate_metric(pred_ress)
        return {'f1': f1, 'precision': p, 'recall': r}



def evaluate_metric(pred_ress):
    sentid2chunk_true = {}
    sentid2chunk_score_pred = {}
    idx2score = {}
    true_chunks = []
    for pred_res in pred_ress:
        span_idx = tuple(pred_res['span_idx'])
        true_tag = pred_res['true_tag']
        pred_tag = pred_res['pred_tag']
        score = pred_res['score']
        sent_id = pred_res['sentence_id']
        sid, eid = span_idx

        # get the true tag in sentence-level
        if sent_id not in sentid2chunk_true:
            sentid2chunk_true[sent_id] = []
        if true_tag != 'O':
            true_chunk = (sent_id, sid, eid, true_tag)
            sentid2chunk_true[sent_id].append(true_chunk)
            true_chunks.append(true_chunk)
        if sent_id not in sentid2chunk_score_pred:
            sentid2chunk_score_pred[sent_id] = {}
        if sid not in sentid2chunk_score_pred[sent_id]:
            sentid2chunk_score_pred[sent_id][sid] = []

        # if pred_tag != 'O':
        pred_chunk_score = (sent_id, sid, eid, pred_tag, score)
        # sentid2chunk_score_pred[sent_id].append(pred_chunk_score)
        sentid2chunk_score_pred[sent_id][sid].append(pred_chunk_score)
        span_idx_score = (sent_id, sid, eid, pred_tag)
        if span_idx_score not in idx2score:
            idx2score[span_idx_score] = score


    # prune the overlaping span,
    pred_chunks = []
    for sentid, spchunks in sentid2chunk_score_pred.items():
        for sid, pchunks in spchunks.items():
            # # begin{not prune...}
            # for pchunk in pchunks:
            # 	sent_id, sid, eid, pred_tag, score = pchunk
            # 	if pred_tag != 'O':
            # 		chunk_new = (sent_id, sid, eid, pred_tag)
            # 		pred_chunks.append(chunk_new)
            # # end{not prune...}

            # begin{half prune...}
            kchunk = half_prune(pchunks)
            ksent_id, ksid, keid, kptag = kchunk
            if kptag != 'O':
                pred_chunks.append(kchunk)
        # end{half prune...}

    true_chunks = set(true_chunks)
    pred_chunks = set(pred_chunks)
    f1, p, r, correct_preds, total_preds, total_correct = chunk_eval(true_chunks, pred_chunks)

    print('f1, p, r, correct_preds, total_preds, total_correct:')
    print(f1, p, r, correct_preds, total_preds, total_correct)
    return f1, p, r, correct_preds, total_preds, total_correct


def half_prune(pchunks):
    scores = []
    ptaggs = []
    poss = []
    for pchunk in pchunks:
        # print('pchunk: ',pchunk)
        sent_id, sid, eid, pred_tag, score = pchunk
        scores.append(score)
        ptaggs.append(pred_tag)
        pos = (sent_id, sid, eid)
        poss.append(pos)
    max_score = max(scores)
    max_score_idx = scores.index(max_score)
    # print('max_score: ', max_score)
    # print('max_score_idx: ',max_score_idx)
    kpos = poss[max_score_idx]
    ksent_id, ksid, keid = kpos
    kptag = ptaggs[max_score_idx]
    # print('kpos: ',kpos)
    # print('kptag: ', kptag)
    kchunk = (ksent_id, ksid, keid, kptag)

    return kchunk


def chunk_eval(true_chunks, pred_chunks):
    correct_preds, total_correct, total_preds = 0., 0., 0.
    correct_preds += len(set(true_chunks) & set(pred_chunks))
    total_preds += len(pred_chunks)
    total_correct += len(true_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    # acc = np.mean(accs)
    cp = correct_preds
    tp = total_preds
    return f1, p, r, correct_preds, total_preds, total_correct
