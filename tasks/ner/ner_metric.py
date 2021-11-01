# -*- coding: utf-8 -*
import codecs
import numpy as np
from collections import Counter
import os
import pickle
import random
import json

def get_chunk_type(tok):
	"""
	Args:
		tok: id of token, ex 4
		idx_to_tag: dictionary {4: "B-PER", ...}
	Returns:
		tuple: "B", "PER"
	"""
	# tag_name = idx_to_tag[tok]
	tag_class = tok.split('-')[0]
	# tag_type = tok.split('-')[-1]
	tag_type = '-'.join(tok.split('-')[1:])
	return tag_class, tag_type

def get_chunks(seq):
	"""
	tags:dic{'per':1,....}
	Args:
		seq: [4, 4, 0, 0, ...] sequence of labels
		tags: dict["O"] = 4
	Returns:
		list of (chunk_type, chunk_start, chunk_end)

	Example:
		seq = [4, 5, 0, 3]
		tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
		result = [("PER", 0, 2), ("LOC", 3, 4)]
	"""
	default = 'O'
	# idx_to_tag = {idx: tag for tag, idx in tags.items()}
	chunks = []
	chunk_type, chunk_start = None, None
	for i, tok in enumerate(seq):
		#End of a chunk 1
		if tok == default and chunk_type is not None:
			# Add a chunk.
			chunk = (chunk_type, chunk_start, i)
			chunks.append(chunk)
			chunk_type, chunk_start = None, None

		# End of a chunk + start of a chunk!
		elif tok != default:
			tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
			if chunk_type is None:
				chunk_type, chunk_start = tok_chunk_type, i
			elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
				chunk = (chunk_type, chunk_start, i)
				chunks.append(chunk)
				chunk_type, chunk_start = tok_chunk_type, i
		else:
			pass
	# end condition
	if chunk_type is not None:
		chunk = (chunk_type, chunk_start, len(seq))
		chunks.append(chunk)

	return chunks


def has_overlapping(idx1, idx2):
    overlapping = True
    if (idx1[1] > idx2[2] or idx2[1] > idx1[2]):
        overlapping = False
    return overlapping

def evaluate_metric(pred_ress):
	sentid2chunk_true = {}
	sentid2chunk_score_pred = {}
	idx2score = {}
	true_chunks = []
	pos2span_dic = {}
	for pred_res in pred_ress:
		span = pred_res['span']
		sent_span_idx = tuple(pred_res['sent_span_idx'])
		true_tag = pred_res['true_tag']
		pred_tag = pred_res['pred_tag']
		score = pred_res['score']
		sent_id, sid, eid = sent_span_idx

		# get the true tag in sentence-level
		if sent_id not in sentid2chunk_true:
			sentid2chunk_true[sent_id] = []
		if true_tag!='O':
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

		pos2span_dic[sent_span_idx] = span


	# prune the overlaping span,
	pred_chunks = []
	for sentid,spchunks in sentid2chunk_score_pred.items():
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
			if kptag !='O':
				pred_chunks.append(kchunk)
			# end{half prune...}

	# print_pred(true_chunks, pred_chunks, pos2span_dic)
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

def write_res(pred_ress,fn_write):
	with open(fn_write, mode='w', encoding='utf-8') as f:
		json.dump(pred_ress, f)

def print_pred(true_chunks,pred_chunks,pos2span_dic):

	chunk_dic = {}
	for true_chunk in true_chunks:
		sent_id, sid, eid, true_tag = true_chunk
		# print('input_texts: ',input_texts)
		pos = (sent_id, sid, eid)
		span = pos2span_dic[pos]
		if sent_id not in chunk_dic:
			chunk_dic[sent_id] = {}
		pos = (sid, eid)
		if pos not in chunk_dic[sent_id]:
			chunk_dic[sent_id][pos] = str(pos)+',  '+' '.join(span)+',  ' + true_tag +',  '
		else:
			print('error! one span only have one label!')

	for pred_chunk in pred_chunks:
		sent_id, sid, eid, pred_tag = pred_chunk
		ppos = (sent_id, sid, eid)
		span = pos2span_dic[ppos]
		if sent_id not in chunk_dic:
			chunk_dic[sent_id] = {}
		pos = (sid, eid)
		if pos not in chunk_dic[sent_id]:
			chunk_dic[sent_id][pos] = str(pos)+',  '+' '.join(span)+',  O,  ' + pred_tag +'\t'
		else:
			chunk_dic[sent_id][pos] +=  pred_tag +'\t'

	for sent_id,chunk in chunk_dic.items():
		for pos, string in chunk.items():
			if '\t' not in string:
				str1 = string+'O'+'\t'
				chunk_dic[sent_id][pos] = str1


	for sent_id,chunk in chunk_dic.items():
		# string = str(sent_id)+': '
		string = ""
		for pos, prediction in chunk.items():
			string+=prediction
		print(sent_id, string)

	# # fn_write = 'write.txt'
	# fwrite = open(fn_write, 'w+')
	# for sent_id,chunk in chunk_dic.items():
	# 	# string = str(sent_id)+': '
	# 	string = ""
	# 	for pos, prediction in chunk.items():
	# 		string+=prediction
	# 	print(sent_id, string)
	# 	fwrite.write(string+'\n')





def read_res_eval(fn_res):
	true_chunks = []
	pred_chunks = []
	with codecs.open(fn_res, 'r', 'utf-8') as f:
		lines = f.readlines()
	for k,line in enumerate(lines):
		if line.strip() !='':
			ress = line.split('\t')

			for res in ress:
				if len(res.strip()) ==0:
					continue
				elem = res.split(',  ')
				pos = elem[0]
				span = elem[1]
				ttag = elem[2]
				ptag = elem[3]
				sid = int(pos.split(', ')[0][1:])
				eid = int(pos.split(', ')[1][:-1])
				# print('pos: ',pos)
				# print(sid, eid)
				if ttag != 'O':
					true_chunk = (k, sid, eid, ttag)
					true_chunks.append(true_chunk)
				if ptag !='O':
					pred_chunk = (k, sid, eid, ptag)
					pred_chunks.append(pred_chunk)

	f1, p, r, correct_preds, total_preds, total_correct = chunk_eval(true_chunks, pred_chunks)
	print('f1, p, r, correct_preds, total_preds, total_correct:')
	print(f1, p, r, correct_preds, total_preds, total_correct)







def prune_overlap_span(chunk_score_preds,idx2score):
	kidxs = []
	didxs = []

	for j,cs in enumerate(chunk_score_preds):
		(sent_id1, sid1, eid1, pred_tag1, score1) = cs
		idx1 = (sent_id1, sid1, eid1,pred_tag1)
		kidx = idx1
		kidx1 = True

		for k in range(j+1,len(chunk_score_preds)):
			(sent_id2, sid2, eid2, pred_tag2, score2) = chunk_score_preds[k]
			idx2 = (sent_id2, sid2,eid2,pred_tag2)
			print('idx1: ',idx1)
			print('idx2: ',idx2)
			isoverlapp = has_overlapping(idx1, idx2)
			print('isoverlapp: ',isoverlapp)
			if isoverlapp:
				print('test...')
				print('cs: ',cs)
				print('chunk_score_preds[k]:',chunk_score_preds[k])
				if score1 < score2:
					kidx1 = False
					didxs.append(kidx1)
				elif score1 == score2:
					# print('idx1[2]: ', idx1[2])
					# print('idx1[1]: ', idx1[1])
					len1 = idx1[2] - idx1[1]
					len2 = idx2[2] - idx2[1]
					# print("len1, len2: ", len1, len2)
					if len1 < len2:
						kidx1 = False
						didxs.append(kidx1)
			# if kidx1 ==False:

			print('didxs: ',didxs)
			print()
		if kidx1:
			print('phrase 2...')
			flag = True
			for idx in kidxs:
				isoverlap = has_overlapping(idx1, idx)
				print('idx1 2: ',idx1)
				print('idx: ',idx)
				print('isoverlap22: ',isoverlap)
				if isoverlap:
					flag = False
					score1 = idx2score[idx1]
					score2 = idx2score[idx]
					if score1 > score2:  # del the keept idex
						kidxs.remove(idx)
						kidxs.append(idx1)
					break
			print('kidxs: ',kidxs)
			if flag == True:
				kidxs.append(idx1)

	# last_idx = (chunk_score_preds[-1][0], chunk_score_preds[-1][1], chunk_score_preds[-1][2], chunk_score_preds[-1][3])
	# if len(didxs) == 0:
	# 	kidxs.append(last_idx)
	# else:
	# 	if last_idx[-1] not in didxs:
	# 		kidxs.append(last_idx)

	return kidxs


if __name__ == '__main__':
	# pred_ress = [{'span':'China', 'sent_span_idx':(0,1,2),'true_tag':'O','pred_tag':'LOC', 'score':0.8},
	# 			 {'span': 'China Bank', 'sent_span_idx': (0, 1, 3), 'true_tag': 'LOC', 'pred_tag': 'LOC', 'score': 0.85},
	# 			 {'span': 'Singapore Bank ', 'sent_span_idx': (0, 5, 7), 'true_tag': 'ORG', 'pred_tag': 'ORG', 'score': 0.9}
	# 			 ]
	# f1, p, r, correct_preds, total_preds, total_correct = evaluate_metric(pred_ress)

	# fn_res = '../results/ner/conll03_bartCNN126_msp6_TempNERScore_mask_26680974.txt'
	# read_res_eval(fn_res)

	# fn_read ='../results/ner/conll03_XX_94388167.json'
	# with open(fn_read, mode='r', encoding='utf-8') as f:
	# 	pred_ress = json.load(f)
	#
	# evaluate_metric(pred_ress)


	fn_ress = ["../results/ner/conll03_bartlarge_maxspan6_TempNERScore_gene_NotMinus4_00088741.json",
				"../results/ner/conll03_bartCNN126_maxspan6_TempNERScore_gene_NotMinus4_65943287.json",
		"../results/ner/conll03_bartCNN126_maxspan6_TempNERScore_mask_newTemp_55774946.json",
		"../results/ner/conll03_bartCNN126_maxspan6_TempNERScore_mask_56983066.json",
		"../results/ner/conll03_bartlarge_maxspan6_TempNERScore_gene_ALLdecoderInputIDS_78159076.json",
		"../results/ner/conll03_bartCNN126_maxspan6_TempNERScore_gene_ALLdecoderInputIDS_33803250.json",
		"../results/ner/conll03_bartCNN126_maxspan6_bartScore_mask_66371082.json",
		"../results/ner/conll03_bartCNN126_maxspan6_TempNERScore_gene_newTemp_45702775.json",
		"../results/ner/conll03_bartCNN126_maxspan6_bartScore_gene_48679733.json",
		"../results/ner/conll03_bartCNN126_maxspan6_TempNERScore_gene_95022982.json"]
	fn_ress= ['../results/ner/conll03_bartCNN126_maxspan6_TempNERScore_mask_33912066.json',
			  "../results/ner/conll03_bartCNN126_maxspan6_bartScore_mask_49758046.json",
			  "../results/ner/conll03_bartCNN126_maxspan6_bartScore_gene_55231961.json",
			  "../results/ner/conll03_bartCNN126_maxspan6_TempNERScore_gene_90206629.json"]
	for fn_res in fn_ress:
		with open(fn_res, mode='r', encoding='utf-8') as f:
			pred_ress = json.load(f)
		print("fn_res: ",fn_res)
		evaluate_metric(pred_ress)
		print()





