import argparse
import pickle
import csv
import os
import re
import json
import random

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import *

import pytrec_eval
# import torchvision
# from torchvision import transforms

epoch = 1
debug = False
test = True
train = False
finetuning = False
lock = False
task = 'trec_12'
eval_directly = True

train_batch_size = 8
eval_batch_size = 8

tmp_best_acc = 0.0
from_idx = 0
n_gpu = 1
# g_acc = 4

fold = '4'
#train_file = task + '/cross_validate/train_09_' + fold + '.txt'
# <query> \t <positive doc> \t <negative doc>
train_file = '/data1/private/yushi/conv-coref/ms_marco/triples.train.small.tsv'  # not used
#dev_file = task + '/cross_validate/dev_09_' + fold + '.txt'
# <qid> \t <did> \t <q_ids> \t <d_ids>
dev_file = '../ms_marco/dev.bm25.top100.txt'  # not used

# <qid> 0 <pid> 1
dev_label = '/data1/private/yushi/conv-coref/bert_ranking/CAST2019.qrels'  # not used (for MRR)
dev_bert_vocab = '/data1/private/yushi/conv-coref/ms_marco/bert-base-uncased-vocab.txt'  # not used
save_file = '/data1/private/yushi/conv-coref/models/bert_base'  # do not change
# tmp_file = '../models/bert_base_tmp'  # + fold
#pretrained_model = '../bert-large/class.bin'
# pretrained_model = '../models/multi_trans_c10_new_bert_rb_'
out_trec = '/data3/private/fengtao/Projects/cqr_t5/bert_ranking/runs/reranked/'  # change this
dev_file_conv = '/data3/private/fengtao/Projects/cqr_t5/bert_ranking/runs/'  # change this

regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')

# ft
def readname():
  filePath = '/data3/private/fengtao/Projects/cqr_t5/bert_ranking/runs/'
  names = os.listdir(filePath)
  file_list = []
  for n in names:
    if n[-1] == 't':
      file_list.append(n)
  return file_list

def raw2tok(s):
  lst = regex_multi_space.sub(
    ' ', regex_drop_char.sub(' ', s.lower())).strip().split()
  return lst


class trainFeatures(object):
  def __init__(self, p_input_ids, p_input_mask, p_segment_ids, n_input_ids, n_input_mask, n_segment_ids):
    self.p_input_ids = p_input_ids
    self.p_input_mask = p_input_mask
    self.p_segment_ids = p_segment_ids
    self.n_input_ids = n_input_ids
    self.n_input_mask = n_input_mask
    self.n_segment_ids = n_segment_ids


class devFeatures(object):
  def __init__(self, query_id, doc_id, qd_score, d_input_ids, d_input_mask, d_segment_ids):
    self.query_id = query_id
    self.doc_id = doc_id
    self.qd_score = qd_score
    self.d_input_ids = d_input_ids
    self.d_input_mask = d_input_mask
    self.d_segment_ids = d_segment_ids


class BertForRanking(BertPreTrainedModel):
  def __init__(self, config):
    super(BertForRanking, self).__init__(config)
    self.bert = BertModel(config)
    self.dense = nn.Linear(config.hidden_size, 2)

    self.init_weights()

  def forward(self, inst, tok, mask, init_state=False):
    output = self.bert(inst, token_type_ids=tok, attention_mask=mask)
    if init_state:
      return Variable(output[0][:, 0, :].squeeze(-1), requires_grad=True)
    final_res = self.dense(output[1])[:, 1]
    # final_res = self.dense(output[0][:, 0, :].squeeze(-1)).squeeze(-1)

    return final_res, output[1]


def pack_bert_seq(q_tokens, p_tokens, tokenizer, max_seq_length):
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in q_tokens:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  for token in p_tokens:
    tokens.append(token)
    segment_ids.append(1)
  tokens.append("[SEP]")
  segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  return input_ids, input_mask, segment_ids



def read_clueweb_to_features(input_file, tokenizer, is_training=True, full=False):
  max_seq_length = 384
  max_query_length = 64
  with open(input_file, 'r') as reader:
    #source = csv.reader((line.replace('\0','') for line in reader), delimiter='\t')
    cnt = 0
    features = []
    for line in reader:
      cnt += 1
      if cnt % 10000 == 0:
        print(cnt)
      if is_training and len(features) >= 500000:  # 100000 q-d pairs
        break
      if not is_training and len(features) >= 10000 and not full:  # 100 queries
        break
      s = line.strip('\n').split('\t')
      # if debug and len(features) >= 28000:# debug
      # 	break
      # if is_training and not lock:
      # 	if cnt <= from_idx:
      # 		continue
      # 	if cnt > from_idx + 500000:
      # 		break

      # train
      if is_training:
        q_tokens = tokenizer.tokenize(s[0])
        max_query_length = max_seq_length - 1
        if len(q_tokens) > max_query_length:
          q_tokens = q_tokens[:max_query_length]

        max_doc_length = max_seq_length - len(q_tokens) - 3
        p_tokens = tokenizer.tokenize(s[1])
        if len(p_tokens) > max_doc_length:
          p_tokens = p_tokens[:max_doc_length]
        n_tokens = tokenizer.tokenize(s[2])
        if len(n_tokens) > max_doc_length:
          n_tokens = n_tokens[:max_doc_length]

        p_input_ids, p_input_mask, p_segment_ids = pack_bert_seq(
          q_tokens, p_tokens, tokenizer, max_seq_length)
        n_input_ids, n_input_mask, n_segment_ids = pack_bert_seq(
          q_tokens, n_tokens, tokenizer, max_seq_length)

        features.append(trainFeatures(
          p_input_ids=p_input_ids,
          p_input_mask=p_input_mask,
          p_segment_ids=p_segment_ids,
          n_input_ids=n_input_ids,
          n_input_mask=n_input_mask,
          n_segment_ids=n_segment_ids))
      # dev
      else:
        with open(dev_bert_vocab, 'r') as f:
          vocab = f.readlines()

        def convert_ids_to_tokens(ids: []) -> []:  # another one
          tokens = []
          for id_ in ids:
            tokens.append(vocab[int(id_)][:-1])
          return tokens

        # query_toks_raw = raw2tok(s[2])
        # doc_toks_raw = raw2tok(s[3])

        qd_score = 0
        query_id = s[0]
        doc_id = s[1]

        # s[2] = ' '.join(query_toks_raw)
        # s[3] = ' '.join(doc_toks_raw[:150])

        q_tokens = tokenizer.tokenize(s[2])
        # q_tokens = convert_ids_to_tokens(s[2].split(','))
        # q_tokens_joined = ' '.join(q_tokens)  # join together!
        # q_tokens_joined = q_tokens_joined.replace(' ##', '')
        # q_tokens_joined = q_tokens_joined.replace('##', '')
        # q_tokens = tokenizer.tokenize(q_tokens_joined)

        if len(q_tokens) > max_query_length:
          q_tokens = q_tokens[:max_query_length]

        max_doc_length = max_seq_length - len(q_tokens) - 3

        d_tokens = tokenizer.tokenize(s[3])
        # d_tokens = convert_ids_to_tokens(s[3].split(','))
        # d_tokens_joined = ' '.join(d_tokens)
        # d_tokens_joined = d_tokens_joined.replace(' ##', '')
        # d_tokens_joined = d_tokens_joined.replace('##', '')
        # d_tokens = tokenizer.tokenize(d_tokens_joined)

        if len(d_tokens) > max_doc_length:
          d_tokens = d_tokens[:max_doc_length]

        d_input_ids, d_input_mask, d_segment_ids = pack_bert_seq(
          q_tokens, d_tokens, tokenizer, max_seq_length)

        features.append(devFeatures(
          query_id=query_id,
          doc_id=doc_id,
          qd_score=qd_score,
          d_input_ids=d_input_ids,
          d_input_mask=d_input_mask,
          d_segment_ids=d_segment_ids))

  return features


def trainDataLoader(features, batch_size, shuffle=True):
  n_samples = len(features)
  idx = np.arange(n_samples)
  if shuffle:
    np.random.shuffle(idx)

  for start_idx in range(0, n_samples, batch_size):
    batch_idx = idx[start_idx:start_idx+batch_size]

    p_input_ids = torch.tensor(
      [features[i].p_input_ids for i in batch_idx], dtype=torch.long)
    p_input_mask = torch.tensor(
      [features[i].p_input_mask for i in batch_idx], dtype=torch.long)
    p_segment_ids = torch.tensor(
      [features[i].p_segment_ids for i in batch_idx], dtype=torch.long)
    n_input_ids = torch.tensor(
      [features[i].n_input_ids for i in batch_idx], dtype=torch.long)
    n_input_mask = torch.tensor(
      [features[i].n_input_mask for i in batch_idx], dtype=torch.long)
    n_segment_ids = torch.tensor(
      [features[i].n_segment_ids for i in batch_idx], dtype=torch.long)

    batch = (p_input_ids, p_input_mask, p_segment_ids,
         n_input_ids, n_input_mask, n_segment_ids)
    yield batch
  return


def devDataLoader(features, batch_size):
  n_samples = len(features)
  idx = np.arange(n_samples)

  for start_idx in range(0, n_samples, batch_size):
    batch_idx = idx[start_idx:start_idx+batch_size]

    query_id = [features[i].query_id for i in batch_idx]
    doc_id = [features[i].doc_id for i in batch_idx]
    qd_score = [features[i].qd_score for i in batch_idx]
    d_input_ids = torch.tensor(
      [features[i].d_input_ids for i in batch_idx], dtype=torch.long)
    d_input_mask = torch.tensor(
      [features[i].d_input_mask for i in batch_idx], dtype=torch.long)
    d_segment_ids = torch.tensor(
      [features[i].d_segment_ids for i in batch_idx], dtype=torch.long)

    batch = (query_id, doc_id, qd_score, d_input_ids,
         d_input_mask, d_segment_ids)
    yield batch
  return


related_docs = {}


def read_qrels():
  with open(dev_label, 'r') as f:
    for line in f:
      s = line.split(' ')
      qid = s[0]
      pid = s[2]
      if qid not in related_docs:
        related_docs[qid] = [pid]
      else:
        related_docs[qid].append(pid)


def set_seed():
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  if n_gpu > 0:
    torch.cuda.manual_seed_all(42)


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # device = torch.device("cpu")
  # set_seed()


  config = BertConfig.from_pretrained(
    '/data1/private/yushi/conv-coref/bert_dir/bert-base/bert_config.json')
  tokenizer = BertTokenizer.from_pretrained(
      '/data1/private/yushi/conv-coref/bert_dir/bert-base/bert-base-uncased-vocab.txt')
  model = BertForRanking.from_pretrained(
      '/data1/private/yushi/conv-coref/bert_dir/bert-base/pytorch_model.bin', config=config)
  # load model
  if eval_directly:
    state_dict=torch.load(save_file)
    model.load_state_dict(state_dict)
  model.to(device)

  # train data

  read_qrels()

  if n_gpu > 1:
    model = torch.nn.DataParallel(model)



  if test:
    file_list = readname()
    for name in file_list:
      # test data
      # params = torch.load(tmp_file)
      # model.module.load_state_dict(params)
      test_features = read_clueweb_to_features(
        dev_file_conv + name, tokenizer, is_training=False, full=True)
      test_data = devDataLoader(test_features, eval_batch_size)

      # test
      rst_dict = {}
      for s, batch in enumerate(test_data):

        if s % 1000 == 0:
          print(s)
        query_id = batch[0]
        doc_id = batch[1]
        qd_score = batch[2]
        batch = tuple(t.to(device) for t in batch[3:])
        (d_input_ids, d_input_mask, d_segment_ids) = batch

        with torch.no_grad():
          doc_scores, _ = model(d_input_ids, d_segment_ids, d_input_mask)
        d_scores = doc_scores.detach().cpu().tolist()

        for (q_id, d_id, qd_s, d_s) in zip(query_id, doc_id, qd_score, d_scores):
          if q_id in rst_dict:
            rst_dict[q_id].append((qd_s, d_s, d_id))
          else:
            rst_dict[q_id] = [(qd_s, d_s, d_id)]

      average_mrr10 = 0
      sum_mrr10 = 0
      cnt = 0
      with open(out_trec +'reranked_'+name, 'w') as writer:
        for q_id, scores in rst_dict.items():
          cnt += 1
          # sort by doc score
          res = sorted(scores, key=lambda x: x[1], reverse=True)
          # mrr10 = 0
          # for index, trip in enumerate(res[0:10]):
          #   sc = trip[1]
          #   pid = trip[2]
          #   if pid in related_docs[q_id]:
          #     mrr10 = 1 / (index + 1)
          #     break
          # sum_mrr10 += mrr10
          for rank, value in enumerate(res):
            writer.write(
              q_id+' '+'Q0'+' '+str(value[2])+' '+str(rank+1)+' '+str(value[1])+' '+'BERT-base'+'\n')
      # average_mrr10 = sum_mrr10 / cnt
      # print('mrr@10: ' + str(average_mrr10))
      # if task == 'trec_12':
      #   res = os.popen(f'./gdeval.pl {dev_label} ' + out_trec)
      #   # for r in res:
      #   #   pass
      #   # scores = r.strip('\n').split(',')
      #   # ndcg_20 = float(scores[2])
      #   # err_20 = float(scores[3])
      #   # print('ndcg_20: ' + str(ndcg_20))
      #   # print('err_20: ' + str(err_20))
      #   print(res.read())
      # # print('mrr@10: ' + str(average_mrr10))





if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'
  main()
