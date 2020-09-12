import argparse
import urllib3
import requests
import os
import json
import time
import urllib
import socket
import jsonlines

filePath = '/data3/private/fengtao/Projects/cqr_t5/bert_ranking/runs/reranked/'


def readname(filePath):
    names = os.listdir(filePath)
    return names


parser = argparse.ArgumentParser()

file_list = readname(filePath)

type ='space'
for name in file_list:
    print('-------------')
    print(name)
    file_dir = filePath + name
    os.system('/data1/private/yushi/conv-coref/trec_eval/trec_eval -m ndcg_cut.3 /data1/private/yushi/conv-coref/bert_ranking/CAST2019.qrels ' + file_dir)
    # os.system('python3 mrr10.py --inputFile=' + name + ' --split=' + type)
