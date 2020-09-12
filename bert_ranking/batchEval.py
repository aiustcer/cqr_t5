import argparse
import urllib3
import requests
import os
import json
import time
import urllib
import socket
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default="batchE.txt", help="input jsonlines file")
args = parser.parse_args()
with open(args.input_file, "r+", encoding="utf8") as input:
    for line in input:
        s = line.strip().split(' ')
        name, type = s[0], s[1]

        print(name)
        os.system('/data1/private/yushi/conv-coref/trec_eval/trec_eval -m ndcg_cut.3 /data1/private/yushi/conv-coref/bert_ranking/CAST2019.qrels ' + name)
        # os.system('python3 mrr10.py --inputFile=' + name + ' --split=' + type)
        print('\n\n')