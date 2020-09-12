import argparse
import urllib3
import requests
import os
import json
import time
import urllib
import socket
import jsonlines

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

from pyserini.search import pysearch
import re


cop = re.compile("[^ ^_^a-z^A-Z^0-9]")
dr = re.compile(r'<[^>]+>',re.S)
def removePunctuation(text):
    text = cop.sub('',text)
    return text.strip()

def removeHtmlTag(text):
    text = dr.sub('',text)
    return text.strip()

http = urllib3.PoolManager()


print('begin to initial model')
searcher = pysearch.SimpleSearcher('/data1/private/yangjingqin/anserini/lucene-index-msmarco-CAR17')
searcher.set_bm25_similarity(0.9, 0.4)
print('searcher has been init')
parser = argparse.ArgumentParser()
parser.add_argument('--search_index', default="output", help="target or output")
parser.add_argument('--input_file', default="batch.txt", help="input jsonlines file")
parser.add_argument('--hit_size', default=100, help="number of hits")
parser.add_argument('--stopwords', default='indri.txt', help="list of stopwords, if not use, please set it NONE")
args = parser.parse_args()

print(args.stopwords)
if args.stopwords != 'NONE':
    stopwords = {}.fromkeys([ line.rstrip() for line in open(args.stopwords) ])
#print(stopwords)
url = 'http://boston.lti.cs.cmu.edu/Services/treccast19/lemur.cgi?d=0&s=0&n=100&q='
doc_url = 'http://boston.lti.cs.cmu.edu/Services/treccast19/lemur.cgi?x=false&e='
cnt = 0

with open(args.input_file, "r+", encoding="utf8") as input:
    for line in input:
        line = line.strip()
        input_file = '../../output_file/' + line
        output_file = '../runs/t5-trec_batch_new_' + line.split('.')[0]
        fw_bert = open(output_file + '.txt', 'w', errors = 'ignore')
        fw_trec = open(output_file + '.trec', 'w', errors = 'ignore')
        print('ready to query ' + line)
        with open(input_file, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                if str(item['query_number']) == '2':
                    query_term = item['input'][0]
                    query_id = removePunctuation(query_term)
                    backup = query_id
                    query_words = query_id.split(' ')
                    query_id = ''
                    if args.stopwords != 'NONE':
                        for word in query_words:
                            if word.lower() not in stopwords:
                                query_id = query_id + word + ' '

                    query_id = query_id.strip()
                    if len(query_id) == 0:
                        if len(backup.strip()) != 0:
                            query_id = backup
                        else:
                            query_id = 'empty query'
                    print(query_term)
                    print(query_id)
                    hits = searcher.search(query_id, int(args.hit_size))
                    for i in range(int(args.hit_size)):
                        temphit = hits[i]
                        if i == len(hits):
                            break
                        if len(temphit.docid) <= 10:
                            doc_id = 'MARCO_'
                        else:
                            doc_id = 'CAR_'
                        doc_id = doc_id + temphit.docid
                        fw_bert.write(str(item['topic_number']) + '_' + '1' + '\t' + doc_id + '\t' + query_term.replace('\n','') + '\t' + temphit.content.replace('\t', ' ').replace('\n',' ') + '\t' + str(temphit.score) + '\n')
                        fw_trec.write(str(item['topic_number']) + '_' + '1' + '\tQ0\t' + doc_id + f'\t{i+1}\t' + str(temphit.score) + '\tBM25\n')

                if args.search_index == 'input':
                    query_term = item['input'][-1]
                else:
                    query_term = item[args.search_index]

                query_id = removePunctuation(query_term)
                backup = query_id
                query_words = query_id.split(' ')
                query_id = ''
                if args.stopwords != 'NONE' :
                    for word in query_words:
                        if word.lower() not in stopwords:
                            query_id = query_id + word + ' '

                query_id = query_id.strip()
                if len(query_id) == 0:
                    if len(backup.strip()) != 0:
                        query_id = backup
                    else:
                        query_id = 'empty query'
                print(query_term)
                print(query_id)
                hits = searcher.search(query_id, int(args.hit_size))
                for i in range(int(args.hit_size)):
                    if i == len(hits):
                        break
                    temphit = hits[i]
                    if len(temphit.docid) <= 10:
                        doc_id = 'MARCO_'
                    else:
                        doc_id = 'CAR_'
                    doc_id = doc_id + temphit.docid
                    fw_bert.write(str(item['topic_number']) + '_' + str(item['query_number']) + '\t' + doc_id + '\t' + query_term.replace('\n','') + '\t' + temphit.content.replace('\t', ' ').replace('\n',' ')  + '\t' + str(temphit.score) + '\n')
                    fw_trec.write(str(item['topic_number']) + '_' + str(item['query_number']) + '\tQ0\t' + doc_id + f'\t{i+1}\t' + str(temphit.score) + '\tBM25\n')


        fw_bert.close()
        fw_trec.close()
        print(line + ' Done!')
#resp = http.request('GET', 'http://boston.lti.cs.cmu.edu/Services/treccast19/lemur.cgi?x=false&q=my+name+is+joker', timeout=30)
#
#print(resp.data)
#f = open('1.out', 'wb+')
#f.write(resp.data)
#f.close()
