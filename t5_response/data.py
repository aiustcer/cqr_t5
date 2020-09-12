# Version 1.1
import os
import sys
import codecs
from trec_car.read_data import *
import argparse
import tqdm
from urllib import parse
import csv
import re
import logging
import json
import copy



from torch.utils.data import Dataset
special_tokens_dict = {'sep_token': '<SEP>'}

def writer(p, fp, meta_dict={}):
    """
    Writes each paragraph in the trecweb format
    """
    # Get the paragraph id and text
    para_id = 'CAR_' + str(p.para_id)
    text = p.get_text()

    content = (u'<DOC>\n')
    content += (u'<DOCNO>')
    content += (para_id)
    content += (u'</DOCNO>\n')
    content += (u'<DOCHDR>\n')
    content += (u'\n')
    content += (u'</DOCHDR>\n')
    if meta_dict and p.para_id in meta_dict:
        title = meta_dict[p.para_id]["title"]
        headings = meta_dict[p.para_id]["headings"]
        title_string = (u'<TITLE>\n')
        if title and not title.isspace():
            title_string += (title)
        if headings and not headings.isspace():
            title_string += (u' -- ')
            title_string += (headings)
        title_string += (u'\n</TITLE>\n')
        content += title_string
    elif meta_dict and p.para_id not in meta_dict:
        logging.warning(f'No metadata for ID: {p.para_id}')
    content += (u'<BODY>\n')
    content += (text)
    content += (u'\n</BODY>\n')
    content += (u'</DOC>\n')
    fp.write(content)


def load2dict(p, fp, meta_dict={}):
    """
    Writes each paragraph in the trecweb format
    """
    # Get the paragraph id and text
    para_id = 'CAR_' + str(p.para_id)
    text = p.get_text()


def sanitize_string(s):
    s = parse.unquote(s)
    s = re.sub(r'\W+', ' ', s)
    s = s.replace("enwiki", "")
    return s


def create_metadata_dict(tsvfile):
    if tsvfile == None:
        return {}
    num_lines = sum(1 for line in open(tsvfile))
    assert num_lines == 36005581, f"Number of metadata records ({num_lines}) number is unexpected, expected 36005581. Ensure you are using the correct file."
    with open(args.metadata_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        metadata_dict = {}
        for row in tqdm.tqdm(reader, desc='Loading metadata lines', total=num_lines):
            assert len(row) == 3, f"Row is not formatted correctly: {row}"
            metadata_dict[row[0]] = {"title": sanitize_string(row[1]), "headings": sanitize_string(row[2])}
        return metadata_dict




def make_car_dict_file(file="/data3/private/fengtao/Projects/cqr_t5/data_download/paragraphCorpus/dedup.articles-paragraphs.cbor"):
# make car dict
    car_dict_list = []
    k = 0
    with open(file, 'rb') as rp:
        for p in tqdm.tqdm(iter_paragraphs(rp), desc="Converting to trecweb"):
            # Write to file
            para_id = 'CAR_' + str(p.para_id)
            text = p.get_text()
            dict = {para_id: text}
            car_dict_list.append(dict)




    with open('/data3/private/fengtao/Projects/cqr_t5/t5_response/data/car_dict_2.json', 'w') as fout:
        for data in car_dict_list:
            dict_str = json.dumps(data) + '\n'
            fout.write(dict_str)


def get_dict():
    car_dict_file = '/data3/private/fengtao/Projects/cqr_t5/t5_response/data/car_dict_2.json'

    msmarco_dict_file = '/data3/private/fengtao/Projects/cqr_t5/t5_response/data/collection.tsv'
    dict = {}

    with open(msmarco_dict_file, 'r') as f:

        head_str = 'MARCO_'
        for line in f:
            id = head_str + line.split('\t')[0]
            text = line.split('\t')[1]
            dict[id] = text


    with open(car_dict_file, 'r') as f:
        for line in f:

            dict.update(json.loads(line))


    return dict



def add_response():
    file = '/data3/private/fengtao/Projects/cqr_t5/t5_response/data/2020/2020_manual_evaluation_topics_v1.0.json'
    file_out = '/data3/private/fengtao/Projects/cqr_t5/t5_response/data/processed_data_v4.json'


    ms_car_dick = get_dict()

    with open(file, 'r') as f:
        cast_dick = json.load(f)


    data_processed = []
    for group in cast_dick:
        topic_number, turn = str(group['number']), group['turn']

        queries = []
        response = []
        for query in turn:
            query_number, raw_utterance = str(query['number']), query['raw_utterance']
            queries.append(raw_utterance)
            text = ms_car_dick[query['manual_canonical_result_id']]
            response.append(text)

            record = {'topic_number': topic_number,
                      'query_number': query_number,
                      'input': copy.deepcopy(queries),
                      'target': query['manual_rewritten_utterance'],
                      'response': copy.deepcopy(response)
                      }

            data_processed.append(record)



    with open(file_out, 'w') as f:
        for data in data_processed:
            f.write(json.dumps(data) + '\n')














# # make msmarco dict
#
#     with open('/data3/private/fengtao/Projects/cqr_t5/data_download/collection.tsv', 'r') as fin:
#         annonated_lines = fin.readlines()
#
#     all_annonated = {}
#     for line in annonated_lines:
#         print(line)
#






if __name__ == '__main__':
    add_response()
    # parser = argparse.ArgumentParser(
    #     description='"Usage: python car_treweb.py dedup.articles-paragraphs.cbor DUMP_DIR --metadata_file car_meta.tsv')
    # parser.add_argument('filename', help='cbor file to process')
    # parser.add_argument('dump_dir', help='duplicates file')
    # parser.add_argument('--metadata_file', default=None, help='TSV file containing CAR para IDs and metadata')
    # args = parser.parse_args()
    #
    # filename = args.filename
    # dump_dir = args.dump_dir
    #
    # input_file = os.path.basename(filename)
    #
    # dumper_file = os.path.join(dump_dir, input_file + '.xml')
    # print("Writing output to: " + dumper_file)
    # fp = codecs.open(dumper_file, 'w', 'utf-8')
    # print("Starting processing.")
    # print("Output directory: " + dump_dir)
    #
    # meta_dict = create_metadata_dict(args.metadata_file)
    #
    # # Reads the file and iterates over paragraphs
    # total = 0
    # print("Reading ", filename)
    # with open(filename, 'rb') as rp:
    #     for p in tqdm.tqdm(iter_paragraphs(rp), desc="Converting to trecweb"):
    #         # Write to file
    #         writer(p, fp, meta_dict=meta_dict)
    #         total += 1
    # print("Total paras written = ", total)
    # print("Closing File")
    #
    # rp.close()
    # fp.close()
