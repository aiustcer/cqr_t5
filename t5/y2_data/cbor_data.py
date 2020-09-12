import cbor2

cbor_file_dir = '/data3/private/fengtao/Projects/cqr_t5/data_download/paragraphCorpus/dedup.articles-paragraphs.cbor'
collection_file = '/data3/private/fengtao/Projects/cqr_t5/data_download/collection.tsv'

with open('/data3/private/fengtao/Projects/cqr_t5/data_download/paragraphCorpus/dedup.articles-paragraphs.cbor',
          'rb') as fp:
    data = cbor2.decoder.load(fp)
    data2 = cbor2.load(fp)
    while True:
        a = fp.readline()

