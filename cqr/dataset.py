
import json

from torch.utils.data import Dataset

class ConvSearchExample:
    def __init__(self, topic_number, query_number, input, label):
        self.topic_number = topic_number
        self.query_number = query_number
        self.input = input
        self.label = label

    
    def __repr__(self):
        print('===ConvSearchExample===')
        print(self.topic_number + '_' + self.query_number)
        print('-----------------------')
        print(self.input)
        print('-----------------------')
        print(self.label)
        print('=======================')


class QueryRewriteDataset(Dataset):
    def __init__(self, filenames, tokenizer, args):

        # debug
        # str = '<BOS> <SEP> Hello, my dog is cute'
        # a = tokenizer.tokenize(str)
        # b = tokenizer.convert_tokens_to_ids(a)
        #
        # str1 = 'good hi <BOS> <SEP> Hello, my<BOS> dog is cute <BOS><SEP> hi </s>'
        # a2 = tokenizer.tokenize(str1)
        # b2 = tokenizer.convert_tokens_to_ids(a2)
        #
        # encodings = tokenizer.batch_encode_plus([str] + [str1],
        #                                              return_tensors='pt',
        #                                              padding=True
        #                                              )


        self.examples = []
        for filename in filenames:
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    input_sents = record['input']
                    target_sent = record['target']
                    topic_number = record['topic_number']
                    query_number = record['query_number']

                    input = ''
                    lable = ''

                    max_len = len(input_sents)
                    for t,sent in enumerate(input_sents):

                        input += sent
                        if t < max_len - 1:
                         input += tokenizer.sep_token


                    lable +=  target_sent + tokenizer.eos_token


                    self.examples.append(ConvSearchExample(topic_number, query_number, input, lable))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

