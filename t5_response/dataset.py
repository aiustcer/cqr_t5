import json
from torch.utils.data import Dataset

import transformers as tf
from t5_response.source import QueryRewriteDataset, NUM_FOLD, set_seed, special_tokens_dict

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


        self.examples = []
        for filename in filenames:
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)


                    topic_number = record['topic_num']
                    query_number = record['number']

                    input = ''
                    if record['histroy_str'] != '':
                        input += record['histroy_str'] + tokenizer.sep_token

                    if record['response'] != '':
                        input += record['response'] + tokenizer.sep_token

                    input += record['raw_utterance']+tokenizer.eos_token
                    lable = record['manual_rewritten_utterance'] + tokenizer.eos_token

                    self.examples.append(ConvSearchExample(topic_number, query_number, input, lable))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    args = None
    tokenizer = tf.T5Tokenizer.from_pretrained('t5-base')
    filenames = ['/data3/private/fengtao/Projects/cqr_t5/t5_response/data/processed_data_his.json']

    tokenizer.add_special_tokens(special_tokens_dict)

    a = QueryRewriteDataset(filenames, tokenizer, args)
    print('ok')

