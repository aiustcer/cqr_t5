import json
from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import random
import torch
import numpy as np

NUM_FOLD = 5
QUESTION_WORD_LIST = ["what", "when", "why", "who", "how", "where", "whose", "is", "are", "were", "was", "do", "does",
                      "did", "can"]
OTHER_WORD_LIST = ["tell"]
special_tokens_dict = {'sep_token': '<SEP>'}

def from_record_get_input(record, tokenizer):
    def v1(record, tokenizer):
        input_sents = record['input']
        response_list = record['response']

        input = ''

        if len(response_list) == 1:
            input = ''
        else:
            response = response_list[-2]
            input += response + tokenizer.sep_token

        max_len = len(input_sents)
        for t, sent in enumerate(input_sents):
            input += sent
            if t < max_len - 1:
                input += tokenizer.sep_token

        input += tokenizer.eos_token

        return input

    def v2(record, tokenizer):
        input_sents = record['input']
        response_list = record['response']

        input = ''

        max_len = len(input_sents)
        for t, sent in enumerate(input_sents):
            input += sent
            if t < max_len - 1:
                input += tokenizer.sep_token

        if len(response_list) != 1:

            response = response_list[-2]
            input += tokenizer.sep_token + response



        input += tokenizer.eos_token

        return input


    def v3(record, tokenizer):
        input_sents = record['input']

        input = ''

        max_len = len(input_sents)
        for t, sent in enumerate(input_sents):
            input += sent
            if t < max_len - 1:
                input += tokenizer.sep_token
        # input += tokenizer.eos_token

        return input

    return v3(record, tokenizer)



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using nucleus (top-p) filtering
        Args:
            top_p:
            logits: logits distribution shape (batch size x vocabulary size)
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


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
    def response_add_v1(self, filenames, tokenizer, args):
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
                    for t, sent in enumerate(input_sents):

                        input += sent
                        if t < max_len - 1:
                            input += tokenizer.sep_token

                    # response
                    input += tokenizer.sep_token + record['response'] + tokenizer.eos_token

                    lable += target_sent + tokenizer.eos_token

                    self.examples.append(ConvSearchExample(topic_number, query_number, input, lable))

    def response_add_v2(self, filenames, tokenizer, args):
        self.examples = []
        for filename in filenames:
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    input_sents = record['input']
                    target_sent = record['target']
                    topic_number = record['topic_number']
                    query_number = record['query_number']

                    input = record['response'] + tokenizer.sep_token
                    lable = ''

                    max_len = len(input_sents)
                    for t, sent in enumerate(input_sents):

                        input += sent
                        if t < max_len - 1:
                            input += tokenizer.sep_token

                    input += tokenizer.eos_token

                    lable += target_sent + tokenizer.eos_token

                    self.examples.append(ConvSearchExample(topic_number, query_number, input, lable))

    def response_add_v3(self, filenames, tokenizer, args):
        self.examples = []
        for filename in filenames:
            with open(filename, encoding="utf-8") as f:
                response = ''
                for line in f:
                    record = json.loads(line)
                    input_sents = record['input']
                    target_sent = record['target']
                    topic_number = record['topic_number']
                    query_number = record['query_number']

                    input = ''
                    lable = ''


                    input = from_record_get_input(record, tokenizer)

                    lable += target_sent + tokenizer.eos_token

                    self.examples.append(ConvSearchExample(topic_number, query_number, input, lable))

    def response_add_v4(self, filenames, tokenizer, args):
        self.examples = []
        for filename in filenames:
            with open(filename, encoding="utf-8") as f:
                response = ''
                topic_number = -1
                for line in f:
                    record = json.loads(line)
                    input_sents = record['input']
                    target_sent = record['target']


                    query_number = record['query_number']
                    if response != '':
                        input = record['response'] + tokenizer.sep_token
                    else:
                        input = ''

                    lable = ''

                    max_len = len(input_sents)
                    for t, sent in enumerate(input_sents):

                        input += sent
                        if t < max_len - 1:
                            input += tokenizer.sep_token

                    input += tokenizer.eos_token

                    lable += target_sent + tokenizer.eos_token

                    self.examples.append(ConvSearchExample(topic_number, query_number, input, lable))

    def __init__(self, filenames, tokenizer, args):

        self.response_add_v4(filenames, tokenizer, args)

        # self.examples = []
        # for filename in filenames:
        #     with open(filename, encoding="utf-8") as f:
        #         for line in f:
        #             record = json.loads(line)
        #             input_sents = record['input']
        #             target_sent = record['target']
        #             topic_number = record['topic_number']
        #             query_number = record['query_number']
        #
        #             input = ''
        #             lable = ''
        #
        #             max_len = len(input_sents)
        #             for t, sent in enumerate(input_sents):
        #
        #                 input += sent
        #                 if t < max_len - 1:
        #                     input += tokenizer.sep_token
        #
        #             # response
        #             input += tokenizer.sep_token + record['response'] + tokenizer.eos_token
        #
        #             lable += target_sent + tokenizer.eos_token
        #
        #             self.examples.append(ConvSearchExample(topic_number, query_number, input, lable))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


class InferenceModel_bandoned:

    def __init__(self, args):
        model_class, tokenizer_class = T5ForConditionalGeneration, T5Tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(args.model_path)
        self.model = model_class.from_pretrained(args.model_path)
        self.model.to(args.device)
        self.model.eval()

        self.device = args.device
        self.length = args.length
        # if self.model.config.max_position_embeddings < args.length:
        #     self.length = model.config.max_position_embeddings # No generation bigger than model size
        self.temperature = args.temperature
        self.top_p = args.top_p

        self.special_tokens = ['<SEP>', '<PAD>', '<BOS>', '<EOS>']

    def get_input_seq(self, record):

        input = ''
        if record['histroy_str'] != '':
            input += record['histroy_str'] + self.tokenizer.sep_token

        if record['response'] != '':
            input += record['response'] + self.tokenizer.sep_token

        input += record['raw_utterance'] + self.tokenizer.eos_token

        input_ecodings = self.tokenizer.batch_encode_plus([input], return_tensors='pt', padding=True)
        input_ecodings.to(self.model.device)

        input_ids = input_ecodings['input_ids']

        return input_ids

    def remove_special_tokens(self, text):
        # Remove special tokens from the output text in rare cases
        for token in self.special_tokens:
            text = text.replace(token, "")
        return text

    def predict(self, record):
        input_ids = self.get_input_seq(record)

        outputs = self.model.generate(input_ids,
                                      temperature=self.temperature if self.temperature > 0 else 1.,
                                      top_p=self.top_p,
                                      do_sample=True)
        # outputs1 = self.model.generate(input_ids)
        # outputs2 = self.model.generate(input_ids,
        #                               temperature=self.temperature if self.temperature > 0 else 1.,
        #                               top_p=self.top_p,
        #                               do_sample=True,
        #                                num_beams=5)
        #
        # pred_text_debug1 = self.tokenizer.decode(outputs1[0])
        # pred_text_debug2 = self.tokenizer.decode(outputs2[0])

        pred_text = self.tokenizer.decode(outputs[0])

        return pred_text

class InferenceModel:

    def __init__(self, args):
        model_class, tokenizer_class = T5ForConditionalGeneration, T5Tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(args.model_path)
        self.model = model_class.from_pretrained(args.model_path)
        self.model.to(args.device)
        self.model.eval()

        self.device = args.device
        self.length = args.length
        # if self.model.config.max_position_embeddings < args.length:
        #     self.length = model.config.max_position_embeddings # No generation bigger than model size
        self.temperature = args.temperature
        self.top_p = args.top_p

        self.special_tokens = ['<SEP>', '<PAD>', '<BOS>', '<EOS>']

    def get_input_seq(self, record):

        input = from_record_get_input(record, self.tokenizer)
        input_ecodings = self.tokenizer.batch_encode_plus([input], return_tensors='pt', padding=True)
        input_ecodings.to(self.model.device)

        input_ids = input_ecodings['input_ids']

        return input_ids

    def remove_special_tokens(self, text):
        # Remove special tokens from the output text in rare cases
        for token in self.special_tokens:
            text = text.replace(token, "")
        return text

    def predict(self, record):
        input_ids = self.get_input_seq(record)


        outputs = self.model.generate(input_ids,
                                      temperature=self.temperature if self.temperature > 0 else 1.,
                                      top_p=self.top_p,
                                      do_sample=True)
        # outputs1 = self.model.generate(input_ids)
        # outputs2 = self.model.generate(input_ids,
        #                               temperature=self.temperature if self.temperature > 0 else 1.,
        #                               top_p=self.top_p,
        #                               do_sample=True,
        #                                num_beams=5)
        #
        # pred_text_debug1 = self.tokenizer.decode(outputs1[0])
        # pred_text_debug2 = self.tokenizer.decode(outputs2[0])

        pred_text = self.tokenizer.decode(outputs[0])

        return pred_text
