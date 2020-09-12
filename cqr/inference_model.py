
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using nucleus (top-p) filtering
        Args:
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

    def get_input_seq(self, input_sents):

        input = ''
        lable = ''

        max_len = len(input_sents)
        for t, sent in enumerate(input_sents):

            input += sent
            if t < max_len - 1:
                input += self.tokenizer.sep_token

        input_ecodings = self.tokenizer.batch_encode_plus([input], return_tensors='pt', padding=True)
        input_ecodings.to(self.model.device)

        input_ids = input_ecodings['input_ids']

        return input_ids

    def remove_special_tokens(self, text):
        # Remove special tokens from the output text in rare cases
        for token in self.special_tokens:
            text = text.replace(token, "")
        return text

    def predict(self, input_sents):
        input_ids = self.get_input_seq(input_sents)
        input_length = len(input_ids)


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

