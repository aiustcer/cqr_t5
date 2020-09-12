import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import torch.nn as nn

class cqr_t5(nn.Module):
    def __init__(self, config, word_embed, entity_embed):
        super(cqr_t5, self).__init__()
        self.is_inference = False

        # t5
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base')



    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path)

        params = {
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

    def load(self,file_path):
        if file_path == 't5-base':
            return

        else:
            params = torch.load(file_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(params['state_dict'])