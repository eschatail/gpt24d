from transformers import GPT2TokenizerFast
from hilbert4d import decode as h_decode

class GPT2Tokenizer4D(GPT2TokenizerFast):
    def __call__(self, text, **kwargs):
        enc = super().__call__(text, **kwargs)
        base = self.pad_token_id + 1
        coords = [
            h_decode(i + base, bits=8)
            for i in range(len(enc["input_ids"]))
        ]
        enc["coords_4d"] = coords
        return enc

