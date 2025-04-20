import torch
import torch.nn as nn
from transformers.models.gpt2 import GPT2Model, GPT2Config


class Embed4D(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.word = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.ModuleList([
            nn.Embedding(cfg.max_position_embeddings, cfg.n_embd)
            for _ in range(4)
        ])
    def forward(self, ids, coords):
        w = self.word(ids)
        p = sum(tab(coords[..., i]) for i, tab in enumerate(self.pos))
        return w + p

class GPT2Small4D(nn.Module):
    def __init__(self, switch_epoch=1, quant_mode="int8"):
        super().__init__()
        self.cfg = GPT2Config.from_pretrained("gpt2", n_layer=6, n_head=8, 
                                              n_embd=512, vocab_size=50257, n_positions=1024)
        self.config = self.cfg 
        base = GPT2Model(self.cfg)
        base.wte = Embed4D(self.cfg)
        self.gpt2 = base
        self.lm_head = nn.Linear(self.cfg.n_embd, self.cfg.vocab_size, bias=False)
        self.switch_epoch = switch_epoch
        self.quant_mode = quant_mode
        self._switched = False

    def _make_quant(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if self.quant_mode=="int8":
                    new = Int8Linear(child.in_features, child.out_features, bias=child.bias is not None)
                elif self.quant_mode=="binary":
                    new = BinaryLinear(child.in_features, child.out_features, bias=child.bias is not None)
                else:
                    new = TernaryLinear(child.in_features, child.out_features, bias=child.bias is not None)
                # copy weights/bias
                new.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new.bias.data.copy_(child.bias.data)
                setattr(module, name, new)
            else:
                self._make_quant(child)

    def maybe_switch(self, epoch, optimizer=None):
        if not self._switched and epoch >= self.switch_epoch:
            self._make_quant(self)
            if optimizer is not None:
                for g in optimizer.param_groups:
                    g["weight_decay"] = 0.0        # BEFORE the next step
            self._switched = True

    # *** NEW manual forward path that feeds coords into embeddings ***
    def forward(self, ids, coords_4d, labels=None, attention_mask=None):
        # 1. embeddings
        hidden = self.gpt2.drop(self.gpt2.wte(ids, coords_4d))
        if attention_mask is not None:
            # original pad mask: [B, T]  →  SDPA mask: [B, H, T, T]
            pad = attention_mask[:, None, None, :]                     # [B,1,1,T]
           # attention_mask = pad.expand(-1, self.config.n_head,        # H = n_head
           #                             pad.size(-1), pad.size(-1))    # [B,H,T,T]
            attention_mask = attention_mask.to(hidden.dtype)

        # 2. blocks
        for blk in self.gpt2.h:
            hidden = blk(hidden, attention_mask=attention_mask)[0]
        hidden = self.gpt2.ln_f(hidden)
        # 3. LM head
        logits = self.lm_head(hidden)
        if labels is None:
            return logits
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        return loss, logits


