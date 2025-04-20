import torch
import torch.nn as nn
from torch.autograd.function import Function

class _ClampSTE(Function):
    @staticmethod
    def forward(ctx, w, low, high):
        return w.clamp(low, high)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def ste_clamp(w, low=-1.0, high=1.0):
    return _ClampSTE.apply(w, low, high)

class BinaryLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        bw = ste_clamp(w, -1, 1).sign() if self.training else w.sign()
        return nn.functional.linear(x, bw, self.bias)

class TernaryLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        if self.training:
            tw = ste_clamp(w, -1, 1)
            tw = torch.where(tw.abs() < 0.05, torch.zeros_like(tw), tw.sign())
        else:
            tw = torch.where(w.abs() < 0.05, torch.zeros_like(w), w.sign())
        return nn.functional.linear(x, tw, self.bias)

class Int8Linear(nn.Linear):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int))

    def quantize_weight(self):
        w = self.weight.detach()
        w_min, w_max = w.min(), w.max()
        # symmetric quant
        max_abs = max(-w_min, w_max)
        self.scale      = torch.tensor(float(max_abs / 127))
        self.zero_point = torch.tensor(0, dtype=torch.int)
        q = torch.quantize_per_tensor(w, float(self.scale), int(self.zero_point), torch.qint8)
        self.weight_int8 = q.int_repr()
        del self.weight  # drop float

    def forward(self, x):
        if self.training:
            w = ste_clamp(self.weight, -128, 127) / (127 * self.scale)
            return nn.functional.linear(x, w, self.bias)
        else:
            w = (self.weight_int8.float() - self.zero_point.item()) * self.scale.item()
            return nn.functional.linear(x, w, self.bias)


