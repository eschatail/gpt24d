import torch, argparse, pathlib, math
from torch.utils.data import DataLoader, IterableDataset
from tokenizer_4d import GPT2Tokenizer4D
from model_4d     import GPT2Small4D
from torch_optimizer import Lamb
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--dataset", type=pathlib.Path, required=True)
p.add_argument("--epochs",  type=int,  default=3)
p.add_argument("--batch",   type=int,  default=1)          # micro‑batch on GPU
p.add_argument("--grad_accum", type=int, default=8)        # virtual batch = 8×
p.add_argument("--seq_len", type=int,  default=512)        # trim context
p.add_argument("--quant_mode", choices=["int8","binary","ternary"], default="int8")
args = p.parse_args()

# ---------- device ----------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using", device)

# ---------- FP16 as default BEFORE model build ----------
torch.set_default_dtype(torch.float16)

# ---------- tokenizer ----------
tok = GPT2Tokenizer4D.from_pretrained("gpt2", padding_side="right")
tok.pad_token = tok.eos_token

# ---------- model ----------
model = GPT2Small4D(switch_epoch=1, quant_mode=args.quant_mode)
model.gpt2.gradient_checkpointing_enable()     # saves ≈ 2× activations
model = model.half().to(device)           # keep single FP16 copy
model.config.pad_token_id = tok.pad_token_id

# ---------- optimiser + scheduler ----------
optim = Lamb(model.parameters(), lr=1e-3, weight_decay=0.01)
steps_per_epoch_est = 10_000              # rough – fine for cosine
scheduler = CosineAnnealingLR(optim, T_max=steps_per_epoch_est * args.epochs)

# ---------- dataset ----------
class TextLineDataset(IterableDataset):
    def __init__(self, path): self.path = pathlib.Path(path)
    def _files(self):
        yield from (self.path.rglob("*.txt") if self.path.is_dir() else [self.path])
    def __iter__(self):
        for f in self._files():
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line := line.strip(): yield line

ds = TextLineDataset(args.dataset)

def collate(batch_lines):
    enc = tok(batch_lines, padding=True, truncation=True,
          max_length=args.seq_len, return_tensors="pt")
    ids   = enc["input_ids"].to(device)
    attn  = enc["attention_mask"].to(device)  
    coords = torch.zeros(ids.size(0), ids.size(1), 4,
                         dtype=torch.long, device=device)
    for i, tup in enumerate(enc["coords_4d"]):
        coords[i, :len(tup)] = torch.tensor(tup, device=device)
    # 4‑D mask for SDPA
    attn = attn[:, None, None, :].to(dtype=torch.float16)
    return ids, coords, attn

loader = DataLoader(ds, batch_size=args.batch, collate_fn=collate, num_workers=0)

# ---------- train ----------
virtual_bs = args.batch * args.grad_accum
print(f"virtual batch = {virtual_bs}, seq_len = {args.seq_len}")

step = 0
for epoch in range(args.epochs):
    model.train()
    for ids, coords, attn in loader:
        with torch.autocast("mps", dtype=torch.float16):
            loss, _ = model(ids, coords, labels=ids, attention_mask=attn)
            loss = loss / args.grad_accum
        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            model.maybe_switch(epoch, optimizer=optim)   # quant switch once
            optim.step(); scheduler.step(); optim.zero_grad()
            gc.collect()
            torch.mps.empty_cache()
        step += 1

    print(f"epoch {epoch}  loss {loss.item()*args.grad_accum:.4f}")

    # save every epoch
    model.save_pretrained(f"ckpt-{epoch}")
    tok.save_pretrained(f"ckpt-{epoch}")
    torch.save(optim.state_dict(),     f"ckpt-{epoch}/optim.pt")
    torch.save(scheduler.state_dict(), f"ckpt-{epoch}/sched.pt")
