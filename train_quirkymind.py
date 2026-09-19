#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_quirkymind.py — train the dual-stream persona steering module.

Produces the checkpoint the demo loads in init_prefix(), so that the
QUIRKYMIND arm in the interface is actually running the method rather than a
placeholder prefix.

------------------------------------------------------------------------
What this trains
------------------------------------------------------------------------
Two small modules on top of a frozen LM and a frozen SBERT encoder:

  g_phi : R^{d_sbert} -> R^{S x H}   MLP adapter. Turns the SBERT summary of
                                     a persona into S soft tokens. These
                                     soft tokens ARE the persona prefix the
                                     demo splices into the sequence.
  f_theta : R^{H} -> R^{d_sbert}     Projection head. Maps an LM hidden state
                                     back into SBERT space so a contrastive
                                     loss can be applied there.

The base LM stays frozen throughout. With --lora, low-rank adapters are added
to the LM and trained alongside; without it only g_phi and f_theta move.

------------------------------------------------------------------------
WHAT g_phi IS FED  (--gphi-input)  — read this before training
------------------------------------------------------------------------
The prompt is deliberately stance-free (see build_prompt). That means the
ONLY route by which the persona's incongruous stance can reach the model is
the soft prefix, i.e. whatever g_phi is given as input.

  backstory   g_phi(SBERT(backstory)).  The stance is NOT in this text, so
              two personas with the same demographic line and opposite
              stances get the same prefix. g_phi cannot represent the
              incongruity; the best it can learn is the training set's
              average stance per demographic — the stereotype. This is the
              configuration the first checkpoints were trained with, and it
              explains what they showed: a weak gain on an axis where the
              selected personas all share one stance, none on an axis where
              they split 50/50, and nothing that generalises.
  declared    g_phi(SBERT("<backstory>. This person <declared>."))
              The stance enters the latent and only the latent; the prompt
              still withholds it. This is the setting under which
              "prefix vs no prefix, stance withheld" is a real test of
              steering. RECOMMENDED.
  history     g_phi(SBERT(backstory + the persona's prior in-character
              responses)). The paper's Traits Anchoring ("encodes dialogue
              history and transient stances", Sec. 3.1). Turn 0 has no
              history, so it falls back to `declared`.

The choice is saved in the checkpoint config; the demo reads it and builds
the identical string, so training and inference cannot silently diverge.
`--check-data` reports how many personas share a backstory string with a
persona of a different stance — every such pair is a prefix collision under
`backstory`.

Alternating objective, per the paper:

  discriminative step   InfoNCE in SBERT space. Pull f_theta(h) toward the
                        embedding of the persona's own response, push away
                        from the stereotypical response and from other
                        personas' responses in the batch. Updates g_phi and
                        f_theta.
  generative step       Cross-entropy on the response tokens, with f_theta
                        frozen. Updates g_phi only.

------------------------------------------------------------------------
Data
------------------------------------------------------------------------
JSONL, one persona per line:

{
  "backstory": "A weekly churchgoer who has led the same parish study group ...",
  "declared":  "believes same-sex couples deserve the same marriage rights",
  "expected":  "oppose same-sex marriage on doctrinal grounds",
  "turns": [
    {"user": "Why do you hold that position?",
     "response": "I have sat with this for years. My faith is what brought me ...",
     "negative": "Scripture is clear on this, and I am not free to rewrite it."},
    {"user": "Your congregation would call that a betrayal.",
     "response": "Some of them do. I still take communion beside them."}
  ]
}

Validate a file before a long run:
    python train_quirkymind.py --data personas.jsonl --check-data

------------------------------------------------------------------------
Running
------------------------------------------------------------------------
    python train_quirkymind.py \
        --data atp_training.jsonl --bank atp_bank.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --gphi-input declared --soft-tokens 16 --epochs 8 --lr 1e-4 \
        --out checkpoints/quirkymind_v3

Then in the demo:  QUIRKYMIND_CKPT=checkpoints/quirkymind_v3/quirkymind.pt
(the demo picks up gphi_input from the checkpoint automatically).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_LOGITS_KW = ""          # set in main(): "logits_to_keep" / "num_logits_to_keep" / ""


# =====================================================================
# Data
# =====================================================================
@dataclass
class Turn:
    user: str
    response: str
    negative: Optional[str] = None


@dataclass
class PersonaRecord:
    backstory: str
    declared: str
    expected: str
    turns: List[Turn]

    def expected_sentence(self) -> str:
        return f"Like most people in this position, they {self.expected.rstrip('.')}."


def gphi_text(mode: str, backstory: str, declared: str,
              history_responses: Optional[List[str]] = None) -> str:
    """The text g_phi summarises. Byte-identical construction to the demo's
    _gphi_text(); change both or neither."""
    if mode == "declared":
        return f"{backstory.rstrip('.')}. This person {declared.rstrip('.')}."
    if mode == "history":
        if history_responses:
            return f"{backstory.rstrip('.')}. " + " ".join(r.strip() for r in history_responses)
        return f"{backstory.rstrip('.')}. This person {declared.rstrip('.')}."
    return backstory


class PersonaData(Dataset):
    def __init__(self, path: str, max_history: int = 6):
        self.records: List[PersonaRecord] = []
        self.index: List[tuple] = []
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rec = PersonaRecord(
                    backstory=row["backstory"], declared=row["declared"],
                    expected=row["expected"],
                    turns=[Turn(**t) for t in row["turns"]])
                ri = len(self.records)
                self.records.append(rec)
                for ti in range(len(rec.turns)):
                    self.index.append((ri, ti))
        self.max_history = max_history

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ri, ti = self.index[i]
        rec = self.records[ri]
        hist = rec.turns[max(0, ti - self.max_history):ti]
        return {"record": rec, "turn": rec.turns[ti], "history": hist,
                "turn_index": ti}


def collate(batch):
    return batch


def check_data(path: str) -> None:
    ds = PersonaData(path)
    n_neg = sum(1 for ri, ti in ds.index if ds.records[ri].turns[ti].negative)
    turns = [len(r.turns) for r in ds.records]
    print(f"personas           {len(ds.records)}")
    print(f"training examples  {len(ds)}")
    print(f"turns per persona  min {min(turns)} / mean {sum(turns)/len(turns):.1f} "
          f"/ max {max(turns)}")
    print(f"hard negatives     {n_neg}/{len(ds)} ({n_neg/max(1,len(ds)):.0%})")

    # Prefix collisions under --gphi-input backstory: identical backstory
    # text, different declared stance -> identical soft prefix, contradictory
    # targets. g_phi cannot fit these; it averages them.
    by_back: Dict[str, set] = defaultdict(set)
    for r in ds.records:
        by_back[r.backstory.strip()] .add(r.declared.strip())
    collide = {b: d for b, d in by_back.items() if len(d) > 1}
    n_pers_collide = sum(1 for r in ds.records if r.backstory.strip() in collide)
    print(f"distinct backstory texts   {len(by_back)} (of {len(ds.records)} personas)")
    print(f"backstories with >1 stance {len(collide)} — covering {n_pers_collide} personas")
    leak = sum(1 for r in ds.records if r.declared.strip().lower()[:25] in r.backstory.lower())
    print(f"stance already in backstory text: {leak} personas")
    if n_pers_collide:
        print("\n  Under --gphi-input backstory these personas get the SAME prefix with\n"
              "  different targets; the prefix can only learn their average stance.\n"
              "  Use --gphi-input declared (default) so the stance reaches g_phi.")
    if n_neg / max(1, len(ds)) < 0.5:
        print("\n  Fewer than half the turns have a hard negative; the contrastive\n"
              "  objective will lean on in-batch negatives that differ by topic.")
    if len(ds.records) < 20:
        print("\n  Few personas: expect g_phi to memorise rather than generalise.")


# =====================================================================
# Modules
# =====================================================================
class SoftPrefix(nn.Module):
    def __init__(self, d_sbert: int, hidden: int, n_tokens: int, width: int = 2048):
        super().__init__()
        self.n_tokens, self.hidden = n_tokens, hidden
        self.net = nn.Sequential(
            nn.Linear(d_sbert, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, n_tokens * hidden))
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        out = self.net(summary)
        return out.view(summary.size(0), self.n_tokens, self.hidden)


class Projector(nn.Module):
    def __init__(self, hidden: int, d_sbert: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_sbert))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=-1)


# =====================================================================
# Prompt assembly
# =====================================================================
def build_prompt(tok, rec: PersonaRecord, history: List[Turn], user: str) -> str:
    """Chat-template prompt WITHOUT the declared stance. Same system wording
    as the demo's Persona.system_text() under the paper protocol."""
    msgs = [{"role": "system",
             "content": f"{rec.backstory.rstrip('.')}. You are answering as this "
                        f"person. Speak in first person, stay in character under "
                        f"pressure, and never break role."}]
    for t in history:
        msgs.append({"role": "user", "content": t.user})
        msgs.append({"role": "assistant", "content": t.response})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def embed_with_prefix(lm, tok, prompt: str, prefix: torch.Tensor,
                      answer: Optional[str], device, max_len: int,
                      max_answer_len: int = 96):
    """inputs_embeds = [bos] + soft tokens + prompt (+ answer). Prefix after
    the first token, as in the demo's _prefixed_embeds()."""
    p_ids = tok(prompt, return_tensors="pt", truncation=True,
                max_length=max_len, add_special_tokens=False).input_ids.to(device)
    a_ids = None
    if answer is not None:
        a_ids = tok(answer, return_tensors="pt", add_special_tokens=False,
                    truncation=True, max_length=max_answer_len).input_ids.to(device)
        ids = torch.cat([p_ids, a_ids], dim=1)
    else:
        ids = p_ids
    emb = lm.get_input_embeddings()(ids)
    head, tail = emb[:, :1, :], emb[:, 1:, :]
    full = torch.cat([head, prefix.to(emb.dtype), tail], dim=1)
    attn = torch.ones(full.shape[:2], dtype=torch.long, device=device)
    n_answer = a_ids.size(1) if a_ids is not None else 0
    return full, attn, n_answer, a_ids


# =====================================================================
# Losses
# =====================================================================
def info_nce(anchor: torch.Tensor, pos: torch.Tensor, negs: torch.Tensor,
             temp: float = 0.07) -> torch.Tensor:
    pos_sim = (anchor * pos).sum(-1, keepdim=True) / temp
    neg_sim = anchor @ negs.t() / temp
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    target = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, target)


# =====================================================================
# Training
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sbert-device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="where the frozen sentence encoder runs. 'auto' puts it on the "
                         "CPU whenever the backbone is quantised or memory is capped, to "
                         "leave a second CUDA context out of a shared GPU; it is frozen "
                         "and runs under no_grad, so the device costs a little speed and "
                         "nothing else.")
    ap.add_argument("--gphi-input", choices=["backstory", "declared", "history"],
                    default="declared",
                    help="what g_phi summarises; see module docstring. "
                         "Saved to the checkpoint; the demo reads it.")
    ap.add_argument("--soft-tokens", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--disc-ratio", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--max-len", type=int, default=768,
                    help="prompt truncation. Activation memory is roughly linear "
                         "in this; 768 fits an 8B model in ~19GB with 4-bit weights.")
    ap.add_argument("--max-answer-len", type=int, default=96,
                    help="answer tokens scored per generative step. Logits are "
                         "only materialised for these positions.")
    ap.add_argument("--load-4bit", action="store_true",
                    help="load the backbone in NF4 (bitsandbytes). ~6GB instead of "
                         "~16GB for an 8B model. Needed when the GPU is shared.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute activations in the backward pass: much less "
                         "memory, ~30%% slower. Use with --load-4bit on a shared GPU.")
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help="cap this process at a fraction of total GPU memory "
                         "(e.g. 0.55) so a neighbouring job is not starved.")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora", action="store_true",
                    help="also train Stage-3 LoRA adapters on the backbone (paper Sec. 3.3). "
                         "Saved to <out>/lora; load in the demo with QUIRKYMIND_LORA.")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--bank", default="", help="MCQA bank json for per-epoch eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/quirkymind")
    args = ap.parse_args()

    if args.check_data:
        check_data(args.data)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Shared GPU: reserve a slice and report what is actually free, so an OOM
    # halfway through an epoch becomes a refusal to start instead.
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"[gpu] {free_b/2**30:.1f} GiB free of {total_b/2**30:.1f} GiB "
              f"({(total_b-free_b)/2**30:.1f} GiB used by other processes)")
        if args.mem_fraction > 0:
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
            print(f"[gpu] capped at {args.mem_fraction:.0%} of total for this process")
        need = 6.5 if args.load_4bit else 17.0
        if free_b / 2**30 < need:
            print(f"[gpu] WARNING: ~{need:.0f} GiB needed for weights alone and only "
                  f"{free_b/2**30:.1f} GiB is free."
                  + ("" if args.load_4bit else "  Add --load-4bit."))

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # transformers renamed this argument; detect once rather than try/except
    # inside the training loop.
    global _LOGITS_KW
    import inspect
    from transformers.models.llama import modeling_llama as _ml
    _sig = inspect.signature(_ml.LlamaForCausalLM.forward).parameters
    _LOGITS_KW = ("logits_to_keep" if "logits_to_keep" in _sig
                  else ("num_logits_to_keep" if "num_logits_to_keep" in _sig else ""))
    if not _LOGITS_KW:
        print("[mem] this transformers version has no logits_to_keep; the full "
              "logits tensor will be materialised (more memory per step)")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kw = dict(dtype=dtype, device_map={"": device})
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        load_kw.pop("dtype")
    lm = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    lm.config.use_cache = False
    for p in lm.parameters():
        p.requires_grad_(False)
    if args.grad_checkpoint and not args.lora:
        # With --lora this is done inside prepare_model_for_kbit_training().
        lm.gradient_checkpointing_enable()
        lm.enable_input_require_grads()      # needed: our input is inputs_embeds

    # SBERT lives on the CPU: it is tiny, it is frozen, and the two encoder
    # calls per batch are not the bottleneck — but a second CUDA context is
    # ~300MB that the neighbouring job could use.
    from sentence_transformers import SentenceTransformer

    if args.lora:
        # QLoRA order matters: cast norms/lm_head to fp32 and re-enable input
        # grads on the QUANTISED model BEFORE wrapping it, then add the
        # adapters. Doing it the other way round silently yields zero
        # gradients through the frozen 4-bit layers.
        from peft import LoraConfig, get_peft_model
        if args.load_4bit:
            from peft import prepare_model_for_kbit_training
            lm = prepare_model_for_kbit_training(
                lm, use_gradient_checkpointing=args.grad_checkpoint)
        lm = get_peft_model(lm, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM"))
        lm.enable_input_require_grads()   # our input is inputs_embeds, not ids
        lm.config.use_cache = False
        lm.print_trainable_parameters()

    if args.sbert_device == "auto":
        sbert_device = "cpu" if (args.load_4bit or args.mem_fraction) else device
    else:
        sbert_device = args.sbert_device
    print(f"[mem] sbert on {sbert_device}, backbone {'NF4 4-bit' if args.load_4bit else str(dtype)}"
          f"{', gradient checkpointing' if args.grad_checkpoint else ''}")
    sbert = SentenceTransformer(args.sbert, device=sbert_device)
    for p in sbert.parameters():
        p.requires_grad_(False)
    d_sbert = sbert.get_sentence_embedding_dimension()
    hidden = lm.config.hidden_size

    g_phi = SoftPrefix(d_sbert, hidden, args.soft_tokens).to(device).to(torch.float32)
    f_theta = Projector(hidden, d_sbert).to(device).to(torch.float32)

    params = list(g_phi.parameters()) + list(f_theta.parameters())
    if args.lora:
        params += [p for p in lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    ds = PersonaData(args.data)
    if args.limit:
        ds.index = ds.index[:args.limit]
        print(f"[smoke test] using {len(ds)} of the full example set")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    total_steps = max(1, math.ceil(args.epochs * len(dl) / args.grad_accum))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    @torch.no_grad()
    def sb(texts: List[str]) -> torch.Tensor:
        v = sbert.encode(texts, convert_to_tensor=True, device=sbert_device)
        return F.normalize(v.float().to(device), dim=-1)

    print(f"[train] g_phi input = {args.gphi_input!r}; example: "
          f"{gphi_text(args.gphi_input, ds.records[0].backstory, ds.records[0].declared)[:120]!r}")

    os.makedirs(args.out, exist_ok=True)
    # Separate counters per phase: a step is either discriminative or
    # generative, so dividing both sums by the same `seen` (and multiplying by
    # grad_accum) reported numbers inflated by ~2 x grad_accum and made a
    # converging run look like it was diverging.
    step, run_disc, run_gen, seen = 0, 0.0, 0.0, 0
    n_disc, n_gen = 0, 0
    t0 = time.time()

    oom = 0
    for epoch in range(args.epochs):
        for batch in dl:
            discriminative = random.random() < args.disc_ratio

            try:
                summaries = sb([gphi_text(args.gphi_input, b["record"].backstory,
                                          b["record"].declared,
                                          [t.response for t in b["history"]])
                                for b in batch])
                prefixes = g_phi(summaries).to(dtype)

                losses = []
                for i, b in enumerate(batch):
                    rec, turn = b["record"], b["turn"]
                    prompt = build_prompt(tok, rec, b["history"], turn.user)
                    prefix = prefixes[i:i + 1]

                    if discriminative:
                        full, attn, _, _ = embed_with_prefix(
                            lm, tok, prompt, prefix, None, device, args.max_len)
                        out = lm(inputs_embeds=full, attention_mask=attn,
                                 output_hidden_states=True)
                        h = out.hidden_states[-1][:, -1, :].float()
                        del out
                        losses.append(("disc", f_theta(h), turn, rec))
                    else:
                        full, attn, n_ans, a_ids = embed_with_prefix(
                            lm, tok, prompt, prefix, turn.response, device,
                            args.max_len, args.max_answer_len)
                        # Only the answer positions need logits. A full-sequence
                        # logits tensor is seq_len x 128k vocab; at 900 tokens
                        # that is ~0.9 GiB in fp32, almost all of it discarded,
                        # and it was the largest allocation in the step.
                        kw = {_LOGITS_KW: n_ans + 1} if _LOGITS_KW else {}
                        logits = lm(inputs_embeds=full, attention_mask=attn, **kw).logits
                        pred = logits[:, -n_ans - 1:-1, :].float()
                        ce = F.cross_entropy(pred.transpose(1, 2), a_ids)
                        del logits, pred
                        losses.append(("gen", ce, turn, rec))

                if discriminative:
                    anchors = torch.cat([l[1] for l in losses], dim=0)
                    pos = sb([l[2].response for l in losses])
                    neg_texts = [turn.negative or rec.expected_sentence()
                                 for _, _, turn, rec in losses]
                    negs = torch.cat([sb(neg_texts), pos], dim=0)
                    loss = info_nce(anchors, pos, negs, args.temp)
                    run_disc += loss.item(); n_disc += 1
                else:
                    for p in f_theta.parameters():
                        p.requires_grad_(False)
                    loss = torch.stack([l[1] for l in losses]).mean()
                    run_gen += loss.item(); n_gen += 1

                (loss / args.grad_accum).backward()
                del losses, loss
                for p in f_theta.parameters():
                    p.requires_grad_(True)

            except torch.OutOfMemoryError:
                # Skip this batch rather than lose the run. Gradients already
                # accumulated from earlier micro-batches are dropped with them,
                # which is why the step counter is not advanced here.
                oom += 1
                for p in f_theta.parameters():
                    p.requires_grad_(True)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if oom <= 3 or oom % 25 == 0:
                    print(f"[oom] skipped batch ({oom} so far). If this repeats: "
                          f"--load-4bit --grad-checkpoint --batch 1 --max-len 512",
                          flush=True)
                # Every batch failing means the configuration cannot run at
                # all; continuing just burns the GPU while learning nothing.
                # Stop early and say exactly what to change.
                if step == 0 and oom >= 10:
                    free_b, total_b = torch.cuda.mem_get_info()
                    raise SystemExit(
                        f"\n[abort] the first {oom} batches all ran out of memory, so no "
                        f"optimizer step has happened.\n"
                        f"        free now: {free_b/2**30:.1f} GiB of {total_b/2**30:.1f} GiB"
                        + (f"; this process is capped at {args.mem_fraction:.0%} "
                           f"(~{args.mem_fraction*total_b/2**30:.1f} GiB) by --mem-fraction"
                           if args.mem_fraction > 0 else "") + "\n"
                        f"        current: --batch {args.batch} --max-len {args.max_len} "
                        f"--soft-tokens {args.soft_tokens}"
                        + (f" --lora-rank {args.lora_rank}" if args.lora else "") + "\n"
                        f"        try:     --batch 1 --grad-accum {args.batch * args.grad_accum} "
                        f"--max-len 512"
                        + ("  (and raise or drop --mem-fraction: the GPU is mostly free)"
                           if args.mem_fraction > 0 and free_b / total_b > 0.7 else "") + "\n")
                if oom > 100:
                    raise
                continue

            seen += 1
            if seen % 8 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()     # the neighbour can use what we freed
            if seen % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if step < total_steps:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step:5d}  "
                          f"InfoNCE {run_disc / max(1, n_disc):.4f}  "
                          f"CE {run_gen / max(1, n_gen):.4f}  "
                          f"{time.time() - t0:.0f}s", flush=True)
                    run_disc = run_gen = 0.0; n_disc = n_gen = 0   # window mean

        save(args, g_phi, f_theta, lm, tok, d_sbert, hidden, epoch)
        if torch.cuda.is_available():
            print(f"  [gpu] peak this epoch {torch.cuda.max_memory_allocated()/2**30:.1f} GiB "
                  f"· {oom} batches skipped for OOM")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        if args.bank:
            evaluate_bank(args, lm, tok, sbert, g_phi, device, dtype)

    print(f"done. checkpoint in {args.out}")


def save(args, g_phi, f_theta, lm, tok, d_sbert, hidden, epoch):
    os.makedirs(args.out, exist_ok=True)
    torch.save({"g_phi": g_phi.state_dict(),
                "f_theta": f_theta.state_dict(),
                "config": {"d_sbert": d_sbert, "hidden": hidden,
                           "n_tokens": args.soft_tokens, "sbert": args.sbert,
                           "model": args.model, "epoch": epoch,
                           "gphi_input": args.gphi_input,
                           "lora": bool(args.lora)}},
               os.path.join(args.out, "quirkymind.pt"))
    if args.lora:
        # Adapter weights only (a few MB). The demo re-attaches them to the
        # same base model with QUIRKYMIND_LORA=<this directory>.
        lm.save_pretrained(os.path.join(args.out, "lora"))
        print(f"  saved LoRA adapter -> {args.out}/lora  "
              f"(demo: QUIRKYMIND_LORA={args.out}/lora)")
    print(f"  saved epoch {epoch} -> {args.out}/quirkymind.pt")


# =====================================================================
# Per-epoch evaluation on the demo's own MCQA bank
# =====================================================================
@torch.no_grad()
def evaluate_bank(args, lm, tok, sbert, g_phi, device, dtype):
    """with-prefix vs no-prefix, stance withheld from the prompt, split into
    stance and backstory items — the demo's own (e)/(f) reading."""
    with open(args.bank, encoding="utf-8") as f:
        personas = json.load(f)

    def score(prompt_msgs, option, prefix):
        text = tok.apply_chat_template(prompt_msgs, tokenize=False,
                                       add_generation_prompt=True)
        p_ids = tok(text, return_tensors="pt", add_special_tokens=False,
                    truncation=True, max_length=args.max_len).input_ids.to(device)
        o_ids = tok(option, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(device)
        ids = torch.cat([p_ids, o_ids], dim=1)
        emb = lm.get_input_embeddings()(ids)
        if prefix is not None:
            emb = torch.cat([emb[:, :1, :], prefix.to(emb.dtype), emb[:, 1:, :]], dim=1)
        attn = torch.ones(emb.shape[:2], dtype=torch.long, device=device)
        n = o_ids.size(1)
        kw = {_LOGITS_KW: n + 1} if _LOGITS_KW else {}
        logits = lm(inputs_embeds=emb, attention_mask=attn, **kw).logits
        lp = logits[0, -n - 1:-1, :].float().log_softmax(-1)
        return lp.gather(-1, o_ids[0].unsqueeze(-1)).mean().item()

    agg = {"with prefix": [0, 0], "no prefix": [0, 0]}     # stance hits, stance total
    # Per-domain breakdown, Table-2 style, so one checkpoint trained on every
    # domain can still be read domain by domain — and a domain where the
    # prefix gain is negative shows up instead of averaging away.
    by_dom = {}
    for p in personas:
        dom = p.get("domain", "-")
        neutral = p.get("neutral_backstory") or p["identity"]
        text = gphi_text(args.gphi_input, p["backstory"], p.get("declared", ""))
        summary = F.normalize(
            sbert.encode([text], convert_to_tensor=True, device=device).float(), dim=-1)
        prefix = g_phi(summary).to(dtype)
        hits = {"with prefix": [0, 0], "no prefix": [0, 0]}   # [stance, bio]
        for item in p["probes"]:
            k = 0 if item.get("kind") == "stance" else 1
            lettered = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(item["options"]))
            msgs = [{"role": "system",
                     "content": f"{neutral.rstrip('.')}. You are answering as this "
                                f"person. Speak in first person, stay in character under "
                                f"pressure, and never break role."},
                    {"role": "user",
                     "content": f"{item['question']}\n{lettered}\n\nAnswer as this person "
                                f"would. Reply with the exact wording of the option you "
                                f"choose, and nothing else."}]
            for name, pre in (("with prefix", prefix), ("no prefix", None)):
                scores = [score(msgs, o, pre) for o in item["options"]]
                if int(max(range(len(scores)), key=lambda i: scores[i])) == item["target"]:
                    hits[name][k] += 1
                if k == 0:
                    ok = int(int(max(range(len(scores)), key=lambda i: scores[i])) == item["target"])
                    agg[name][1] += 1
                    agg[name][0] += ok
                    d = by_dom.setdefault(dom, {"with prefix": [0, 0], "no prefix": [0, 0]})
                    d[name][1] += 1
                    d[name][0] += ok
        ns = sum(1 for it in p["probes"] if it.get("kind") == "stance")
        nb = len(p["probes"]) - ns
        print(f"  [bank] {p['label'][:40]:42s} "
              f"with prefix  stance {hits['with prefix'][0]}/{ns} bio {hits['with prefix'][1]}/{nb} · "
              f"no prefix  stance {hits['no prefix'][0]}/{ns} bio {hits['no prefix'][1]}/{nb}")
    for name, (h, n) in agg.items():
        if n:
            print(f"  [bank] {name:12s} stance items overall: {h}/{n} = {h/n:.2f}")
    if len(by_dom) > 1:
        print("  [bank] by domain (stance items, with prefix / no prefix / gain):")
        for dom, d in sorted(by_dom.items()):
            hp, n = d["with prefix"]
            hn, _ = d["no prefix"]
            if n:
                print(f"         {dom:4s} {hp}/{n} = {hp/n:.2f}   {hn}/{n} = {hn/n:.2f}   {hp/n - hn/n:+.2f}")
    if agg["with prefix"][1]:
        gain = agg["with prefix"][0] / agg["with prefix"][1] - agg["no prefix"][0] / agg["no prefix"][1]
        print(f"  [bank] prefix gain on stance items: {gain:+.2f}  "
              f"({'steering is carrying the stance' if gain >= 0.10 else 'no real steering signal yet'})")


if __name__ == "__main__":
    main()