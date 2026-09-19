#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_banks.py — turn the seven ATP domain CSVs (GL, JN, MT, PL, RL, SC, SO)
into demo banks and training data, with a persona-level held-out split.

    python build_banks.py --csv-dir . --out banks/ --per-domain 40 --bank-per-domain 6

writes
    banks/<DOM>_bank.json      one demo bank per domain (held-out personas)
    banks/<DOM>_train.jsonl    training records for that domain
    banks/all_bank.json        every domain in one file — QUIRKYMIND_BANK=banks/all_bank.json
    banks/all_train.jsonl      every domain's training data — train_quirkymind.py --data
    banks/report.txt           what was found per domain (read this first)

------------------------------------------------------------------------
How an item is classified — the same three rules for every domain
------------------------------------------------------------------------
For each respondent and each survey item:

  STATED      the answer text appears in the backstory (e.g. the income band,
              "Conservative", "believes religion is losing influence" vs the
              option "Losing influence"). The probe can read it off the prompt:
              kind=bio, context="backstory". Retention test, open-book.
  INCONGRUOUS the answer is a MINORITY position in the respondent's group —
              held by at most --minority-max of the group (default 30%) and
              not the mode: kind=stance, context="none". The paper's
              low-likelihood stance. (On a 5-point scale almost every answer
              differs from the mode, so "not the mode" alone is too loose.)
  CONGRUOUS   the answer IS the group's modal answer: kind=bio, context="none".
              Predictable from the group prior; still unseen by the probe.
  (answers that are neither the mode nor a clear minority are skipped)

"Group" is the respondent's political ideology when the domain asks for it
(GL); otherwise every respondent sharing the same backstory line (JN, MT, SO
have ~60-190 distinct demographic lines); otherwise the whole population (PL,
RL, SC have one unique named backstory per persona). This is the paper's
p_{B,q}(r): the answer distribution GIVEN the backstory.

One incongruous item per persona becomes the seed pair (the paper's
"Example Q / Example A"): its question and answer go into the history, so it
is context="seed" (open-book). Every other item stays unseen. Only unseen
items are comparable to Table 2; the demo reports them first.

A persona enters the bank only if it has >= 2 incongruous items (one to
seed, one to probe unseen) and >= --min-items items in total. Stated items
are welcome but not required: JN and MT backstories name ideology and income
while their items ask about news habits and life satisfaction, so nothing
there is stated, and that is fine — those items are all unseen.

Meta items (survey form, tablet household) are dropped.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

DOMAINS = {"GL": "Global", "JN": "Journalism", "MT": "Methodology", "PL": "Politics",
           "RL": "Religion", "SC": "Science", "SO": "Social"}
IDEO = "What is your political ideology?"
META_RE = re.compile(r"survey form|tablet household|which survey", re.I)
NOANSWER = {"refused", "don't know/refused", "not sure", "none/other", "other", ""}
STOP = {"a", "an", "the", "of", "to", "in", "for", "and", "or", "is", "are", "my", "their",
        "than", "with", "on", "as", "at", "it", "its", "not", "very", "too", "off", "up"}


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
# JN, MT and SO ship every backstory with a dangling ", and reports economic
# situation N/A" — the wave has no economic-situation item, so the template
# filled in a placeholder. Left in, it is text the model must read as a fact
# about the person, and it shows up in the demo's persona panel.
_NA_CLAUSE = re.compile(r",?\s*and\s+reports\s+economic\s+situation\s+N/?A\.?\s*$", re.I)


def fix_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = (s.replace("?™", "'").replace("â€™", "'").replace("??", " ")
          .replace("  ", " ").strip())
    s = _NA_CLAUSE.sub("", s).rstrip(" ,")
    return s


def load(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    df["MCQ"] = df["MCQ"].map(fix_text)
    df["backstory"] = df["backstory"].map(fix_text)

    def decode(row):
        try:
            opts = [fix_text(o) for o in ast.literal_eval(row.options)]
        except Exception:
            return None
        return opts[int(row.label)] if 0 <= int(row.label) < len(opts) else None

    df["answer"] = df.apply(decode, axis=1)
    return df[~df["MCQ"].str.contains(META_RE)]


def option_lists(df: pd.DataFrame) -> Dict[str, List[str]]:
    out = {}
    for q in df.MCQ.unique():
        opts = [fix_text(o) for o in ast.literal_eval(df[df.MCQ == q].options.iloc[0])]
        out[q] = [o for o in opts if o.lower() not in NOANSWER]
    return out


def val(v) -> Optional[str]:
    if not isinstance(v, str):
        return None
    v = v.strip()
    return None if v.lower() in NOANSWER else v


# ------------------------------------------------------------------
# Rules
# ------------------------------------------------------------------
def content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9$,]+", s.lower()) if w not in STOP and len(w) > 1}


def stated_in(answer: str, backstory: str) -> bool:
    """Is the answer readable from the backstory text? Exact substring, or
    every content word of the answer appears in the backstory."""
    a, b = answer.lower(), backstory.lower()
    if a in b:
        return True
    cw = content_words(answer)
    return bool(cw) and all(w in b for w in cw)


def group_of(row: pd.Series, backstory: str, has_ideo: bool, backstory_counts: Counter) -> str:
    if has_ideo and val(row.get(IDEO)):
        return f"ideo:{val(row.get(IDEO))}"
    if backstory_counts.get(backstory, 0) >= 8:
        return f"back:{backstory}"
    return "all"


def group_dists(wide: pd.DataFrame, groups: pd.Series):
    """dist[question][group] = {answer: share}; groups with < 8 answers fall
    back to the population distribution. This is p_{B,q}(r)."""
    dist: Dict[str, Dict[str, Dict[str, float]]] = {}
    for q in wide.columns:
        col = wide[q].map(val).dropna()
        if col.empty:
            continue
        pop = (col.value_counts(normalize=True)).to_dict()
        dist[q] = {"all": pop}
        for g, idx in groups.groupby(groups).groups.items():
            sub = col.reindex(idx).dropna()
            dist[q][g] = sub.value_counts(normalize=True).to_dict() if len(sub) >= 8 else pop
    return dist


def mode_of(d: Dict[str, float]) -> Optional[str]:
    return max(d, key=d.get) if d else None


def persona_items(row, backstory, opts, dist, g, minority_max: float) -> Tuple[list, list, list]:
    """(stated, incongruous, congruous) lists of (question, answer, mode, share)."""
    stated, inc, con = [], [], []
    for q, options in opts.items():
        a = val(row.get(q))
        if a is None or a not in options:
            continue
        d = dist.get(q, {}).get(g, {})
        mode = mode_of(d)
        share = d.get(a, 0.0)
        if stated_in(a, backstory):
            stated.append((q, a, mode, share))
        elif mode is not None and a == mode:
            con.append((q, a, mode, share))
        elif mode is not None and share <= minority_max:
            inc.append((q, a, mode, share))
        # else: neither modal nor clearly minority — not informative, skipped
    return stated, inc, con


def clause(q: str, a: str) -> str:
    """Third-person clause for panel A: answers "<a>" to "<q>"."""
    q_ = q.rstrip("?.").strip()
    return f'answers "{a}" when asked "{q_}"'


# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------
def build_domain(code: str, path: str, args, rng: random.Random, report: list):
    df = load(path)
    wide = df.pivot_table(index="persona_id", columns="MCQ", values="answer", aggfunc="first")
    opts = option_lists(df)
    back = df.drop_duplicates("persona_id").set_index("persona_id")["backstory"].reindex(wide.index)
    has_ideo = IDEO in wide.columns
    bcounts = Counter(back.dropna())
    groups = pd.Series({pid: group_of(wide.loc[pid], back[pid] or "", has_ideo, bcounts)
                        for pid in wide.index})
    dist = group_dists(wide, groups)

    grouping = ("ideology" if has_ideo else
                "backstory line" if any(k.startswith("back:") for k in groups.unique())
                else "population")
    report.append(f"[{code} {DOMAINS[code]}] {len(wide)} personas · {len(opts)} items · "
                  f"grouping = {grouping} ({groups.nunique()} groups)")

    cands = []
    n_stated_hist, n_inc_hist = Counter(), Counter()
    for pid in wide.index:
        b = back[pid] or ""
        if not b:
            continue
        st, inc, con = persona_items(wide.loc[pid], b, opts, dist, groups[pid], args.minority_max)
        n_stated_hist[len(st)] += 1
        n_inc_hist[len(inc)] += 1
        if len(inc) >= 2 and len(st) + len(inc) + len(con) >= args.min_items:
            cands.append((pid, st, inc, con))
    report.append(f"    stated items per persona: " + ", ".join(f"{k}:{v}" for k, v in sorted(n_stated_hist.items())))
    report.append(f"    incongruous (<= {args.minority_max:.0%} of group) per persona: "
                  + ", ".join(f"{k}:{v}" for k, v in sorted(n_inc_hist.items())))
    report.append(f"    eligible (>=2 incongruous, >={args.min_items} items): {len(cands)}")
    if not cands:
        return [], []

    rng.shuffle(cands)
    cands = cands[: args.per_domain]
    n_h = max(args.bank_per_domain, int(round(len(cands) * args.heldout_frac)))
    heldout, train = cands[:n_h], cands[n_h:]
    report.append(f"    split: {len(train)} train · {len(heldout)} held-out · bank {min(args.bank_per_domain, len(heldout))}")

    bank, train_recs = [], []
    for split_name, pool in (("held-out", heldout), ("train", train)):
        for pid, st, inc, con in pool:
            b = back[pid]
            inc_sorted = sorted(inc, key=lambda it: it[3])          # rarest first
            seed_q, seed_a, seed_mode, seed_share = inc_sorted[0]
            declared = " and ".join(clause(q, a) for q, a, _, _ in inc_sorted[:2])
            expected = " and ".join(clause(q, m) for q, _, m, _ in inc_sorted[:2])
            # training record: every incongruous item is a turn, mode = hard negative
            turns = [{"user": q, "response": f"{a}.", "negative": f"{m}."} for q, a, m, _ in inc_sorted]
            for q, a, _, _ in st[:2]:
                turns.append({"user": q, "response": f"{a}."})
            if split_name == "train":
                train_recs.append({"backstory": b, "declared": declared, "expected": expected,
                                   "domain": code, "turns": turns})
                continue
            if len(bank) >= args.bank_per_domain:
                continue
            probes, nb, ns = [], 0, 0
            for q, a, _, _ in st[:args.stated_max]:
                nb += 1
                probes.append({"id": f"b{nb}", "kind": "bio", "question": q, "options": opts[q],
                               "target": opts[q].index(a), "context": "backstory"})
            for q, a, _, _ in con[:3]:
                nb += 1
                probes.append({"id": f"b{nb}", "kind": "bio", "question": q, "options": opts[q],
                               "target": opts[q].index(a), "context": "none"})
            for q, a, m, share in inc_sorted[:3]:
                ns += 1
                probes.append({"id": f"s{ns}", "kind": "stance", "question": q, "options": opts[q],
                               "target": opts[q].index(a),
                               "context": "seed" if q == seed_q else "none",
                               "mode": m, "group_share": round(share, 3)})
            n_unseen = sum(p["context"] == "none" for p in probes)
            n_unseen_st = sum(p["context"] == "none" and p["kind"] == "stance" for p in probes)
            n_open = len(probes) - n_unseen
            ident = b.split(".")[0].rstrip(".")
            # Label reads as the evaluation does: how many items the persona
            # must infer (unseen), how many of those are its heterodox
            # stances, and how many it can simply read off the prompt.
            bank.append({
                "label": f"{code} #{pid} · {n_unseen} unseen ({n_unseen_st} stance) · {n_open} open-book",
                "domain": code, "split": split_name,
                "identity": ident, "role": ident,
                "neutral_backstory": b if b.endswith(".") else b + ".",
                "backstory": b if b.endswith(".") else b + ".",
                "expected": expected, "declared": declared,
                "seed_question": seed_q, "seed_statement": f"{seed_a}",
                "probes": probes,
                "source": {"dataset": f"ATP {code}", "persona_id": int(pid) if str(pid).isdigit() else str(pid),
                           "group": groups[pid], "seed_mode_answer": seed_mode,
                           "seed_group_share": round(seed_share, 3)},
            })
    return bank, train_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=".")
    ap.add_argument("--out", default="banks")
    ap.add_argument("--domains", default="GL,JN,MT,PL,RL,SC,SO")
    ap.add_argument("--per-domain", type=int, default=40, help="max personas per domain")
    ap.add_argument("--bank-per-domain", type=int, default=6)
    ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--minority-max", type=float, default=0.30,
                    help="an answer counts as incongruous only if at most this share of "
                         "the respondent's group gives it")
    ap.add_argument("--stated-max", type=int, default=2,
                    help="max open-book items per persona (they prove alignment; the "
                         "unseen items carry the evaluation)")
    ap.add_argument("--min-items", type=int, default=4,
                    help="minimum usable items (stated + congruous + incongruous) per persona")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    report, all_bank, all_train = [], [], []
    for code in args.domains.split(","):
        path = os.path.join(args.csv_dir, f"{code}_test.csv")
        if not os.path.exists(path):
            report.append(f"[{code}] missing: {path}")
            continue
        bank, train = build_domain(code, path, args, rng, report)
        with open(os.path.join(args.out, f"{code}_bank.json"), "w", encoding="utf-8") as f:
            json.dump(bank, f, ensure_ascii=False, indent=1)
        with open(os.path.join(args.out, f"{code}_train.jsonl"), "w", encoding="utf-8") as f:
            for r in train:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_bank += bank
        all_train += train
        report.append(f"    wrote {len(bank)} bank personas, {len(train)} training records "
                      f"({sum(len(r['turns']) for r in train)} turns)")
        if bank:
            e = bank[0]
            report.append(f"    e.g. {e['label']}: seed Q = {e['seed_question'][:70]!r} -> {e['seed_statement']!r}")
    with open(os.path.join(args.out, "all_bank.json"), "w", encoding="utf-8") as f:
        json.dump(all_bank, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "all_train.jsonl"), "w", encoding="utf-8") as f:
        for r in all_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report.append(f"\nTOTAL: {len(all_bank)} bank personas · {len(all_train)} training records · "
                  f"{sum(sum(p['context'] == 'none' for p in e['probes']) for e in all_bank)} unseen items in the bank")
    text = "\n".join(report)
    print(text)
    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()