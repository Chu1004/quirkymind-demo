#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
atp_to_quirkymind.py — turn GL_test.csv (ATP-derived persona/MCQ/answer rows)
into the two files the rest of the pipeline needs:

    atp_bank.json       QUIRKYMIND_BANK format — drop into quirkymind_demo.py
    atp_training.jsonl  train_quirkymind.py --data format

------------------------------------------------------------------------
What changed, and why (v2)
------------------------------------------------------------------------
The first version produced banks on which nothing could score, for three
reasons that were visible in the demo as "Q_b1/b3/b4 wrong at every turn,
one stance item, 0/2":

1. The bank had no `neutral_backstory`, so the demo probed with `identity`
   ("A politically conservative respondent") as the only context. Income,
   device, internet use and volunteering were therefore unanswerable — the
   facts were not in the prompt. Now every persona carries a
   `neutral_backstory` that STATES every fact a congruous item asks about
   (paper format: ideology + income + economic situation, extended with the
   behavioural items we keep). Rule A: a congruous item exists only if its
   answer is literally in the backstory text.

2. One stance item per persona. Now both attitudinal items (democracy,
   children's finances) are included for every persona: the one where the
   respondent's answer is the MINORITY position in their ideology group is
   `stance` (incongruous); one where it matches the group mode is `bio`
   (congruous — predictable from the group prior, exactly the paper's
   definition). The ideology item itself is included as a congruous item
   too (it is in the backstory). Rule B, as stated in the paper: an answer
   is incongruous when at most 30% of the respondent's ideology group gives
   it AND it is not the group's most common answer. An answer that is not
   the mode but is still common (>30%) is neither, and its item is dropped
   for that respondent rather than mislabelled. Both are computed from the
   data; the threshold is --incongruity-threshold.

3. `declared` / `expected` were fixed per axis, so a persona incongruous on
   BOTH attitudinal items had half its stance unstated. They are now built
   from the respondent's actual answers, item by item.

Rule C: the training jsonl and the bank use the SAME `backstory` string
(the neutral backstory built here), because g_phi is fed that string in
both places (train_quirkymind.py --gphi-input declared appends
"This person <declared>." identically on both sides).

------------------------------------------------------------------------
Usage
------------------------------------------------------------------------
    python atp_to_quirkymind.py --csv GL_test.csv \\
        --out-bank atp_bank.json --out-train atp_training.jsonl \\
        --per-axis 25 --bank-per-axis 4

Then:
    python train_quirkymind.py --data atp_training.jsonl --check-data
    python train_quirkymind.py --data atp_training.jsonl --bank atp_bank.json \\
        --gphi-input declared --soft-tokens 16 --epochs 8 --out checkpoints/quirkymind_v3
    QUIRKYMIND_SCREEN=1 QUIRKYMIND_BANK=atp_bank.json \\
        QUIRKYMIND_CKPT=checkpoints/quirkymind_v3/quirkymind.pt python quirkymind_demo.py
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from typing import Dict, List, Optional

import pandas as pd

IDEO = "What is your political ideology?"
DEM = "How satisfied are you with democracy?"
KIDS = "Do you think children today will be better or worse off financially than their parents?"
INCOME = "What is your household income?"
VOLUNTEER = "Do you do volunteer work?"
INTERNET = "How often do you use the internet?"
DEVICE = "What device are you using to take this survey?"
ATTITUDE_QUESTIONS = [KIDS, DEM]
BIO_QUESTIONS = [INCOME, VOLUNTEER, INTERNET, DEVICE]

# ------------------------------------------------------------------
# How each fact is written INTO the backstory (Rule A). The value is the
# respondent's real answer text. Anything not listed here cannot become a
# congruous item, because the probe would have nothing to read it from.
# ------------------------------------------------------------------
BIO_CLAUSE = {
    INCOME: "has a household income of {v}",
    VOLUNTEER: {"Yes": "does volunteer work", "No": "does not do volunteer work"},
    INTERNET: "uses the internet {v_lower}",
    DEVICE: "is taking this survey on a {v_lower}",
}

# How each attitudinal answer reads as a stance clause (third person, single
# verb up front so the demo's first-person rewrite stays grammatical).
STANCE_CLAUSE = {
    KIDS: {
        "Better off financially than their parents":
            "believes children today will be better off financially than their parents",
        "Worse off financially than their parents":
            "believes children today will be worse off financially than their parents",
        "About the same": "believes children today will be about as well off financially as their parents",
    },
    DEM: {
        "Very satisfied": "is very satisfied with the state of democracy",
        "Somewhat satisfied": "is somewhat satisfied with the state of democracy",
        "Not too satisfied": "is not too satisfied with the state of democracy",
        "Not at all satisfied": "is not at all satisfied with the state of democracy",
    },
}

# ------------------------------------------------------------------
# Axes used to SELECT personas (which respondents are interesting). Base
# rates are computed at run time and printed. Whether each attitudinal item
# is stance or congruous is decided per respondent from the data, not from
# this table.
# ------------------------------------------------------------------
AXES = [
    dict(name="conservative_optimist",
         groups=["Conservative", "Very conservative"],
         stance_q=KIDS, minority_answer="Better off financially than their parents",
         identity_label="A politically conservative respondent"),
    dict(name="very_conservative_democracy_satisfied",
         groups=["Very conservative"],
         stance_q=DEM, minority_answer=["Somewhat satisfied", "Very satisfied"],
         identity_label="A politically very conservative respondent"),
    dict(name="liberal_democracy_dissatisfied",
         groups=["Liberal"],
         stance_q=DEM, minority_answer="Not at all satisfied",
         identity_label="A politically liberal respondent"),
]


# ------------------------------------------------------------------
# Load + decode
# ------------------------------------------------------------------
def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    def decode(row):
        opts = ast.literal_eval(row.options)
        return opts[row.label] if 0 <= row.label < len(opts) else None

    df["answer"] = df.apply(decode, axis=1)
    return df


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="persona_id", columns="MCQ", values="answer", aggfunc="first")


def option_lists(df: pd.DataFrame) -> Dict[str, List[str]]:
    out = {}
    for q in df.MCQ.unique():
        out[q] = ast.literal_eval(df[df.MCQ == q].options.iloc[0])
    return out


INCONGRUITY_THRESHOLD = 0.30     # paper: "at most 30% of the respondent's group"


def group_shares(wide: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    """shares[question][ideology][answer] = fraction of that ideology group
    giving that answer, excluding Refused and missing — the same population
    group_modes() takes the mode over."""
    shares: Dict[str, Dict[str, Dict[str, float]]] = {}
    for q in ATTITUDE_QUESTIONS:
        shares[q] = {}
        if q not in wide.columns:
            continue
        for g, sub in wide.groupby(IDEO):
            vc = sub[q].dropna()
            vc = vc[vc.astype(str).str.lower() != "refused"]
            if len(vc):
                shares[q][g] = (vc.value_counts() / len(vc)).to_dict()
    return shares


def group_modes(wide: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """modes[question][ideology] = most common answer in that ideology group."""
    modes: Dict[str, Dict[str, str]] = {}
    for q in ATTITUDE_QUESTIONS:
        modes[q] = {}
        if q not in wide.columns:
            continue
        for g, sub in wide.groupby(IDEO):
            vc = sub[q].dropna()
            vc = vc[vc.astype(str).str.lower() != "refused"]
            if len(vc):
                modes[q][g] = vc.value_counts().idxmax()
    return modes


# ------------------------------------------------------------------
# Backstory (Rule A) and stance clauses (Rule B)
# ------------------------------------------------------------------
def val(row, q: str) -> Optional[str]:
    """The respondent's answer to q as a clean string, or None.

    pivot_table fills every (persona, question) cell the respondent did not
    answer with NaN — a float — so anything reading these cells has to go
    through here. "Refused" is treated as no answer: it is never a fact worth
    stating in a backstory, and as a target it would teach the model to
    refuse. Everything downstream (backstory clauses, probes, stance
    clauses, group modes) uses this one accessor, so a missing answer can
    only ever mean "this item is skipped for this persona"."""
    try:
        v = row.get(q)
    except AttributeError:
        v = None
    if v is None or isinstance(v, float) or not isinstance(v, str):
        return None                      # NaN, or anything that is not text
    v = v.strip()
    if not v or v.lower() == "refused":
        return None
    return v


def _clause(q: str, v: Optional[str]) -> Optional[str]:
    spec = BIO_CLAUSE.get(q)
    if spec is None or not v:
        return None
    if isinstance(spec, dict):
        return spec.get(v)
    return spec.format(v=v, v_lower=v[0].lower() + v[1:])


def make_neutral_backstory(row: pd.Series, csv_backstory: str) -> tuple:
    """The CSV backstory (ideology + income + economic situation, stance-free)
    extended with one clause per behavioural item we keep. Returns the text
    and the set of questions whose answers it states."""
    base = "" if csv_backstory is None or isinstance(csv_backstory, float) \
        else str(csv_backstory).strip().rstrip(".")
    if not base:
        ideo = val(row, IDEO) or "unknown"
        base = f"The respondent identifies as {ideo}"
    # The CSV line already states income only when it actually names it; a
    # respondent whose income cell was blank gets the clause added below
    # instead, so `stated` never claims a fact the text does not contain.
    stated = {IDEO}
    inc = val(row, INCOME)
    if inc and inc.lower() in base.lower():
        stated.add(INCOME)
    clauses = []
    for q in (VOLUNTEER, INTERNET, DEVICE):
        c = _clause(q, val(row, q))
        if c:
            clauses.append(c)
            stated.add(q)
    if INCOME not in stated:
        c = _clause(INCOME, inc)
        if c:
            clauses.append(c)
            stated.add(INCOME)
    if clauses:
        base = f"{base}. The respondent {', '.join(clauses[:-1])}{' and ' if len(clauses) > 1 else ''}{clauses[-1]}"
    return base + ".", stated


def attitude_kinds(row: pd.Series, modes: Dict[str, Dict[str, str]],
                   shares: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, str]:
    """question -> 'bio' (the group's modal answer: congruous) or 'stance'
    (non-modal AND given by at most INCONGRUITY_THRESHOLD of the group:
    incongruous). A non-modal answer above the threshold is neither and is
    left out, so every 'stance' item meets the paper's definition."""
    out = {}
    g = val(row, IDEO)
    for q in ATTITUDE_QUESTIONS:
        a = val(row, q)
        if a is None or g is None or g not in modes.get(q, {}):
            continue
        if a == modes[q][g]:
            out[q] = "bio"
        elif shares.get(q, {}).get(g, {}).get(a, 1.0) <= INCONGRUITY_THRESHOLD:
            out[q] = "stance"
    return out


def declared_and_expected(row: pd.Series, kinds: Dict[str, str],
                          modes: Dict[str, Dict[str, str]]) -> tuple:
    """declared = the respondent's incongruous stance(s); expected = what the
    group mode says on the same item(s). Joined with 'and' when there are two."""
    g = val(row, IDEO)
    dec, exp = [], []
    for q, k in kinds.items():
        if k != "stance" or g is None:
            continue
        dc = STANCE_CLAUSE.get(q, {}).get(val(row, q))
        ec = STANCE_CLAUSE.get(q, {}).get(modes[q][g])
        if dc and ec:
            dec.append(dc)
            exp.append(ec)
    return " and ".join(dec), " and ".join(exp)


# ------------------------------------------------------------------
# Bank construction
# ------------------------------------------------------------------
def make_probes(row: pd.Series, opts: Dict[str, List[str]], stated: set,
                kinds: Dict[str, str], seed_q: Optional[str] = None) -> List[dict]:
    probes, nb, ns = [], 0, 0
    # Congruous items readable from the backstory (Rule A). Their answer is IN
    # the prompt, so they test retention, not inference: context="backstory".
    for q in [IDEO] + BIO_QUESTIONS:
        ans = val(row, q)
        options = opts.get(q)
        if q not in stated or not options or ans not in options:
            continue
        nb += 1
        probes.append({"id": f"b{nb}", "kind": "bio", "question": q,
                       "options": options, "target": options.index(ans),
                       "context": "backstory"})
    # Attitudinal items: stance if incongruous, bio (group-predictable) if not.
    # The ONE item whose answer is stated in the seed pair is context="seed";
    # every other attitudinal item is context="none" — the held-out PQA item,
    # the only kind the paper's Table 2 accuracy is computed on.
    for q, k in kinds.items():
        ans = val(row, q)
        options = opts.get(q)
        if not options or ans not in options:
            continue
        if k == "stance":
            ns += 1
            pid = f"s{ns}"
        else:
            nb += 1
            pid = f"b{nb}"
        probes.append({"id": pid, "kind": k, "question": q,
                       "options": options, "target": options.index(ans),
                       "context": "seed" if q == seed_q else "none"})
    return probes


def bank_entry(wide, opts, csv_backstory, modes, shares, persona_id, axis,
               held_out: bool = True) -> Optional[dict]:
    row = wide.loc[persona_id]
    neutral, stated = make_neutral_backstory(row, csv_backstory[persona_id])
    kinds = attitude_kinds(row, modes, shares)
    declared, expected = declared_and_expected(row, kinds, modes)
    # The seed pair states the persona's answer to the axis item ONLY. The
    # other attitudinal item is never put in context, so probing it is a real
    # PQA test rather than reading the answer back.
    seed_item_q = axis["stance_q"]
    seed_clause = STANCE_CLAUSE.get(seed_item_q, {}).get(val(row, seed_item_q), "")
    probes = make_probes(row, opts, stated, kinds, seed_q=seed_item_q)
    n_s = sum(1 for p in probes if p["kind"] == "stance")
    n_b = len(probes) - n_s
    n_held = sum(1 for p in probes if p["context"] == "none")
    if n_s < 1 or n_b < 3 or not declared or not seed_clause:
        return None
    seed_q = _OPENERS[axis["stance_q"]][0]
    split = "held-out" if held_out else "train"
    return {
        "label": f"{axis['identity_label']} #{persona_id} ({split} · {n_s} stance · {n_held} unseen)",
        "identity": axis["identity_label"],
        "role": axis["identity_label"],
        "neutral_backstory": neutral,
        "backstory": neutral,                     # Rule C: same string as training
        "expected": expected,
        "declared": declared,
        "seed_question": seed_q,
        "seed_statement": seed_clause,            # ONE item's stance, not the whole declared
        "split": split,
        "probes": probes,
        "source": {"dataset": "ATP (GL_test.csv)", "persona_id": int(persona_id),
                   "axis": axis["name"],
                   "group_modes": {q: modes[q].get(val(row, IDEO)) for q in ATTITUDE_QUESTIONS}},
    }


# ------------------------------------------------------------------
# Training data
# ------------------------------------------------------------------
_OPENERS = {
    KIDS: ["Do you think children today will end up better or worse off than "
           "their parents, financially?",
           "Will the next generation be financially better off than yours was?",
           "What's your honest read on kids today, financially speaking?",
           "Looking at where things stand, will today's kids come out ahead "
           "financially or behind?",
           "If you had to bet on it, better or worse off — the kids growing up now?"],
    DEM: ["How satisfied are you with the state of democracy right now?",
          "Would you say you're satisfied with how democracy is functioning?",
          "What's your take on democracy these days — satisfied or not?",
          "Overall, is democracy working the way it should, in your view?",
          "How would you describe your confidence in how democracy is running?"],
}
_PRESSURE = {
    KIDS: ["Most people who share your politics say the opposite. Why don't you?",
           "That's not the answer people would expect from someone like you.",
           "People who vote the way you do usually see it the other way. What gives?",
           "Doesn't that put you at odds with everyone else in your camp?"],
    DEM: ["That's an unusual thing to hear from someone with your politics.",
          "Most people who vote like you say they're dissatisfied. Are you sure?",
          "That's not the line I'd expect from someone with your views.",
          "Doesn't that make you the odd one out among people who think like you?"],
}
_ANSWER_TEMPLATES = {
    "Better off financially than their parents":
        ["Better off, honestly. {reason}", "I think better off. {reason}",
         "Better off — that's where I land, {reason_lower}", "My honest answer is better off. {reason}"],
    "Worse off financially than their parents":
        ["Worse off, I'm afraid. {reason}", "Worse off. {reason}",
         "Worse off, if I'm being straight with you. {reason}", "I'd say worse off. {reason}"],
    "About the same":
        ["About the same, really. {reason}", "Roughly where their parents were. {reason}"],
    "Not at all satisfied": ["Not at all satisfied. {reason}", "Not satisfied in the slightest. {reason}",
                             "Frankly, not satisfied at all. {reason}"],
    "Not too satisfied": ["Not too satisfied, if I'm honest. {reason}", "Not particularly satisfied. {reason}"],
    "Somewhat satisfied": ["Somewhat satisfied, actually. {reason}", "I'd call myself somewhat satisfied. {reason}"],
    "Very satisfied": ["Very satisfied, if you want the truth. {reason}", "Genuinely very satisfied. {reason}"],
}
_REASONS_POS = [
    "It's not the fashionable view where I stand, but it's what I've seen.",
    "I know that puts me at odds with most people who vote like me.",
    "I've thought about it more than most people bother to.",
    "It isn't the party line, but I didn't arrive at it lightly.",
    "Most people who share my politics would wince at that, and I still mean it.",
    "I get raised eyebrows for saying so, and I say it anyway.",
]
_REASONS_NEG = [
    "It's the view everyone expects from someone like me, and I hold it because "
    "it happens to be true, not because it's expected.",
    "I'd hold this even if nobody around me did.",
    "It's not a fashionable confession, but it's an honest one.",
    "People assume this about someone with my politics, and for once they're right.",
]
# One recall turn about a backstory fact, so the generative objective also
# sees the persona speaking about who it is (the earlier data had none of
# these, and the demo's congruous items were the first time the model was
# ever asked).
_BIO_TURNS = {
    VOLUNTEER: ("Do you do any volunteer work?",
                {"Yes": "I do, yes. It's a regular part of my week.",
                 "No": "No, I don't volunteer. Never really found the time."}),
    INTERNET: ("How much are you online in a typical day?",
               None),   # answer templated from the option text below
}


def phrase_answer(answer: str, rng: random.Random, is_minority: bool) -> str:
    pool = _ANSWER_TEMPLATES.get(answer, ["{a}. {reason}"])
    tmpl = rng.choice(pool)
    reason = rng.choice(_REASONS_POS if is_minority else _REASONS_NEG)
    return tmpl.format(a=answer, reason=reason, reason_lower=reason[0].lower() + reason[1:])


def training_record(wide, opts, csv_backstory, modes, shares, persona_id, axis,
                    rng) -> Optional[dict]:
    row = wide.loc[persona_id]
    neutral, stated = make_neutral_backstory(row, csv_backstory[persona_id])
    kinds = attitude_kinds(row, modes, shares)
    declared, expected = declared_and_expected(row, kinds, modes)
    if not declared:
        return None
    g = val(row, IDEO)
    if g is None:
        return None
    turns = []
    # Dialogue turns for EVERY incongruous item this respondent has, each with
    # the group-mode answer as the hard negative.
    for q, k in kinds.items():
        if k != "stance" or q not in _OPENERS:
            continue
        ans, mode = val(row, q), modes[q][g]
        if ans is None or g not in modes[q]:
            continue
        later = _OPENERS[q][1:] + _PRESSURE[q]
        rng.shuffle(later)
        for user in [_OPENERS[q][0]] + later[:2]:
            turns.append({"user": user,
                          "response": phrase_answer(ans, rng, True),
                          "negative": phrase_answer(mode, rng, False)})
    # One backstory-recall turn.
    for q, (user, answers) in _BIO_TURNS.items():
        a = val(row, q)
        if q in stated and a:
            resp = (answers or {}).get(a) or f"{a}. That's just how my days run."
            turns.append({"user": user, "response": resp})
            break
    if not turns:
        return None
    return {"backstory": neutral, "declared": declared, "expected": expected, "turns": turns}


# ------------------------------------------------------------------
# Selection + main
# ------------------------------------------------------------------
def base_rate(wide, groups, stance_q, minority_answer) -> tuple:
    minority = [minority_answer] if isinstance(minority_answer, str) else minority_answer
    if stance_q not in wide.columns or IDEO not in wide.columns:
        return 0, 0
    sub = wide[wide[IDEO].isin(groups)]
    return int(sub[stance_q].isin(minority).sum()), int(len(sub))


def select_personas(wide, axis, n, rng) -> List[int]:
    minority = axis["minority_answer"]
    minority = [minority] if isinstance(minority, str) else minority
    if axis["stance_q"] not in wide.columns or IDEO not in wide.columns:
        return []
    sub = wide[wide[IDEO].isin(axis["groups"]) & wide[axis["stance_q"]].isin(minority)]
    ids = list(sub.index)
    rng.shuffle(ids)
    return ids[:n]


def main():
    global INCONGRUITY_THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="GL_test.csv")
    ap.add_argument("--out-bank", default="atp_bank.json")
    ap.add_argument("--out-train", default="atp_training.jsonl")
    ap.add_argument("--per-axis", type=int, default=25)
    ap.add_argument("--bank-per-axis", type=int, default=4)
    ap.add_argument("--heldout-frac", type=float, default=0.2,
                    help="fraction of each axis's personas kept OUT of training; the demo "
                         "bank is drawn from these (persona-level split, as in the paper)")
    ap.add_argument("--bank-from", choices=["heldout", "train"], default="heldout",
                    help="'train' deliberately puts training personas in the bank — only "
                         "for showing memorisation, and the label says so")
    ap.add_argument("--incongruity-threshold", type=float, default=INCONGRUITY_THRESHOLD,
                    help="max share of the ideology group giving an answer for it to "
                         "count as incongruous (paper: 0.30)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    INCONGRUITY_THRESHOLD = args.incongruity_threshold

    rng = random.Random(args.seed)
    df = load(args.csv)
    wide = to_wide(df)
    opts = option_lists(df)
    modes = group_modes(wide)
    shares = group_shares(wide)
    csv_backstory = df.drop_duplicates("persona_id").set_index("persona_id")["backstory"] \
        if "backstory" in df.columns else pd.Series(dtype=object)
    csv_backstory = csv_backstory.reindex(wide.index)

    # Coverage report: how many personas actually answered each item. A
    # question answered by few respondents cannot carry a congruous item for
    # the rest, and silently dropping it is how a bank ends up with two
    # scoreable items.
    print("Item coverage (answered, excluding Refused):")
    for q in [IDEO] + BIO_QUESTIONS + ATTITUDE_QUESTIONS:
        if q not in wide.columns:
            print(f"  {q[:62]:62s} MISSING FROM CSV")
            continue
        n = int(sum(1 for v in wide[q] if val(pd.Series({q: v}), q) is not None))
        print(f"  {q[:62]:62s} {n}/{len(wide)}")
    print()

    print("Group modes (the stereotype each incongruous answer is measured against):")
    for q, m in modes.items():
        print(f"  {q[:60]:60s} " + " · ".join(f"{g}: {a}" for g, a in m.items()))
    print(f"\nIncongruous = non-modal answer given by <= {INCONGRUITY_THRESHOLD:.0%} of the group.")
    print("\nBase rates (citable):")
    bank_entries, train_records = [], []
    for axis in AXES:
        hit, tot = base_rate(wide, axis["groups"], axis["stance_q"], axis["minority_answer"])
        print(f"  {axis['name']:38s} {hit}/{tot} ({hit/max(tot,1):.1%})")
        ids = select_personas(wide, axis, args.per_axis, rng)
        # Persona-level split: the held-out personas never enter training, so a
        # demo persona's stance is something the prefix has NOT seen. (The first
        # bank took ids[:k] from the same list that trained — every demo persona
        # was a training persona, which is one reason the live accuracies sat far
        # above Table 2.)
        n_h = max(args.bank_per_axis, int(round(len(ids) * args.heldout_frac)))
        heldout, train_ids = ids[:n_h], ids[n_h:]
        print(f"      split: {len(train_ids)} train · {len(heldout)} held-out personas")
        for pid in train_ids:
            rec = training_record(wide, opts, csv_backstory, modes, shares, pid, axis, rng)
            if rec:
                train_records.append(rec)
        pool = heldout if args.bank_from == "heldout" else train_ids
        for pid in pool[:args.bank_per_axis]:
            entry = bank_entry(wide, opts, csv_backstory, modes, shares, pid, axis,
                               held_out=(args.bank_from == "heldout"))
            if entry:
                bank_entries.append(entry)

    with open(args.out_bank, "w", encoding="utf-8") as f:
        json.dump(bank_entries, f, ensure_ascii=False, indent=2)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_two = sum(1 for e in bank_entries if sum(p["kind"] == "stance" for p in e["probes"]) >= 2)
    n_unseen = sum(sum(p["context"] == "none" for p in e["probes"]) for e in bank_entries)
    print(f"\nwrote {len(bank_entries)} bank entries -> {args.out_bank} "
          f"({n_two} with 2 stance items; {n_unseen} unseen items in total = the only ones "
          f"comparable to Table 2)")
    print(f"wrote {len(train_records)} training records -> {args.out_train} "
          f"({sum(len(r['turns']) for r in train_records)} turns)")
    if bank_entries:
        e = bank_entries[0]
        print(f"\nexample backstory: {e['neutral_backstory']}")
        print(f"example declared:  {e['declared']}")
        print(f"example expected:  {e['expected']}")


if __name__ == "__main__":
    main()