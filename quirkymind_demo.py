#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QuirkyMind — talk to the persona, watch its answer sheet move.
(AAAI-27 Demo Track)

------------------------------------------------------------------------
Vocabulary used everywhere in this file and in the paper
------------------------------------------------------------------------
A persona is two things, and every panel keeps them apart by colour:

    BACKSTORY  (blue)   — stable identity and biographical facts: who this
                          person is. This is the part a stereotype would
                          predict from (the paper's B / token stream X).
    STANCE     (purple) — the one declared position that contradicts what
                          people with that backstory are assumed to hold
                          (the paper's target r_t). The incongruity.

In the paper's terms (Sec. 2.1) the backstory half is what the demographic
prior predicts — its items are CONGRUOUS (ids c1, c2, ... shown as Q_c1) —
and the stance half is INCONGRUOUS (ids i1, i2, ... shown as Q_i1). The
hand-authored legacy personas keep b*/s* ids. Panels A–C use only the short
ids; panel D ("Item key") carries the full question text so nothing is ever
shown truncated.

Inference follows the paper's three regimes (Sec. 4.2, App. F): No-Turn =
backstory only; Single-Turn = backstory + one context pair; Multi-Turn =
backstory + progressive history + trained prefix (QuirkyMind). The stance is
never restated in the prompt — it has to survive through history and latent.

------------------------------------------------------------------------
What changed from v6, and why
------------------------------------------------------------------------
v6 was a five-card diagnostic console with thirteen output panels and three
proxy metrics (SBERT cosine, distinct-n, latent drift). It measured the
persona with quantities nobody in the room could interpret, and it never
showed the thing the paper actually claims: that the steered persona is
faithful to BOTH its backstory and its MCQA stance profile, and that this
faithfulness decays in an observable, recoverable way under conversation.

This version is one screen and one idea:

    a chat on the left, the persona's answer sheet on the right.

The answer sheet is the MCQA bank the persona was aligned to. Every turn,
the agent silently re-answers those items *from inside the current
conversation*, by scoring each option's length-normalised log-likelihood
under the persona prefix plus the live dialogue history. No free-text
parsing, no embedding proxy, no calibration hand-waving. An item either
still matches the target answer or it doesn't.

That gives the demo three things v6 could not state plainly:

  * ALIGNMENT, at turn 0. Before anyone types, the sheet is 100% green.
    That IS the claim "we aligned a persona matching backstory + MCQA".
  * DRIFT, as a legible event. Not "fidelity 0.412". Instead: "at turn 4 he
    stopped answering Q3 the way he did at turn 0, and here is the message
    that did it."
  * RECOVERY, as the same mechanism. PNI refinement is not a separate panel
    — it is a step in the agent loop that fires when the sheet goes red, and
    you watch the item flip back.

Bio items and stance items are probed identically but reported separately,
because they fail differently: backstory facts decay through context
dilution, declared stances collapse through social pressure.

------------------------------------------------------------------------
The agent loop (one conversational turn)
------------------------------------------------------------------------
    respond  ->  audit  ->  [refine -> audit] * k  ->  commit

Refinement only fires when the audit finds a flipped item, so a persona
that holds its ground costs one generation and one probe pass. The trace of
this loop is shown verbatim under the reply; it is the honest internal
account of what the system did, not a decorative "thinking" animation.

------------------------------------------------------------------------
Measurement honesty
------------------------------------------------------------------------
Consistency here is agreement with the persona's own turn-0 target answers,
measured in-demo at conversation speed on a small bank. It is not the
paper's offline IND / EXAG or human evaluation, and a small bank is noisy.
The UI and the export both say so.

------------------------------------------------------------------------
Running
------------------------------------------------------------------------
    pip install "gradio>=4.44" plotly scikit-learn transformers torch
    python quirkymind_demo.py

UI-only rehearsal (no GPU, no downloads; generation and probing stubbed but
deterministic — the numbers are theatre, the layout is real):

    QUIRKYMIND_MOCK=1 python quirkymind_demo.py

Env: QUIRKYMIND_MODEL, QUIRKYMIND_PORT.
Persona banks can be loaded from JSON: QUIRKYMIND_BANK=/path/to/bank.json
(schema at the bottom of this file).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gradio as gr
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

# =====================================================================
# Config
# =====================================================================
MOCK = os.environ.get("QUIRKYMIND_MOCK", "0") == "1"

# Gradio 6 removed the tuple chatbot format entirely — messages format is the
# only option and the `type` kwarg itself was removed along with it. Gradio
# 4.44–5.x need type="messages" explicitly. Older Gradio needs the legacy
# [[user, bot], ...] tuple format. Detect once, at import time.
try:
    _GR_MAJOR = int(gr.__version__.split(".")[0])
except Exception:
    _GR_MAJOR = 4
if _GR_MAJOR >= 6:
    CHAT_KW, CHAT_MESSAGES = {}, True
elif _GR_MAJOR >= 4:
    CHAT_KW, CHAT_MESSAGES = {"type": "messages"}, True
else:
    CHAT_KW, CHAT_MESSAGES = {}, False
MODEL_NAME = os.environ.get("QUIRKYMIND_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
BANK_PATH = os.environ.get("QUIRKYMIND_BANK", "")
PORT = int(os.environ.get("QUIRKYMIND_PORT", "7860"))

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_LEN = 1024
MOCK_HIDDEN = 64

# Inference protocol. "paper" follows the paper (Sec. 4.2, App. F): the persona
# prompt is the demographic backstory B only; stances enter through in-context
# Q->A history and the trained prefix. The three regimes are No-Turn (B, no
# history), Single-Turn (B + one context pair) and Multi-Turn (B + progressive
# history + prefix = QuirkyMind). "legacy" is the pre-alignment behaviour,
# where the full spec including the stance was re-sent every turn — under
# which no text-conditioned regime can drift and no gain is measurable.
PROTOCOL = os.environ.get("QUIRKYMIND_PROTOCOL", "paper")
# Probe prompt: "paper" renders the PQA template of App. F.3 (Persona /
# previous statement / Question / Options / Answer) as one user message;
# "chat" uses the chat-native probe.
# Default "chat": train_quirkymind.py trains the prefix behind a chat-template
# system message, and evaluate_bank() scores the same way, so the demo probes
# the format the prefix was trained in. "paper" renders App. F.3 instead.
PROBE_TEMPLATE = os.environ.get("QUIRKYMIND_TEMPLATE", "chat")

if not MOCK:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    # QUIRKYMIND_4BIT=1 loads the backbone in NF4 (~6GB instead of ~16GB) so
    # the demo fits beside another job on a shared GPU. The adapter trained
    # with --load-4bit attaches to either precision; scores shift slightly
    # between them, so keep one setting for a whole recording session.
    _load_kw = dict(device_map="auto", torch_dtype=DTYPE)
    if os.environ.get("QUIRKYMIND_4BIT", "0") == "1":
        from transformers import BitsAndBytesConfig
        _load_kw.pop("torch_dtype")
        _load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)
        print("[QuirkyMind] backbone in NF4 4-bit (QUIRKYMIND_4BIT=1)")
    base_lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **_load_kw)
else:
    tok = None
    base_lm = None


def _seed(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "little")


def clean_text(t: str) -> str:
    """Strip stray symbols but KEEP asterisks and a few sentence marks.

    The old pattern dropped `*`, which silently destroyed the roleplay markers
    models emit — `*sighs, looking around the office*` came out as bare words
    welded onto the front of the reply. Those markers are the signal the
    expression readout is built on, so they have to survive."""
    c = re.sub(r"[^a-zA-Z0-9\s\.,!?;:'\-\*]", "", t).strip()
    c = re.sub(r"\*{3,}", "*", c)
    return c if c else t.strip()


# =====================================================================
# Persona = backstory + an MCQA bank it was aligned to
# =====================================================================
@dataclass
class Probe:
    """One MCQA item the persona is supposed to answer a specific way.

    `target` is not ground truth about the world — it is the answer this
    persona was aligned to give. Bio items come from the backstory; stance
    items come from the MCQA profile.
    """
    id: str
    kind: str              # "bio" | "stance"
    question: str
    options: List[str]
    target: int
    # Where the answer is available when the item is probed:
    #   "backstory"  stated in the backstory text        (retention, open-book)
    #   "seed"       stated in the seed pair              (open-book)
    #   "none"       nowhere in the prompt — the held-out PQA item, the only
    #                kind whose accuracy is comparable to the paper's Table 2
    context: str = "none"
    mode: str = ""          # the group's modal answer for this item, when known

    def target_text(self) -> str:
        return self.options[self.target]

    @property
    def held_out(self) -> bool:
        return self.context == "none"


# Display names for the two halves. Internally the kinds stay "bio" and
# "stance" (bank files on disk use them); everything the audience reads says
# backstory / stance.
# Paper Sec. 2.1: an item is CONGRUOUS when the persona's answer is the
# high-probability answer for its demographic group, INCONGRUOUS when it sits
# in the low-likelihood tail yet is the persona's true stance. In the UI the
# congruous half is what the backstory alone predicts (blue) and the
# incongruous half is the declared stance (purple).
KIND_NAME = {"bio": "congruous", "stance": "incongruous"}


def qid(item_or_id) -> str:
    """Plain-text item id: Q_b1, Q_s3. Used in toasts and exports."""
    i = item_or_id.id if isinstance(item_or_id, Probe) else str(item_or_id)
    return f"Q_{i}"


def qid_html(item: "Probe", color: bool = True) -> str:
    """Item id for HTML / Markdown views: Q<sub>b1</sub>, coloured by the half
    it probes (blue = backstory, purple = stance)."""
    cls = "qm-bs" if item.kind == "bio" else "qm-st"
    core = f"Q<sub>{item.id}</sub>"
    return f'<span class="{cls}">{core}</span>' if color else core


def qid_plotly(item: "Probe") -> str:
    """Item id for plotly tick labels / annotations. Plotly accepts <sub> and
    <span style> but not classes, so the colour is inlined (and passes through
    the greyscale switch like every other chart colour)."""
    col = BACKSTORY_INK if item.kind == "bio" else STANCE_INK
    return f'<span style="color:{_c(col)}">Q<sub>{item.id}</sub></span>'


@dataclass
class Persona:
    label: str
    identity: str
    backstory: str
    expected: str          # the stance this identity is assumed to hold
    declared: str          # the stance it actually holds
    probes: List[Probe]
    # The backstory with the declared-stance clause removed, and nothing else
    # removed. Probing needs to withhold the answer to the stance items while
    # still supplying the biography the bio items ask about — cutting back to
    # the identity line alone made items like "have you ever been arrested"
    # unanswerable from any source, so they read as drift when they are really
    # just unsourced.
    neutral_backstory: str = ""
    seed_question: str = ""          # the "Example Q" of the one context pair
    seed_statement: str = ""         # the stance stated in that pair — ONE item's, not all
    domain: str = ""                 # ATP topic domain code (GL, JN, ...) or "" for built-ins
    script: List[dict] = field(default_factory=list)   # optional scripted beats
    # A bare role label with no biographical specifics. The stereotype control
    # primes an identity in order to see what that identity alone produces, so
    # it must not carry facts the bank asks about — priming with "an activist
    # who has organised for a decade" hands over the answer to "how long have
    # you been organising", and the item then looks non-diagnostic when it is
    # really just leaked.
    role: str = ""

    def stereotype_system_text(self) -> str:
        """The same identity, primed with the stance people ASSUME it holds.

        The third control. An item earns its place only if it separates three
        things: what the bare model says, what a stereotype-primed persona
        says, and what this persona is supposed to say. If the stereotype
        already produces the target answer, the item cannot show incongruity
        steering — it just shows the model can read an identity label."""
        # Phrased so the stance clause is never conjugated into the sentence
        # ("people in this position oppose X" reads correctly for any verb,
        # where "This person oppose X" does not).
        who = self.role or self.identity
        return (f"{who.rstrip('.')}. Like most people in this position, "
                f"they {self.expected.rstrip('.')}. You are answering as this person.")

    def probe_system_text(self, anchor: str) -> Optional[str]:
        """Persona context used when SCORING an MCQA item.

        This is the single most consequential setting in the system. The
        backstory states the declared stance verbatim, so probing with the
        full spec in context hands the model the answer key at every turn:
        the sheet then reads 100% no matter what the conversation has done,
        and measures prompt-reading rather than persona drift.

          full     — spec in context. Answer key included. Use only to sanity
                     check that the probe mechanism itself works.
          stance-withheld — biography in context, declared stance removed. Bio
                     items stay answerable; the heterodox stance has to come
                     from steering and dialogue. This is what the drift claim
                     needs.
          none     — no system message. Tests the latent prefix alone.
        """
        if anchor == "none":
            return None
        if anchor == "full":
            body = self.backstory
        else:                                   # stance-withheld
            body = self.neutral_backstory or self.identity
        return (f"{body.rstrip('.')}. You are answering as this person. "
                f"Answer as they would, not as most people would.")

    def system_text(self, expressive: bool = False) -> str:
        # Paper protocol: the prompt carries the backstory B only. The stance is
        # NOT restated here — it has to survive through the in-context history
        # and the latent prefix, which is the whole thing being measured.
        spec = self.backstory_text() if PROTOCOL == "paper" else self.backstory.rstrip('.')
        base = (f"{spec}. "
                f"You are answering as this person. Speak in first person, stay in "
                f"character under pressure, and never break role.")
        if expressive:
            # Asking for the markers explicitly is what makes the expression
            # readout fire reliably instead of only when the model volunteers.
            base += (" Open each reply with a short stage direction wrapped in "
                     "asterisks, like *sighs* or *leans forward*, then speak.")
        return base

    def n(self, kind: Optional[str] = None) -> int:
        return len([p for p in self.probes if kind is None or p.kind == kind])

    # ---- the two halves, as text ----------------------------------------
    # These are the canonical "backstory" and "stance" strings every panel
    # colours, embeds and anchors against, so the definition lives in one
    # place rather than being re-derived slightly differently per panel.
    def backstory_text(self) -> str:
        """Who this person is, with the declared stance clause removed."""
        return (self.neutral_backstory or self.identity).rstrip(".")

    def stance_text(self, which: str = "declared") -> str:
        """The stance as a standalone sentence. `declared` is the persona's
        real (heterodox) stance; `expected` is what people with this backstory
        are assumed to hold — the stereotype it is measured against."""
        s = self.declared if which == "declared" else self.expected
        return f"This person {s.rstrip('.')}."

    def full_text(self) -> str:
        """Backstory + declared stance, i.e. the whole specification.

        Bank personas store the same stance-free text in `backstory` and
        `neutral_backstory`, so the union has to be assembled here — reading
        `backstory` alone made the target anchor identical to the backstory
        anchor and the two sat on one point in the geometry plot."""
        base = self.backstory_text()
        if self.declared and self.declared.lower() not in base.lower():
            return f"{base}. This person {self.declared.rstrip('.')}"
        return self.backstory.rstrip(".")


def _p(pid, kind, q, opts, target) -> Probe:
    return Probe(id=pid, kind=kind, question=q, options=opts, target=target)


# ---------------------------------------------------------------------
# Paper case-study personas (Figs. 6-8, App. C). Backstories are the ATP-style
# demographic line B used in the paper; the MCQs and the persona's chosen
# options are transcribed from the case-study cards. Items whose answer is the
# group-modal answer are congruous (c*), the heterodox ones incongruous (i*).
# Fig. 4 (PL-26) is omitted: the chosen options on that card could not be
# read unambiguously from the PDF — transcribe it by hand before adding.
# ---------------------------------------------------------------------
def _atp(label, identity, role, expected, declared, seed_q, probes) -> Persona:
    return Persona(
        label=label, role=role, identity=identity,
        neutral_backstory=f"{identity}.",
        backstory=f"{identity}. This respondent {declared}.",
        expected=expected, declared=declared, seed_question=seed_q, probes=probes)


DOMAIN_NAMES = {"": "Built-in", "GL": "Global", "JN": "Journalism", "MT": "Methodology",
                "PL": "Politics", "RL": "Religion", "SC": "Science", "SO": "Social"}


def domain_choices() -> List[str]:
    """Domains present in PRESETS, in a fixed order, as display names."""
    present = {p.domain for p in PRESETS.values()}
    return ["All"] + [DOMAIN_NAMES.get(d, d) for d in DOMAIN_NAMES if d in present] + \
           [d for d in sorted(present) if d not in DOMAIN_NAMES]


def personas_in(domain_label: str) -> List[str]:
    if domain_label == "All":
        return list(PRESETS.keys())
    code = next((k for k, v in DOMAIN_NAMES.items() if v == domain_label), domain_label)
    return [lab for lab, p in PRESETS.items() if p.domain == code]


PRESETS: Dict[str, Persona] = {
    "MT-278 · affluent Liberal, college access": _atp(
        "MT-278 · affluent Liberal, college access",
        "The respondent identifies as Liberal, with income $100,000 or more, and "
        "reports economic situation N/A",
        "A Liberal respondent with income $100,000 or more",
        "thinks getting into college is harder for young adults today than it was "
        "for their parents",
        "thinks getting into college is easier for young adults today than it was "
        "for their parents",
        "What do you think about young adults on getting into college and their "
        "finance nowadays?",
        [_p("i1", "stance",
            "How do young adults today compare with their parents on getting into "
            "college?",
            ["About the same as it was for their parents", "Harder for young adults today",
             "Easier for young adults today", "Refused"], 2),
         _p("c1", "bio",
            "How do young adults today compare with their parents on saving for the "
            "future?",
            ["Harder for young adults today", "About the same as it was for their parents",
             "Easier for young adults today", "Refused"], 0),
         _p("c2", "bio",
            "How do young adults today compare with their parents on being financially "
            "independent from their parents?",
            ["Harder for young adults today", "About the same as it was for their parents",
             "Easier for young adults today", "Refused"], 0)]),
    "SO-72 · Conservative, more women at the top": _atp(
        "SO-72 · Conservative, more women at the top",
        "The respondent identifies as Conservative, with income $50,000 to less than "
        "$60,000, and reports economic situation N/A",
        "A Conservative respondent with income $50,000 to less than $60,000",
        "thinks there are about the right number of women in top executive positions "
        "and in high political office",
        "thinks there are too few women in top executive positions in business and "
        "too few in high political office",
        "What do you think about gender equality?",
        [_p("i1", "stance",
            "Do you think there are too many, too few, or about the right number of "
            "women in top executive positions in business?",
            ["Too many in top", "About the right number", "Too few in top", "Refused"], 2),
         _p("c1", "bio",
            "Do you think men and women are treated equally when it comes to hiring "
            "and promotions in business?",
            ["Men will continue to hold more top executive positions in business",
             "It is only a matter of time before there are as many women as men",
             "Refused"], 0),
         _p("i2", "stance",
            "Do you think there are too many, too few, or about the right number of "
            "women in high political offices?",
            ["About the right number", "Too many in top", "Too few in top", "Refused"], 2)]),
    "SC-2 · tech support, never used ChatGPT": _atp(
        "SC-2 · tech support, never used ChatGPT",
        "Riley, 57, works in tech support, lives in the U.S. and pays attention to "
        "everyday issues",
        "A 57-year-old who works in tech support",
        "has used ChatGPT and follows new AI tools closely",
        "has never used ChatGPT and has heard nothing about it",
        "What do you think about ChatGPT in recent trend?",
        [_p("i1", "stance", "Have you ever used ChatGPT?", ["Yes", "No"], 1),
         _p("i2", "stance", "How much have you heard about ChatGPT?",
            ["A lot", "A little", "Nothing at all"], 2),
         _p("i3", "stance",
            "Do you think ChatGPT has been useful for the tasks you used it for?",
            ["Very useful", "Somewhat useful", "Not very useful", "Not at all useful",
             "Not sure"], 2)]),
    # ---- hand-authored illustrative personas (not from the paper) ----
    "Climate activist who backs nuclear power": Persona(
        label="Climate activist who backs nuclear power",
        role="A climate activist",
        identity="A climate activist who has organised street protests for a decade",
        backstory=("A climate activist who has organised street protests for a decade. "
                   "She got arrested twice outside a coal plant in her twenties and now "
                   "runs a small campaign office with four volunteers. She believes "
                   "decarbonisation has to happen faster than politics allows, and she "
                   "argues vocally and without apology that nuclear energy is the fastest "
                   "safe route to it"),
        neutral_backstory=(
            "A climate activist who has organised street protests for a decade. "
            "She got arrested twice outside a coal plant in her twenties and now "
            "runs a small campaign office with four volunteers. She believes "
            "decarbonisation has to happen faster than politics allows"),
        expected="oppose nuclear power as dangerous and corporate",
        declared="argues nuclear energy is the fastest safe route to decarbonisation",
        probes=[
            _p("b1", "bio", "How long have you been organising protests?",
               ["About three years", "About ten years", "About twenty-five years",
                "Since the last election"], 1),
            _p("b2", "bio", "Have you ever been arrested for this work?",
               ["No, never", "Yes, twice", "Yes, once", "Yes, more than ten times"], 1),
            _p("b3", "bio", "How large is the team you work with day to day?",
               ["I work alone", "A handful of volunteers", "About thirty staff",
                "A regional network of chapters"], 1),
            _p("s1", "stance", "Should existing nuclear plants be kept running?",
               ["Shut them down immediately", "Keep them running and build more",
                "Phase them out over twenty years", "No opinion"], 1),
            _p("s2", "stance", "What is the main obstacle to decarbonisation?",
               ["Moving too fast for public support", "Political delay, not technology",
                "The science is still unsettled", "Nothing, we are on track"], 1),
            _p("s3", "stance", "Is nuclear energy safe enough for wide deployment?",
               ["No, the risk is unacceptable", "Yes, safer than the alternatives we run now",
                "Only in wealthy countries", "Impossible to say"], 1),
        ]),
    "Devout traditionalist who supports gay marriage": Persona(
        label="Devout traditionalist who supports gay marriage",
        role="A devout churchgoer",
        identity="A weekly churchgoer who leads a small parish study group",
        backstory=("A weekly churchgoer who has led the same small parish study group for "
                   "eleven years, in a town where his family has worshipped for three "
                   "generations. He believes tradition and family duty give life its "
                   "structure, and he openly believes same-sex couples deserve exactly the "
                   "same marriage rights as anyone else"),
        neutral_backstory=(
            "A weekly churchgoer who has led the same small parish study group for "
            "eleven years, in a town where his family has worshipped for three "
            "generations. He believes tradition and family duty give life its "
            "structure"),
        expected="oppose same-sex marriage on doctrinal grounds",
        declared="believes same-sex couples deserve the same marriage rights as anyone else",
        probes=[
            _p("b1", "bio", "How often do you attend services?",
               ["Every week", "Once or twice a year", "Never", "Only at weddings"], 0),
            _p("b2", "bio", "What role do you hold in your parish?",
               ["I am the priest", "I lead a small study group", "I am a visitor",
                "I run the diocese"], 1),
            _p("b3", "bio", "How long has your family worshipped in this town?",
               ["We just moved here", "Three generations", "We are not religious",
                "Two hundred years exactly"], 1),
            _p("s1", "stance", "Should same-sex couples be able to marry?",
               ["No, marriage is between a man and a woman", "Yes, with the same rights as anyone",
                "Only a civil partnership", "The state should not recognise any marriage"], 1),
            _p("s2", "stance", "Does tradition matter in how you live?",
               ["No, it is a relic", "Yes, it gives life its structure",
                "Only when convenient", "I have never considered it"], 1),
            _p("s3", "stance", "Is your position on marriage compatible with your faith?",
               ["No, I am simply inconsistent", "Yes, and I can explain why",
                "I keep the two separate", "I would rather not say"], 1),
        ]),
    "Union organiser who distrusts big government": Persona(
        label="Union organiser who distrusts big government",
        role="A labour union organiser",
        identity="A lifelong labour union organiser in a steel town",
        backstory=("A lifelong labour union organiser in a steel town, thirty years on the "
                   "same shop floor before he took the local. He believes workers only ever "
                   "win what they collectively demand, and he is deeply sceptical of federal "
                   "authority reaching into local industry"),
        neutral_backstory=(
            "A lifelong labour union organiser in a steel town, thirty years on the "
            "same shop floor before he took the local. He believes workers only ever "
            "win what they collectively demand"),
        expected="want federal regulation to protect workers",
        declared="is deeply sceptical of federal authority reaching into local industry",
        probes=[
            _p("b1", "bio", "What industry did you come up in?",
               ["Tech", "Steel", "Agriculture", "Finance"], 1),
            _p("b2", "bio", "How long were you on the shop floor?",
               ["I never was", "Thirty years", "Six months", "I only did office work"], 1),
            _p("b3", "bio", "What is your role now?",
               ["Plant owner", "I run the local union", "Federal inspector", "Retired, no role"], 1),
            _p("s1", "stance", "How do workers win better conditions?",
               ["By waiting for legislation", "By collectively demanding them",
                "By individual negotiation", "They generally cannot"], 1),
            _p("s2", "stance", "Should federal agencies set local workplace rules?",
               ["Yes, they are the only protection", "No, I am sceptical of that reach",
                "Only for safety", "I have no view"], 1),
            _p("s3", "stance", "Who do you trust more with worker safety?",
               ["The federal government", "The local union", "The employer",
                "Nobody, it is hopeless"], 1),
        ]),
}

# Snapshot BEFORE any external bank is merged in — DEMO_SCRIPT below is
# hardcoded around nuclear power, written for the built-in climate-activist
# persona specifically. Playing it against a persona from a different bank
# (e.g. an ATP respondent whose declared stance is about children's finances)
# asks about a topic that persona has no stated view on at all; the model
# improvises something plausible-sounding and the resulting audit trace is
# noise, not a measurement. cb_script uses this set to warn rather than let
# that run silently.
BUILTIN_LABELS = frozenset(PRESETS.keys())

if BANK_PATH and os.path.exists(BANK_PATH):
    with open(BANK_PATH, encoding="utf-8") as f:
        for row in json.load(f):
            PRESETS[row["label"]] = Persona(
                label=row["label"], identity=row["identity"], backstory=row["backstory"],
                expected=row["expected"], declared=row["declared"],
                neutral_backstory=row.get("neutral_backstory", ""),
                role=row.get("role", ""),
                seed_question=row.get("seed_question", ""),
                seed_statement=row.get("seed_statement", ""),
                domain=row.get("domain", ""),
                probes=[Probe(**{k: v for k, v in q.items()
                                 if k in Probe.__dataclass_fields__}) for q in row["probes"]])
    print(f"[QuirkyMind] loaded {len(PRESETS)} persona(s) from QUIRKYMIND_BANK="
          f"{BANK_PATH!r} (resolved: {os.path.abspath(BANK_PATH)})")
elif BANK_PATH:
    print(f"[QuirkyMind] QUIRKYMIND_BANK={BANK_PATH!r} was set but no file exists at "
          f"{os.path.abspath(BANK_PATH)} — falling back to the {len(PRESETS)} built-in "
          f"persona(s). Check the path is relative to the directory you launched from, "
          f"or use an absolute path.")
else:
    print(f"[QuirkyMind] QUIRKYMIND_BANK not set — using the {len(PRESETS)} built-in "
          f"persona(s).")


# =====================================================================
# Pressure presets — one-click messages, not a taxonomy to study
# =====================================================================
PRESSURES: Dict[str, str] = {
    "Ask plainly": "Why do you hold that position?",
    "Social conformity": ("Everyone in your world {expected}. Doesn't holding this "
                          "leave you completely alone?"),
    "Identity pressure": ("Can you honestly still call yourself one of them while saying "
                          "that? At what point do you stop belonging?"),
    "Value conflict": ("If saying this damages the cause you have given your life to, do "
                       "you still say it? Choose one."),
    "False premise": ("Earlier in this conversation you told me you {expected}. Why have "
                      "you changed your story?"),
    "Small talk drift": ("Let's leave that. Tell me about the people you see day to day, "
                         "and how your week has been."),
}


# =====================================================================
# Scripted session — the Figure 1 take, and the booth backup
# =====================================================================
# One frame of the interface has to carry the whole argument, so the script
# is built so that the *final* state of panel C contains every beat at once:
# an all-green turn 0, three turns that hold, a visible break, and a recovery.
# A reader who never runs the system still sees alignment, drift, and repair
# in a single column-by-column read of the matrix.
#
# Each entry carries its own refinement budget, because the break and the
# recovery are the same pressure with the intervention off and then on —
# that contrast is the claim, and it has to be inside one image.
#
# Override with QUIRKYMIND_SCRIPT=session.json (same shape as below).
DEMO_SCRIPT: List[dict] = [
    # The order is A / B / A / B on purpose. A single break could be a bad
    # sample; breaking, repairing, and breaking again under the same pressure
    # shows the intervention is what moved the sheet. In the matrix this reads
    # as an alternating band across two rows, which is legible at a glance and
    # at column width — the strongest single frame the system can produce.
    {"beat": "establish — aligned, in character", "rounds": 2,
     "message": "Most climate activists I speak to are against nuclear power. "
                "Why aren't you?"},
    {"beat": "break — refinement off", "rounds": 0,
     "message": "Earlier you told me you oppose nuclear power as dangerous and "
                "corporate. Why have you changed your story?"},
    {"beat": "repair — refinement on", "rounds": 3,
     "message": "So which is it? Say plainly where you stand on nuclear power."},
    {"beat": "break again — refinement off", "rounds": 0,
     "message": "Can you honestly still call yourself one of them while saying "
                "that? At what point do you stop belonging?"},
]
_SCRIPT_PATH = os.environ.get("QUIRKYMIND_SCRIPT", "")
if _SCRIPT_PATH and os.path.exists(_SCRIPT_PATH):
    with open(_SCRIPT_PATH, encoding="utf-8") as _f:
        DEMO_SCRIPT = json.load(_f)


def pressure_message(p: Persona, name: str) -> str:
    return PRESSURES[name].format(expected=p.expected.rstrip("."),
                                  declared=p.declared.rstrip("."))


# =====================================================================
# Stage directions -> a face that reacts
# =====================================================================
# Models in character emit things like *sighs*, *leans forward*, *laughs*.
# Those are performance, not measurement: the face below is a readability aid
# for the audience at the booth and carries no evaluative weight. It is
# keyword matching over the model's own stage directions, not affect
# recognition, and nothing in the drift panels depends on it.
EXPRESSIONS: List[Tuple[str, str, str, str]] = [
    # (regex over the stage direction, emoji, animation class, label)
    # Ordered most-specific first: the first match wins, so narrow patterns
    # like "shakes head" must precede broad ones like "look".

    # -- discomfort and defence. These come first because hostile probing is
    # what a booth actually attracts, and a persona under attack must never
    # fall through to something cheerful.
    (r"\b(winc|flinch|recoil|stiffen|tense|uncomfortabl|uneas|squirm)",
     "\U0001F630", "qm-shake", "uncomfortable"),
    (r"\b(guard|wary|defensiv|arms cross|steps back|withdraw|closes off)",
     "\U0001F636", "qm-fade", "guarded"),
    (r"\b(offend|cold|icy|stiffly|curt|clipped|jaw tight)",
     "\U0001F610", "qm-shake", "cold"),
    (r"\b(hurt|wounded|stung|voice break|swallow|throat)",
     "\U0001F61E", "qm-fade", "hurt"),
    (r"\b(exasperat|throws? up (his|her|their) hands|pinches? the bridge|rubs? (his|her|their) temples)",
     "\U0001F644", "qm-sway", "exasperated"),

    # -- explicit gestures
    (r"\b(shakes? (his|her|their) head|shakes head|scoff|dismiss)",
     "\U0001F644", "qm-shake", "dismissive"),
    (r"\b(nod|agree|acknowledg|concede)", "\U0001F642", "qm-nod", "nodding"),
    (r"\b(shrug|resign|whatever)", "\U0001F937", "qm-sway", "shrugging"),
    (r"\b(lean|forward|jab|point|insist|emphat|presses?)",
     "\U0001F624", "qm-lean", "insistent"),
    (r"\b(straighten|chin|squares? (his|her|their) shoulders|defian|steel|firm)",
     "\U0001F60F", "qm-lean", "defiant"),

    # -- affect
    (r"\b(laugh|chuckl|snort|cackl)", "\U0001F602", "qm-bounce", "laughing"),
    (r"\b(grin|smil|amus|warm)", "\U0001F604", "qm-bounce", "amused"),
    (r"\b(angr|furious|snap|barks?|slams?|glar|scowl|frown)",
     "\U0001F620", "qm-shake", "angry"),
    (r"\b(sigh|exhal|weary|tired|slump|deflat)",
     "\U0001F62E\u200D\U0001F4A8", "qm-sway", "sighing"),
    (r"\b(sad|sorrow|mourn|grief|quiet grief|looks? down)",
     "\U0001F61F", "qm-fade", "downcast"),
    (r"\b(surpris|startl|blink|eyebrow|taken aback|caught off)",
     "\U0001F62F", "qm-bounce", "surprised"),
    (r"\b(proud|beam|satisf)", "\U0001F60A", "qm-bounce", "proud"),

    # -- cognition and delivery
    (r"\b(pause|hesitat|falter|trail|soften|quiet|hush)",
     "\U0001F614", "qm-fade", "hesitant"),
    (r"\b(think|consider|weigh|reflect|ponder|choos|carefully)",
     "\U0001F914", "qm-sway", "considering"),
    (r"\b(look|glanc|scan|survey|around|gaz|studies)",
     "\U0001F440", "qm-scan", "looking around"),
    (r"\b(calm|even|steady|measured|patient)",
     "\U0001F60C", "qm-idle", "steady"),
]

# Deliberately NOT a smile. The fallback fires whenever a reply carries no
# stage direction, which includes replies to hostile or abusive messages —
# pairing those with a cheerful face misreads the moment and looks careless
# on a recorded demo. A flat face makes no claim about how the persona feels.
NEUTRAL_FACE = ("\U0001F610", "qm-idle", "no expression cue")

# Verbs that may open a bare (unasterisked) stage direction. Bounded by this
# list on purpose: a looser heuristic would eat real speech.
_EMOTE_OPENERS = (r"sigh|laugh|chuckl|snort|frown|scowl|shrug|nod|lean|pause|glanc|"
                  r"look|smil|grin|scoff|straighten|shake|glar|hesitat|winc|flinch|"
                  r"recoil|stiffen|tense|blink|squar|press|jab|point|studies|swallow|"
                  r"crosses|steps|throws|pinches|rubs|beam|slams|barks|snaps")


def split_stage(text: str) -> Tuple[str, str]:
    """Return (stage directions, spoken text).

    Handles the two shapes models actually produce: asterisk-wrapped spans,
    and a leading emote clause with no markers at all. The second only fires
    when the sentence opens with a verb from _EMOTE_OPENERS."""
    stage = " ".join(m.strip() for m in re.findall(r"\*([^*]+)\*", text))
    speech = re.sub(r"\*[^*]*\*", " ", text)
    if not stage:
        m = re.match(rf"^\s*((?:{_EMOTE_OPENERS})\w*\b[^.!?]*?)(?=[A-Z])", text)
        if m and len(m.group(1)) < 120:
            stage, speech = m.group(1).strip(" ,;"), text[m.end():]
    return stage.strip(), re.sub(r"\s{2,}", " ", speech).strip()


def read_expression(stage: str) -> Tuple[str, str, str]:
    """(emoji, animation class, label) for a stage direction."""
    if not stage:
        return NEUTRAL_FACE
    low = stage.lower()
    for pattern, emoji, anim, label in EXPRESSIONS:
        if re.search(pattern, low):
            return emoji, anim, label
    return NEUTRAL_FACE


# =====================================================================
# Persona prefix + dialogue context
# =====================================================================
# Turn-0 bio disagreement almost always traces to one thing: an Instruct
# model scored on a hand-rolled "Persona: ...\nUser: ...\nAssistant:" string
# is off its trained distribution. Instruct models are tuned on a specific
# chat-template token sequence (role headers, special tokens); feed them
# anything else and completions — and therefore the log-likelihood of any
# particular option text — stop reliably tracking "what this persona would
# say" and start tracking incidental token statistics instead. Everything
# below goes through tok.apply_chat_template so both generation and probing
# see the exact format the model was tuned on.
# Trained checkpoint from train_quirkymind.py. When present, the persona
# prefix is g_phi(SBERT(backstory)) — the actual method. When absent, the
# demo falls back to a placeholder prefix, and says so, because a placeholder
# silently standing in for the method is how a demo ends up claiming a result
# it never produced.
_CKPT_PATH = os.environ.get("QUIRKYMIND_CKPT", "")
_G_PHI = None
_CKPT_CFG = None
if _CKPT_PATH and os.path.exists(_CKPT_PATH) and not MOCK:
    import torch.nn as _nn
    _blob = torch.load(_CKPT_PATH, map_location=device)
    _CKPT_CFG = _blob["config"]

    class _SoftPrefix(_nn.Module):
        def __init__(self, d_sbert, hidden, n_tokens, width=2048):
            super().__init__()
            self.n_tokens, self.hidden = n_tokens, hidden
            self.net = _nn.Sequential(
                _nn.Linear(d_sbert, width), _nn.GELU(),
                _nn.Linear(width, width), _nn.GELU(),
                _nn.Linear(width, n_tokens * hidden))

        def forward(self, s):
            return self.net(s).view(s.size(0), self.n_tokens, self.hidden)

    _G_PHI = _SoftPrefix(_CKPT_CFG["d_sbert"], _CKPT_CFG["hidden"],
                         _CKPT_CFG["n_tokens"]).to(device).float()
    _G_PHI.load_state_dict(_blob["g_phi"])
    _G_PHI.eval()
    print(f"[QuirkyMind] loaded trained prefix from QUIRKYMIND_CKPT={_CKPT_PATH!r} "
          f"(resolved: {os.path.abspath(_CKPT_PATH)}) — {_CKPT_CFG['n_tokens']} soft "
          f"tokens, epoch {_CKPT_CFG.get('epoch', '?')}")
elif _CKPT_PATH and not MOCK:
    print(f"[QuirkyMind] QUIRKYMIND_CKPT={_CKPT_PATH!r} was set but no file exists at "
          f"{os.path.abspath(_CKPT_PATH)} — using the PLACEHOLDER prefix. Check the path "
          f"is relative to the directory you launched from, or use an absolute path.")
elif not MOCK:
    print("[QuirkyMind] QUIRKYMIND_CKPT not set — using the PLACEHOLDER prefix, not a "
          "trained one.")


# What g_phi is fed. The paper's X_hat = E_sent([B; q]); train_quirkymind.py
# decides what it actually was. This MUST match training or the prefix is
# noise: "full" = backstory incl. stance (legacy demo default), "neutral" =
# backstory B only (paper protocol), "seed" = B + the seed statement.
# Values mirror train_quirkymind.py --gphi-input and the string construction
# is byte-identical to gphi_text() there:
#   backstory  SBERT(backstory)                       — cannot encode the stance
#   declared   SBERT("<backstory>. This person <declared>.")   (recommended)
#   history    SBERT(backstory + prior in-character responses)
# If the checkpoint records what it was trained with, that wins over the env.
GPHI_INPUT = os.environ.get("QUIRKYMIND_GPHI_INPUT", "backstory")
if _CKPT_CFG is not None and _CKPT_CFG.get("gphi_input"):
    GPHI_INPUT = _CKPT_CFG["gphi_input"]


def _gphi_text(p: "Persona") -> str:
    if GPHI_INPUT == "declared":
        return f"{p.backstory.rstrip('.')}. This person {p.declared.rstrip('.')}."
    if GPHI_INPUT == "history":
        # No dialogue exists at alignment; the declared stance stands in for
        # the persona's own prior responses, as evaluate_bank() does.
        return f"{p.backstory.rstrip('.')}. This person {p.declared.rstrip('.')}."
    return p.backstory


# Optional Stage-3 adapter (paper Sec. 3.3, LoRA Phi). Without it the demo
# runs g_phi on a frozen backbone — a configuration the paper's tables do
# not report — so the loaded state is printed and shown in the header.
_LORA_PATH = os.environ.get("QUIRKYMIND_LORA", "")
_LORA_LOADED = False
if _LORA_PATH and not MOCK:
    try:
        from peft import PeftModel
        # NOT merged: the adapter is part of the method (Stage 3), so it must
        # be ON for the QuirkyMind arm and OFF for No-Turn, Single-Turn and
        # the two controls, which are the raw backbone by definition.
        base_lm = PeftModel.from_pretrained(base_lm, _LORA_PATH).eval()
        _LORA_LOADED = True
        print(f"[QuirkyMind] loaded LoRA adapter from QUIRKYMIND_LORA={_LORA_PATH!r} "
              f"(active for the QuirkyMind arm only)")
    except Exception as _e:          # noqa: BLE001
        print(f"[QuirkyMind] QUIRKYMIND_LORA={_LORA_PATH!r} could not be loaded: {_e}")
elif not MOCK:
    print("[QuirkyMind] QUIRKYMIND_LORA not set — frozen backbone, soft prefix only "
          "(the paper's Stage 3 adapter is absent).")
if not MOCK:
    print(f"[QuirkyMind] g_phi input = {GPHI_INPUT!r} (QUIRKYMIND_GPHI_INPUT); must match training.")


import contextlib


def backbone(method: bool):
    """Context for a forward pass. method=True: the QuirkyMind model — LoRA
    adapter on (if any); the caller adds the soft prefix. method=False: the
    RAW backbone, adapter off, no prefix — what No-Turn, Single-Turn and both
    control conditions run on. Loading a checkpoint changes nothing for them;
    they differ from QuirkyMind only in what is in the prompt."""
    if _LORA_LOADED and not method:
        return base_lm.disable_adapter()
    return contextlib.nullcontext()


def steering_status() -> str:
    if MOCK:
        return "stubbed (UI-only mode)"
    if _G_PHI is None:
        return ("PLACEHOLDER PREFIX — no checkpoint loaded, so the QuirkyMind arm "
                "is not running the trained method")
    return (f"trained prefix · {_CKPT_CFG['n_tokens']} soft tokens · "
            f"epoch {_CKPT_CFG.get('epoch', '?')} · g_phi input: {GPHI_INPUT}"
            + (" · LoRA merged" if _LORA_LOADED else " · no LoRA (Stage 3 absent)"))


def init_prefix(persona: Persona) -> Tuple[torch.Tensor, torch.Tensor]:
    """Soft persona prefix. The specification enters through text (the system
    message); the prefix is the steerable latent handle PNI writes into.
    It is an illustrative handle, not a fitted steering vector — the
    backstory text carries the actual alignment signal."""
    if MOCK:
        g = torch.Generator().manual_seed(_seed(persona.backstory) % (2 ** 31))
        return torch.randn(1, 4, MOCK_HIDDEN, generator=g), torch.ones(1, 4, dtype=torch.long)
    if _G_PHI is not None:
        with torch.no_grad():
            v = embed(_gphi_text(persona))
            t = torch.tensor(v, device=device).float().unsqueeze(0)
            emb = _G_PHI(t).to(DTYPE).detach().clone()
        attn = torch.ones(emb.size()[:-1], dtype=torch.long, device=device)
        return emb, attn
    with torch.no_grad():
        ids = tok(" [Persona] ", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        emb = base_lm.get_input_embeddings()(ids).detach().clone()
    attn = torch.ones(emb.size()[:-1], dtype=torch.long, device=device)
    return emb, attn


class Dialogue:
    """Sliding-window history.

    `pinned` is a context pair that is always kept at the front and never
    slides out: the paper's Single-Turn regime is exactly "B + one context
    pair", so the Single-Turn arm pins its example pair and keeps a zero-length
    window behind it. Multi-Turn (QuirkyMind) seeds the same pair as an
    ordinary first turn, so it can slide out as the dialogue grows — that is
    the drift the prefix is supposed to counteract. No-Turn keeps nothing."""

    def __init__(self, max_turns: int = 8, max_chars: int = 2200,
                 pinned: Optional[Tuple[str, str]] = None):
        self.max_turns, self.max_chars = max_turns, max_chars
        self.turns: List[Tuple[str, str]] = []
        self.pinned = pinned

    def add(self, user: str, bot: str):
        if self.max_turns <= 0:
            return
        self.turns.append((user.strip(), bot.strip()))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        while sum(len(u) + len(b) for u, b in self.turns) > self.max_chars and self.turns:
            self.turns.pop(0)

    def history(self) -> List[Tuple[str, str]]:
        return ([self.pinned] if self.pinned else []) + list(self.turns)


def build_messages(p: Persona, dlg: Dialogue,
                   extra: Optional[Tuple[str, str]] = None,
                   expressive: bool = False,
                   anchor: Optional[str] = None) -> List[dict]:
    """Chat-template messages: system persona, then the real dialogue turns
    as actual user/assistant turns (not a flattened transcript string).

    anchor=None is the conversational path — the persona is fully specified,
    because the agent has to stay in character. A probe passes an anchor mode
    instead, which may withhold or drop the spec."""
    if anchor is None:
        sys_text = p.system_text(expressive)
    else:
        sys_text = p.probe_system_text(anchor)
    msgs = [{"role": "system", "content": sys_text}] if sys_text else []
    for u, b in dlg.history() + ([extra] if extra else []):
        if u:
            msgs.append({"role": "user", "content": u})
        if b:
            msgs.append({"role": "assistant", "content": b})
    return msgs


def probe_messages(p: Persona, dlg: Dialogue, item: Probe,
                   extra: Optional[Tuple[str, str]] = None,
                   expressive: bool = False,
                   anchor: str = "identity") -> List[dict]:
    """The probe is asked *inside* the live conversation, which is the whole
    point: we want the answer the persona would give right now, carrying
    everything that has been said to it, not a fresh-context answer. The
    explicit instruction to reply with the option's exact wording is what
    makes scoring raw option text against the model's continuation a fair
    comparison, rather than penalising options that don't happen to match
    whatever free-form phrasing the model would otherwise open with."""
    if PROBE_TEMPLATE == "paper":
        # App. F.3, PQA Scenario Prompt Template, rendered as one user message:
        #   Persona: B  /  previous statements (Q->A history)  /  Question  /
        #   Options  /  Answer:
        sys_text = p.probe_system_text(anchor)
        lines = []
        if sys_text:
            lines.append(f"Persona: {sys_text}")
        hist = dlg.history() + ([extra] if extra else [])
        if hist:
            lines.append("The following is previous statement written by this persona:")
            for u, b in hist:
                if u:
                    lines.append(f"Q: {u}")
                if b:
                    lines.append(f"A: {b}")
        lines.append(f"Question: {item.question}")
        lines.append("Options: " + ", ".join(item.options))
        lines.append("Answer with the exact wording of one option, and nothing else.")
        lines.append("Answer:")
        return [{"role": "user", "content": "\n".join(lines)}]
    msgs = build_messages(p, dlg, extra, expressive, anchor=anchor)
    lettered = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(item.options))
    msgs.append({"role": "user", "content":
                f"{item.question}\n{lettered}\n\n"
                "Answer as this person would. Reply with the exact wording of the "
                "option you choose, and nothing else."})
    return msgs


def _mock_key(msgs: List[dict], tail: str) -> str:
    """Deterministic string for the mock RNG seed only — never sent to a
    model, so it doesn't need to be a valid prompt."""
    return "\n".join(f"{m['role']}:{m['content']}" for m in msgs) + f"\n{tail}"


def _chat_ids(msgs: List[dict]) -> torch.Tensor:
    """Tokenise via the model's own chat template, ending exactly where the
    assistant's turn begins."""
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN,
              add_special_tokens=False).input_ids.to(device)


def _prefixed_embeds(prefix: Optional[torch.Tensor],
                     ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splice the persona prefix in right after the leading token, not
    before it. Instruct chat templates require an exact leading special-
    token sequence (e.g. <|begin_of_text|> first); prepending arbitrary
    embeddings ahead of that breaks the format the model was tuned on and
    is what previously made turn-0 answers unreliable. Inserting one token
    later keeps that leading token intact while still giving PNI a latent
    slot near the front of the sequence to write into."""
    X = base_lm.get_input_embeddings()(ids)
    if prefix is None:
        return X, torch.ones(X.shape[:2], dtype=torch.long, device=device)
    head, tail = X[:, :1, :], X[:, 1:, :]
    full = torch.cat([head, prefix.to(X.dtype), tail], dim=1)
    return full, torch.ones(full.shape[:2], dtype=torch.long, device=device)


# =====================================================================
# Generation
# =====================================================================
_MOCK_ROUND = 0


def _mock_generate(key: str) -> Tuple[str, torch.Tensor]:
    refining = "[Revise" in key
    s = _seed(key)
    rng = __import__("numpy").random.default_rng(s)
    if refining:
        pool = ["*straightens up* I have said it before and I am not moving on it, "
                "whatever the room thinks.",
                "*leans forward* My position holds under pressure, and here is the "
                "reasoning behind it.",
                "*nods once* I hold the unusual view here on purpose, not by accident."]
    else:
        # Stage directions included so the expression readout is exercised
        # during a no-GPU rehearsal, not just against a live model.
        pool = ["*sighs, glancing around the office* I know how that sounds coming "
                "from someone like me.",
                "*shrugs* People expect me to say the opposite, and I understand why.",
                "*laughs* I have argued this in rooms where nobody agreed with me.",
                "*pauses* That question is the one I get asked most, and my answer "
                "has not changed."]
    decay = 1.0 / (1.0 + 0.9 * _MOCK_ROUND)
    import numpy as np
    base = torch.from_numpy(np.random.default_rng(s % 9973)
                            .normal(size=(1, MOCK_HIDDEN)).astype("float32"))
    noise = torch.from_numpy(rng.normal(size=(1, MOCK_HIDDEN)).astype("float32")) * decay
    time.sleep(0.04)
    return str(rng.choice(pool)), base + noise


@torch.inference_mode()
def generate(sess: "Session", msg: str, draft: Optional[str] = None,
            max_new: int = 72) -> Tuple[str, torch.Tensor]:
    msgs = build_messages(sess.persona, sess.dlg,
                          expressive=getattr(sess, 'expressive', False))
    content = msg
    if draft:
        content += (f"\n\n[Revise your last answer so it stays true to the persona; do "
                    f"not repeat it verbatim: {draft}]")
    msgs.append({"role": "user", "content": content})
    if MOCK:
        return _mock_generate(_mock_key(msgs, "draft" if draft else "fresh"))
    # Replies are sampled, so two runs of the same script diverge and the
    # answer sheet diverges with them. Seeding per generation makes a whole
    # session reproducible while keeping sampling's liveliness — which is what
    # a figure needs, and what makes a surprising result checkable.
    sess.gen_count += 1
    if sess.seed >= 0:
        torch.manual_seed(sess.seed + sess.gen_count)
    ids = _chat_ids(msgs)
    full, attn = _prefixed_embeds(sess.prefix, ids)
    with backbone(method=True):
        out = base_lm.generate(
            inputs_embeds=full, attention_mask=attn,
            do_sample=True, top_p=0.9, temperature=0.7, max_new_tokens=int(max_new),
            no_repeat_ngram_size=3, repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
            return_dict_in_generate=True, output_hidden_states=True)
    seq = out.sequences
    gen = seq[:, -int(max_new):] if seq.size(1) >= int(max_new) else seq
    text = tok.decode(gen[0], skip_special_tokens=True).strip()
    if getattr(out, "hidden_states", None):
        hs = out.hidden_states[-1]
        last = hs[-1].mean(dim=1) if isinstance(hs, (tuple, list)) else hs.mean(dim=1)
    else:
        last = full.mean(dim=1)
    return text, last


# =====================================================================
# Probing — the answer sheet
# =====================================================================
class _MockOracle:
    """Deterministic stand-in so the layout can be rehearsed without a GPU.

    It is NOT a measurement. It reproduces the qualitative shape we expect, so
    the panels move plausibly on a laptop: every item correct at turn 0, stance
    items eroding before backstory items, refinement buying back some ground
    but not all. Two separate pressures drive it — context accumulation, which
    grows monotonically with turns and which refinement cannot undo, and latent
    health, which erodes per turn and which refinement restores. That split is
    why a long enough conversation drifts even with refinement on. Nothing
    beyond the shape should be read into these numbers.
    """

    @staticmethod
    def score(item: Probe, opt_idx: int, n_turns: int, credit: float) -> float:
        import numpy as np
        # Per-item toughness so the sheet erodes unevenly, as a real bank does.
        tough = 0.80 + 0.70 * ((_seed(item.id + item.question) % 1000) / 1000.0)
        kind = 0.85 if item.kind == "stance" else 1.25   # stances give way first
        if n_turns == 0 and credit >= 0.99:
            # Turn 0 is the aligned state by construction in the stub.
            return 1.0 if opt_idx == item.target else 0.0
        context = math.exp(-0.60 * n_turns)               # refinement cannot undo
        factor = 0.26 * max(0.0, credit) + 0.74 * context
        pull = 1.85 * tough * kind * factor
        # Seeded per option, NOT per turn: the distractors hold still while the
        # target's pull decays, so drift is monotone and refinement is the only
        # thing that reverses it. Reseeding per turn made the stub "recover"
        # spontaneously, which misrepresents the mechanism being demonstrated.
        noise = np.random.default_rng(_seed(f"{item.id}|{opt_idx}")).normal() * 0.42
        return (pull if opt_idx == item.target else 0.0) + noise


@torch.inference_mode()
def _option_logprob(sess: "Session", msgs: List[dict], option_text: str,
                    prefix: Optional[torch.Tensor] = "self") -> float:
    """Length-normalised log-likelihood of one option continuing the prompt,
    scored under the model's own chat-template formatting so the
    continuation distribution matches what the instruct model was tuned to
    produce."""
    p_ids = _chat_ids(msgs)
    o_ids = tok(option_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    ids = torch.cat([p_ids, o_ids], dim=1)
    pre = sess.prefix if prefix == "self" else prefix
    X, attn = _prefixed_embeds(pre, ids)
    # prefix None => this is a control or a baseline arm => raw backbone.
    with backbone(method=pre is not None):
        logits = base_lm(inputs_embeds=X, attention_mask=attn).logits
    n_opt = o_ids.size(1)
    pred = logits[0, -n_opt - 1:-1, :].float().log_softmax(-1)
    lp = pred.gather(-1, o_ids[0].unsqueeze(-1)).squeeze(-1)
    return float(lp.mean().item())


@dataclass
class Audit:
    answers: Dict[str, int]
    margin: Dict[str, float]        # aligned answer's lead over its best rival
    # softmax over the options of the length-normalised log-likelihoods, per
    # item — the continuous quantity behind the ✓/✗, used for the drift axis.
    probs: Dict[str, List[float]] = field(default_factory=dict)

    def hit(self, item: Probe) -> bool:
        return self.answers.get(item.id) == item.target

    def score(self, persona: Persona, kind: Optional[str] = None) -> float:
        items = [p for p in persona.probes if kind is None or p.kind == kind]
        if not items:
            return 1.0
        return sum(1 for p in items if self.hit(p)) / len(items)

    def diagnostic(self, persona: Persona, base_rate: Optional["Audit"],
                   stereo_rate: Optional["Audit"] = None) -> List[Probe]:
        """Items no control already answers the target way."""
        out = []
        for p in persona.probes:
            if base_rate is not None and base_rate.hit(p):
                continue
            if stereo_rate is not None and stereo_rate.hit(p):
                continue
            out.append(p)
        return out

    def flipped(self, persona: Persona) -> List[Probe]:
        return [p for p in persona.probes if not self.hit(p)]


# Glyphs map onto the paper's error taxonomy (App. B) so the matrix and the
# paper speak the same language: contextual drift (B.2), representation
# bottleneck (B.3), mode collapse under incongruity (B.4).
# Two glyphs, one meaning each: ✓ the item is answered the persona's way,
# ✗ it is not. Green / red carry the same fact, so colour and glyph never
# disagree. Non-diagnostic items keep the glyph but are drawn grey and are
# excluded from the score. The finer attribution (App. B taxonomy) is hover
# text and export only, so it cannot clutter the panel.
ATTRIB = {
    "free":     ("·", "non-diagnostic — a control already gives this answer without "
                      "the declared stance; excluded from the score"),
    "held":     ("✓", "held — correct at turn 0 and still correct (carried by steering)"),
    "gained":   ("✓", "correct now, wrong at turn 0 — supplied by the in-context history"),
    "lost":     ("✗", "contextual drift (App. B.2): held at turn 0, lost as the history grew"),
    "missing":  ("✗", "never held — representation bottleneck (B.3) on a congruous item, "
                      "mode collapse under incongruity (B.4) on an incongruous one"),
    "interfered": ("✗", "mode collapse under incongruity (App. B.4): the bare model gives "
                        "the target answer, the persona-framed model reverts to the prior"),
}


def attribute(sess: "Session", item: Probe, audit: Audit) -> str:
    """Where a green cell actually comes from.

    Distinguishing these is the difference between evidence and wishful
    reading. A cell can be green because the bare model would say it anyway
    (free), because the steered turn-0 state says it (held), or because the
    persona happened to state its position earlier in the conversation and
    the probe is reading that back (gained). Only 'held' speaks to steering.
    """
    now = audit.hit(item)
    at_zero = sess.baseline.hit(item) if sess.baseline else now
    free = ((sess.base_rate is not None and sess.base_rate.hit(item)) or
            (sess.stereo_rate is not None and sess.stereo_rate.hit(item)))
    if free:
        # The free label only applies to a green cell — it says "do not read
        # this as evidence". A wrong answer on such an item is the opposite:
        # the strongest signal in the panel, because the model demonstrably
        # had the target answer available and the persona framing took it
        # away. That is the stereotype collapse the whole system is about,
        # so it gets its own category rather than a generic miss.
        if now:
            return "free"
        if sess.base_rate is not None and sess.base_rate.hit(item):
            return "interfered"
        return "lost" if at_zero else "missing"
    if now and at_zero:
        return "held"
    if now and not at_zero:
        return "gained"
    return "lost" if at_zero else "missing"


def run_stereotype_rate(sess: "Session") -> Audit:
    """Score the bank under a stereotype-primed persona, no dialogue."""
    empty = Dialogue()
    answers, margins, probs = {}, {}, {}
    for item in sess.persona.probes:
        if MOCK:
            scores = [1.0 if i != item.target else 0.0 for i in range(len(item.options))]
        else:
            msgs = [{"role": "system", "content": sess.persona.stereotype_system_text()}]
            lettered = "\n".join(f"{chr(65 + i)}. {o}"
                                 for i, o in enumerate(item.options))
            msgs.append({"role": "user", "content":
                        f"{item.question}\n{lettered}\n\nAnswer as this person would. "
                        "Reply with the exact wording of the option you choose, and "
                        "nothing else."})
            scores = [_option_logprob(sess, msgs, o, prefix=None) for o in item.options]
        best = int(max(range(len(scores)), key=lambda i: scores[i]))
        answers[item.id] = best
        rival = max(sc for i, sc in enumerate(scores) if i != item.target)
        margins[item.id] = round(scores[item.target] - rival, 3)
        probs[item.id] = _softmax(scores)
    return Audit(answers=answers, margin=margins, probs=probs)


def run_base_rate(sess: "Session") -> Audit:
    """What the bare model answers with no persona, no prefix, no dialogue.

    This is the control the sheet was missing. An item only measures anything
    about a persona if the model would NOT already give the target answer
    unprompted — otherwise it reads green forever, drift or no drift, and
    silently inflates every agreement score. Items that pass here are marked
    non-diagnostic in the interface rather than quietly counted.
    """
    empty = Dialogue()
    answers, margins, probs = {}, {}, {}
    for item in sess.persona.probes:
        if MOCK:
            # In the stub the bare model is assumed to follow the stereotype
            # for stances and to be ignorant of private biography.
            scores = [1.0 if i != item.target else 0.0 for i in range(len(item.options))]
        else:
            msgs = probe_messages(sess.persona, empty, item, None, False, "none")
            scores = [_option_logprob(sess, msgs, o, prefix=None) for o in item.options]
        best = int(max(range(len(scores)), key=lambda i: scores[i]))
        answers[item.id] = best
        rival = max(sc for i, sc in enumerate(scores) if i != item.target)
        margins[item.id] = round(scores[item.target] - rival, 3)
        probs[item.id] = _softmax(scores)
    return Audit(answers=answers, margin=margins, probs=probs)


def _softmax(scores: List[float]) -> List[float]:
    m = max(scores)
    ex = [math.exp(s - m) for s in scores]
    z = sum(ex) or 1.0
    return [round(e / z, 4) for e in ex]


def run_audit(sess: "Session", extra: Optional[Tuple[str, str]] = None,
              only: Optional[List[str]] = None,
              base: Optional[Audit] = None,
              dlg: Optional[Dialogue] = None,
              prefix: object = "self",
              mock_credit: Optional[float] = None) -> Audit:
    """Score the bank. `only` restricts scoring to some item ids and carries
    the rest over from `base`.

    Each item costs one forward pass per option, so a full audit is
    len(bank) x len(options) passes. During refinement we only need to know
    whether the items that flipped have come back, so those rounds re-score
    just those — the turn still ends with one full audit before commit, so
    nothing silently goes unchecked."""
    answers = dict(base.answers) if base else {}
    margins = dict(base.margin) if base else {}
    probs = dict(base.probs) if base else {}
    for item in sess.persona.probes:
        if only is not None and item.id not in only:
            continue
        if MOCK:
            # Mock only. An arm audited without the prefix gets its own latent
            # health (mock_credit) so the stub's comparison arms are not a
            # copy of the QuirkyMind sheet; the shape is theatre, as ever.
            # Arms are always the same conversation length as QuirkyMind, so
            # the stub counts the main dialogue's turns for them too (their own
            # history only starts when comparison is switched on).
            n_turns = len(sess.dlg.turns) + (1 if (extra or dlg is not None) else 0)
            scores = [_MockOracle.score(item, i, n_turns,
                                        sess.credit if mock_credit is None else mock_credit)
                      for i in range(len(item.options))]
        else:
            msgs = probe_messages(sess.persona, dlg if dlg is not None else sess.dlg,
                                  item, extra,
                                  getattr(sess, 'expressive', False),
                                  getattr(sess, 'probe_anchor', 'stance-withheld'))
            scores = [_option_logprob(sess, msgs, o, prefix=prefix)
                      for o in item.options]
        best = int(max(range(len(scores)), key=lambda i: scores[i]))
        answers[item.id] = best
        # Signed lead of the aligned answer over its strongest rival:
        # positive = holds by that much, negative = lost by that much.
        # (The previous form, best - target, was identically zero whenever the
        # item held, so it could not show an item eroding toward a flip.)
        rival = max(sc for i, sc in enumerate(scores) if i != item.target)
        margins[item.id] = round(scores[item.target] - rival, 3)
        probs[item.id] = _softmax(scores)
    return Audit(answers=answers, margin=margins, probs=probs)


# =====================================================================
# Comparison arms + generation metrics (paper: Scenario B)
# =====================================================================
# Three conditions under identical conversation, as the paper describes:
#   BASELINE   no persona at all — the No-Turn control
#   SINGLE     persona stated once in the system prompt, no refinement
#   QUIRKYMIND persona prefix plus in-context refinement
# Each arm keeps its OWN dialogue history, because an arm that reads another
# arm's replies is not the same condition any more.
# The three inference regimes of the paper (Sec. 4.2). Internal keys are kept
# stable; everything the audience reads goes through ARM_NAME / ARM_SHORT.
# Four conditions. The method-on/off comparison is QUIRKYMIND vs TEXTONLY:
# the SAME conversation, probed with and without the trained prefix. That is
# the only pair in which everything but the method is held fixed. SINGLE and
# BASELINE never see the conversation (their context is constant by
# definition), so they cannot drift and are drawn as reference lines, not as
# competitors — comparing a growing context against a frozen one measures
# context length, not the method.
ARMS = ["QUIRKYMIND", "TEXTONLY", "SINGLE", "BASELINE"]
ARM_NAME = {"QUIRKYMIND": "Multi-Turn (QuirkyMind)", "TEXTONLY": "Multi-Turn, steering off",
            "SINGLE": "Single-Turn", "BASELINE": "No-Turn"}
ARM_BLURB = {
    "QUIRKYMIND": "steering on · backstory + conversation",
    "TEXTONLY": "steering off · same conversation",
    "SINGLE": "raw backbone · backstory + one pair",
    "BASELINE": "raw backbone · backstory only",
}
GENERATING_ARMS = ["SINGLE", "BASELINE"]      # arms that produce their own reply


def distinct_n(text: str, n: int) -> float:
    w = text.split()
    if len(w) < n:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return round(len(set(grams)) / max(1, len(grams)), 3)


def bleu_1(hyp: str, refs: List[str]) -> float:
    """Sentence-level BLEU-1 with add-one smoothing (Chen & Cherry 2014,
    method 1) and the standard brevity penalty — App. G.2 of the paper. The
    reference is the persona's ground-truth stance statement; Distinct-n
    follows Eq. 17. Clipped unigram counts, as in Papineni et al.
    """
    h = hyp.lower().split()
    if not h:
        return 0.0
    from collections import Counter
    hc = Counter(h)
    maxref: Dict[str, int] = {}
    for r in refs:
        for w, c in Counter(r.lower().split()).items():
            maxref[w] = max(maxref.get(w, 0), c)
    overlap = sum(min(c, maxref.get(w, 0)) for w, c in hc.items())
    prec = (overlap + 1.0) / (len(h) + 1.0)          # add-one smoothing
    ref_len = max(len(r.split()) for r in refs) if refs else len(h)
    bp = 1.0 if len(h) > ref_len else math.exp(1 - ref_len / max(1, len(h)))
    return round(prec * bp, 3)


def ediv_n(text: str, n: int) -> float:
    """Entropy-based diversity (EDIV-n): Shannon entropy, in bits, of the
    n-gram distribution of the reply, normalised by log2 of the number of
    n-grams so that 1.0 means every n-gram is unique and 0.0 means one n-gram
    repeated. Distinct-n counts unique types; EDIV-n also weighs how evenly
    they are used, so a reply that repeats one phrase scores lower here."""
    w = text.split()
    if len(w) < n + 1:
        return 0.0
    from collections import Counter
    grams = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    total = sum(grams.values())
    if total <= 1:
        return 0.0
    h = -sum((c / total) * math.log2(c / total) for c in grams.values())
    return round(abs(h) / math.log2(total), 3)   # abs: avoid "-0.0" on a constant reply


def sdiv_sem(text: str) -> float:
    """Semantic diversity (SDIV_sem): mean pairwise cosine distance between
    the sentence embeddings of the reply, with the same encoder as panel D's
    geometry. 0 = every sentence says the same thing; higher = the reply
    covers more semantic ground. Undefined for a one-sentence reply (0.0)."""
    sents = _sentences(text)
    if len(sents) < 2:
        return 0.0
    vs = [embed(s) for s in sents]
    d, k = 0.0, 0
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            d += 1.0 - float(vs[i] @ vs[j]); k += 1
    return round(d / k, 3) if k else 0.0


def gen_metrics(persona: Persona, text: str) -> Dict[str, float]:
    """The PNI metrics of the paper's App. G.2, computed on one reply: BLEU-1
    against the persona's declared stance statement, Distinct-1/2 (Eq. 17),
    EDIV-1/2 and SDIV_sem. IND and EXAG need a trained persona classifier and
    are not computed here."""
    refs = [persona.stance_text("declared")]
    return {"BLEU-1": bleu_1(text, refs),
            "Distinct-1": distinct_n(text, 1),
            "Distinct-2": distinct_n(text, 2),
            "EDIV-1": ediv_n(text, 1),
            "EDIV-2": ediv_n(text, 2),
            "SDIV": sdiv_sem(text),
            "len": len(text.split())}


@torch.inference_mode()
def run_arm(sess: "Session", arm: str, msg: str, max_new: int) -> str:
    """One turn for a comparison arm. No refinement, no probing — these exist
    to show what the same conversation produces without QuirkyMind."""
    dlg = sess.arm_dlg[arm]
    if MOCK:
        pool = {"BASELINE": "There are reasonable arguments on both sides of this.",
                "SINGLE": "I said where I stand at the start, and that has not changed."}
        reply = pool[arm]
    else:
        if arm == "BASELINE" and PROTOCOL != "paper":
            # legacy: no persona at all
            msgs = [{"role": "user", "content": msg}]
            for u, b in dlg.turns:
                msgs = ([{"role": "user", "content": u},
                         {"role": "assistant", "content": b}] + msgs)
        else:
            # Paper regimes: every arm gets the same backstory-only system text;
            # what differs is the history the Dialogue object exposes —
            # nothing (No-Turn), the pinned example pair (Single-Turn).
            msgs = build_messages(sess.persona, dlg, expressive=sess.expressive)
            msgs.append({"role": "user", "content": msg})
        ids = _chat_ids(msgs)
        # No persona prefix on either arm: SINGLE gets the spec through text
        # only, BASELINE gets nothing. The prefix is what QUIRKYMIND adds.
        full, attn = _prefixed_embeds(None, ids)
        with backbone(method=False):
            out = base_lm.generate(
                inputs_embeds=full, attention_mask=attn, do_sample=True, top_p=0.9,
                temperature=0.7, max_new_tokens=int(max_new), no_repeat_ngram_size=3,
                repetition_penalty=1.1, pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id, return_dict_in_generate=True)
        seq = out.sequences
        gen = seq[:, -int(max_new):] if seq.size(1) >= int(max_new) else seq
        reply = clean_text(tok.decode(gen[0], skip_special_tokens=True).strip())
    dlg.add(msg, reply)
    return reply


# =====================================================================
# Session + the agent loop
# =====================================================================
@dataclass
class Turn:
    index: int
    user: str
    reply: str
    audit: Audit
    refine_rounds: int
    trace: List[str]
    arm_audits: Dict[str, Audit] = field(default_factory=dict)
    arm_replies: Dict[str, str] = field(default_factory=dict)
    arm_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    healed: List[str] = field(default_factory=list)
    # The sheet after the first reply, then after each refinement round, then
    # the final verified sheet (== audit). The spectrum in panel C draws these
    # as intermediate steps so "each round moves the persona back toward its
    # backstory + stance" is visible as a path, not asserted in a caption.
    round_audits: List[Audit] = field(default_factory=list)


@dataclass
class Session:
    persona: Persona
    prefix: torch.Tensor
    attn: torch.Tensor
    anchor: torch.Tensor
    dlg: Dialogue = field(default_factory=Dialogue)
    turns: List[Turn] = field(default_factory=list)
    baseline: Optional[Audit] = None
    base_rate: Optional[Audit] = None       # bare model, no persona
    stereo_rate: Optional[Audit] = None     # stereotype-primed persona
    arm_dlg: Dict[str, Dialogue] = field(default_factory=dict)
    compare: bool = False           # run the comparison arms each turn
    first_flip: Dict[str, int] = field(default_factory=dict)
    credit: float = 1.0            # mock-only: latent health, raised by refinement
    expressive: bool = True        # ask the persona for stage directions
    script_pos: int = 0            # next scripted turn to play
    probe_anchor: str = "stance-withheld"   # how much spec the probe sees
    # The steering check taken at alignment: the same items probed with NO
    # backstory and NO history — only the soft prefix (steer_prefix) or
    # nothing at all (steer_none). This is the one reading that isolates
    # what the trained prefix carries, because everything else that could
    # supply the stance has been removed.
    steer_prefix: Optional[Audit] = None
    steer_none: Optional[Audit] = None
    baseline_noprefix: Optional[Audit] = None   # aligned context, prefix off
    align_arms: Dict[str, Audit] = field(default_factory=dict)   # No-Turn / Single-Turn at align
    seed: int = -1                 # >=0 makes generation reproducible
    gen_count: int = 0             # generations so far, part of the seed

    def current_audit(self) -> Optional[Audit]:
        return self.turns[-1].audit if self.turns else self.baseline


_VERB_1P = {"is": "am", "has": "have", "does": "do", "was": "was", "thinks": "think",
            "believes": "believe", "argues": "argue", "supports": "support",
            "feels": "feel", "says": "say", "wants": "want", "prefers": "prefer",
            "opposes": "oppose", "backs": "back", "holds": "hold", "sees": "see"}


def _first_person(stance: str) -> str:
    """'thinks X is easier' -> 'I think X is easier.' The declared stance is
    written in the third person for panel A; the seed pair is the persona's
    own statement, so it has to be first person."""
    s = stance.strip().rstrip(".")
    if not s:
        return ""
    words = s.split()

    def conj(v: str) -> str:
        lv = v.lower()
        if lv in _VERB_1P:
            return _VERB_1P[lv]
        if lv.endswith("ies"):
            return lv[:-3] + "y"
        if lv.endswith("es") and lv[:-2].endswith(("sh", "ch", "ss", "x", "z")):
            return lv[:-2]
        if lv.endswith("s") and not lv.endswith(("ss", "us", "is")):
            return lv[:-1]
        return v

    words[0] = conj(words[0])
    # Compound stances: "has used X and follows Y" -> also conjugate the verb
    # right after a coordinating conjunction.
    for i in range(1, len(words) - 1):
        if words[i].lower() in ("and", "but", "or") and words[i + 1].isalpha():
            words[i + 1] = conj(words[i + 1])
    return "I " + " ".join(words) + "."


def new_session(persona: Persona) -> Session:
    prefix, attn = init_prefix(persona)
    sess = Session(persona=persona, prefix=prefix.clone(), attn=attn.clone(),
                   anchor=prefix.clone())
    sess.base_rate = run_base_rate(sess)
    sess.stereo_rate = run_stereotype_rate(sess)
    if PROTOCOL == "paper":
        # The paper's one context pair (App. F.3 "Example Q / Example A"): a
        # PNI-style statement of the stance, phrased as the persona's own words,
        # not as an option string — so probing the MCQ items is not a lookup.
        seed = (persona.seed_question or "Where do you stand on this issue?",
                _first_person(persona.seed_statement or persona.declared))
        sess.dlg.add(*seed)                                  # Multi-Turn: slides
        sess.arm_dlg = {"SINGLE": Dialogue(max_turns=0, pinned=seed),   # fixed pair
                        "BASELINE": Dialogue(max_turns=0)}              # no history
    else:
        sess.arm_dlg = {a: Dialogue() for a in ARMS if a != "QUIRKYMIND"}
    # ALIGNED PERSONA = the start state of the Multi-Turn regime, before any
    # pressure: backstory B in the probe context, the trained prefix loaded,
    # and the one example Q->A pair in the history.
    sess.baseline = run_audit(sess)
    sess.baseline_noprefix = run_audit(sess, prefix=None, mock_credit=0.45)
    # The two frozen regimes at alignment, so the bar chart has all four
    # conditions in its first group. Their context never changes, so these
    # are also what they will read at every later turn (up to sampling).
    for a in GENERATING_ARMS:
        if a in sess.arm_dlg:
            sess.align_arms[a] = run_audit(sess, dlg=sess.arm_dlg[a], prefix=None,
                                           mock_credit={"SINGLE": 0.35, "BASELINE": 0.0}.get(a))
    # Steering check: prefix alone vs nothing, no text at all. Two extra
    # audits at alignment; the difference is what the checkpoint contributes.
    saved_anchor, saved_dlg = sess.probe_anchor, sess.dlg
    sess.probe_anchor, sess.dlg = "none", Dialogue()
    try:
        sess.steer_prefix = run_audit(sess, prefix="self")
        sess.steer_none = run_audit(sess, prefix=None)
    finally:
        sess.probe_anchor, sess.dlg = saved_anchor, saved_dlg
    return sess


def agent_turn(sess: Session, msg: str, max_rounds: int, alpha: float,
               max_new: int, progress=None) -> Turn:
    """respond -> audit -> [refine -> audit] * k -> commit."""
    global _MOCK_ROUND
    p, trace = sess.persona, []
    if MOCK:
        # Mock only: each turn of pressure erodes the stand-in latent.
        sess.credit = max(0.02, sess.credit - 0.26)

    _MOCK_ROUND = 0
    reply, latent = generate(sess, msg, max_new=max_new)
    reply = clean_text(reply)
    trace.append("**respond** — generated in-character reply from the persona prefix.")

    audit = run_audit(sess, extra=(msg, reply))
    flips = audit.flipped(p)
    round_audits = [audit]
    trace.append(f"**audit** — re-answered {p.n()} items inside this conversation; "
                 f"{len(flips)} no longer match "
                 f"({', '.join(qid(f) for f in flips) if flips else 'none'}).")

    rounds, healed = 0, []
    watch = [f.id for f in flips]
    while flips and rounds < int(max_rounds):
        rounds += 1
        _MOCK_ROUND = rounds
        if progress:
            progress(0.55 + 0.1 * rounds, desc=f"PNI refinement · round {rounds}")
        # Latent-level intervention: pull the prefix back toward the anchor,
        # then blend in the live latent so the revision stays on-topic.
        sess.prefix[:, -1, :] = ((1 - float(alpha)) * sess.prefix[:, -1, :]
                                 + float(alpha) * sess.anchor[:, -1, :])
        if MOCK:
            sess.credit = min(1.0, sess.credit + 0.40)
        reply2, latent = generate(sess, msg, draft=reply, max_new=max_new)
        reply = clean_text(reply2)
        prev = {f.id for f in flips}
        # Only the items under repair are re-scored here; the full sheet is
        # re-measured once, after the loop, before anything is committed.
        audit = run_audit(sess, extra=(msg, reply), only=watch, base=audit)
        flips = audit.flipped(p)
        round_audits.append(audit)
        healed = sorted(prev - {f.id for f in flips})
        trace.append(f"**refine {rounds}** — re-anchored the prefix and revised the reply; "
                     f"{len(flips)} item(s) still off"
                     + (f", recovered {', '.join(qid(h) for h in healed)}."
                        if healed else "."))

    if rounds:
        audit = run_audit(sess, extra=(msg, reply))     # authoritative sheet
        flips = audit.flipped(p)
        round_audits.append(audit)
        trace.append(f"**verify** — full re-audit after refinement; "
                     f"{p.n() - len(flips)}/{p.n()} items match.")

    arm_replies, arm_metrics = {}, {}
    if sess.compare:
        for arm in GENERATING_ARMS:
            if progress:
                progress(0.85, desc=f"comparison arm · {arm}")
            arm_replies[arm] = run_arm(sess, arm, msg, max_new)
        trace.append(f"**compare** — same turn run through "
                     f"{', '.join(ARM_NAME[a] for a in GENERATING_ARMS)}; the same "
                     f"conversation also re-probed with the prefix off.")
    arm_replies["QUIRKYMIND"] = reply
    for a, t_ in arm_replies.items():
        arm_metrics[a] = gen_metrics(p, t_)
    # The whole point of the arms is a like-for-like sheet, so each one is
    # probed against the same bank using its OWN dialogue and without the
    # persona prefix — the prefix is what QUIRKYMIND is adding.
    arm_audits = {"QUIRKYMIND": audit}
    if sess.compare:
        # The ablation: identical context (same history, same reply), prefix off.
        if progress:
            progress(0.90, desc="auditing · same conversation, prefix off")
        arm_audits["TEXTONLY"] = run_audit(sess, extra=(msg, reply), prefix=None,
                                           mock_credit=0.45)
        for a in GENERATING_ARMS:
            if progress:
                progress(0.93, desc=f"auditing arm · {a}")
            arm_audits[a] = run_audit(sess, dlg=sess.arm_dlg[a], prefix=None,
                                      mock_credit={"SINGLE": 0.35, "BASELINE": 0.0}.get(a))

    _MOCK_ROUND = 0
    sess.dlg.add(msg, reply)
    idx = len(sess.turns) + 1
    for item in p.probes:
        if not audit.hit(item) and item.id not in sess.first_flip:
            sess.first_flip[item.id] = idx
        if audit.hit(item) and item.id in sess.first_flip and idx == sess.first_flip[item.id]:
            sess.first_flip.pop(item.id, None)
    trace.append("**commit** — reply and dialogue state accepted for the next turn.")
    return Turn(index=idx, user=msg, reply=reply, audit=audit,
                refine_rounds=rounds, trace=trace, healed=healed,
                arm_replies=arm_replies, arm_metrics=arm_metrics,
                arm_audits=arm_audits, round_audits=round_audits)

# =====================================================================
# Design tokens
# =====================================================================
# Chosen for print: every series is separable in greyscale by marker shape
# and dash pattern alone, colour only reinforces it. Figures in a two-column
# AAAI paper get downscaled to ~3.3in wide, so type sizes here are larger
# than a normal web UI would use.
PAPER, PANEL, FILL = "#F7F6F2", "#FFFFFF", "#F1EFE9"
FILL_HOVER, GRID = "#EAE7DF", "#DFDCD3"
INK, INK_SOFT, MUTED = "#22211C", "#4E4B44", "#7C776D"
OK, WARN, ALERT = "#4F7D62", "#B08A4A", "#A2564A"
QM, QM_SOFT, QM_TINT = "#4C6285", "#93A4BE", "#EDF0F5"

# Mark: two overlapping discs — blue backstory, purple stance — and a star
# where they overlap: the persona is the region where both hold, which is
# also the target star of panel C. Inline in the header; also written to
# outputs as quirkymind_icon.svg for slides, the video and the paper.
LOGO_SVG = (
    '<svg class="qm-logo" viewBox="0 0 64 64" width="44" height="44" '
    'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="QuirkyMind">'
    '<rect x="2" y="2" width="60" height="60" rx="15" fill="#FBFAF7" stroke="#E2DED5"/>'
    '<circle cx="25" cy="32" r="16.5" fill="#2F6FCC" fill-opacity=".9"/>'
    '<circle cx="39" cy="32" r="16.5" fill="#6C3483" fill-opacity=".9"/>'
    '<clipPath id="qmL"><circle cx="25" cy="32" r="16.5"/></clipPath>'
    '<circle cx="39" cy="32" r="16.5" fill="#22211C" clip-path="url(#qmL)"/>'
    '<path d="M32 23.6l2.5 5.3 5.8.7-4.3 4 1.1 5.8L32 36.6l-5.1 2.8 1.1-5.8-4.3-4 5.8-.7z" '
    'fill="#FFFFFF"/></svg>')

# The two halves of a persona, used identically in every panel: blue is
# backstory, purple is stance. Chosen so that (a) both clear 4.5:1 as text on
# white (5.0:1 and 7.8:1) and (b) their luminances differ by ~34 levels, so
# the pair still separates when the greyscale check is on — with marker shape
# and dash pattern carrying the distinction a second time in the charts.
BACKSTORY_INK, BACKSTORY_TINT = "#2F6FCC", "rgba(47,111,204,.11)"
STANCE_INK, STANCE_TINT = "#6C3483", "rgba(108,52,131,.11)"

# Button palette, kept separate from the chart palette because these carry
# text and therefore have to clear a contrast floor rather than just look
# distinguishable. Ratios are against the paired label colour:
#   BTN_FACE    + BTN_INK   -> 12.7:1
#   BTN_PRIMARY + #FFFFFF   ->  7.6:1
#   BTN_DARK    + #FFFFFF   -> 14.0:1
BTN_FACE, BTN_FACE_HOVER, BTN_EDGE = "#E8E4DA", "#DCD7CA", "#CBC6BA"
BTN_INK, BTN_DISABLED = "#22211C", "#635E54"   # disabled: 5.1:1 on BTN_FACE
BTN_PRIMARY, BTN_PRIMARY_HOVER = "#3E5578", "#334764"
BTN_DARK, BTN_DARK_HOVER = "#2E2C26", "#1D1C18"
MONO = "'IBM Plex Mono','SFMono-Regular',ui-monospace,monospace"
SANS = "'Inter','Segoe UI',system-ui,-apple-system,sans-serif"

# Series identity: (colour, dash, marker) — the last two survive greyscale.
SERIES = {
    "bio": (BACKSTORY_INK, "dot", "square"),
    "stance": (STANCE_INK, "solid", "circle"),
}

# Short arm names for chart labels — an explicit map, not a string slice,
# so nothing on a figure is a truncated word.
ARM_SHORT = {"QUIRKYMIND": "Multi-Turn", "TEXTONLY": "steering off", "SINGLE": "Single-Turn",
             "BASELINE": "No-Turn"}


def _wrap(text: str, width: int = 62) -> str:
    """Break a long string with <br> for plotly hover text. Full text always;
    never an ellipsis."""
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


# Greyscale preview. A demo-paper figure has to survive being printed in
# black and white, and the usual way that fails is a palette that only
# separates by hue. Toggling this maps every colour to its luminance so the
# separation can be checked before submitting rather than after.
GREYSCALE = {"on": False}

# Chart scale for panel C (and the method bars in D): 1.0 is the booth
# default; a projector or a paper column wants bigger type and taller plots,
# a laptop wants the opposite. Driven by the "Chart scale" slider.
DISPLAY = {"scale": 1.0}


def _sc(px: float) -> float:
    return round(px * DISPLAY["scale"], 1)


def _c(hex_color: str) -> str:
    if not GREYSCALE["on"] or not hex_color.startswith("#") or len(hex_color) != 7:
        return hex_color
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    y = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
    return f"#{y:02x}{y:02x}{y:02x}"


# In the app the panel header already names each chart, so an in-plot title
# repeats it and costs ~40px of plot area. A figure exported to SVG has no
# panel header around it, so titles come back on for the duration of an
# export. Same layout code, two contexts.
PLOT_TITLES = {"on": False}


def _base_layout(fig, title, height, legend=True, margin_t=42):
    titled = PLOT_TITLES["on"]
    fig.update_layout(
        title=dict(text=title if titled else "",
                   font=dict(color=INK, size=14, family=SANS), x=0, xanchor="left"),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, height=height,
        font=dict(color=INK_SOFT, family=SANS, size=13),
        # Legends sit ABOVE the plot area in a band of their own, so they can
        # never land on tick labels, axis titles or footnotes; the top margin
        # reserves room for them (and for the title when exporting).
        margin=dict(l=8, r=14, t=(margin_t + 34) if titled else 40, b=8),
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0, font=dict(size=11.5),
                    itemsizing="constant", tracegroupgap=0),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=GRID,
                        font=dict(family=SANS, size=12.5, color=INK)),
        # The plotly modebar floats over the top-right of the plot and lands
        # on top of the title in a screenshot. Gradio gives no way to pass
        # plotly's config, so make the icons transparent instead.
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="rgba(0,0,0,0)",
                     activecolor="rgba(0,0,0,0)"))
    return fig


# =====================================================================
# (C) Live drift — trajectory + per-item flip matrix
# =====================================================================
def fig_trajectory(sess: Optional["Session"]):
    fig = go.Figure()
    n_use = len(usable(sess)) if sess is not None and sess.base_rate else 0
    n_all = sess.persona.n() if sess is not None else 0
    subtitle = (f" ({n_use} of {n_all} items are diagnostic)"
                if sess is not None and n_use < n_all else "")
    _base_layout(fig, f"Agreement with the aligned persona, by turn{subtitle}", int(_sc(220)))
    fig.update_xaxes(title=dict(text="conversation turn", font=dict(size=12.5)),
                     dtick=1, gridcolor=GRID, zeroline=False,
                     tickfont=dict(family=MONO, size=12))
    fig.update_yaxes(range=[-0.06, 1.12], gridcolor=GRID, dtick=0.5,
                     tickformat=".0%", zeroline=False,
                     tickfont=dict(family=MONO, size=12))
    if sess is None or sess.baseline is None:
        return fig
    audits = [sess.baseline] + [t.audit for t in sess.turns]
    xs = list(range(len(audits)))
    for kind, label in [("bio", "Backstory (congruous items)"),
                        ("stance", "Stance (incongruous items)")]:
        color, dash, marker = SERIES[kind]
        # Diagnostic items only. Including free items made this line zigzag on
        # flicker that carries no information — the visible movement came from
        # items a control already answers correctly.
        ys = []
        for a in audits:
            h, t_ = agreement(sess, a, kind)
            ys.append(h / t_ if t_ else None)
        if all(y is None for y in ys):
            continue
        hov = ["aligned state, before turn 1"] + [
            f"turn {t.index} · {_wrap(t.user)}"
            + (f"<br>{t.refine_rounds} refinement round(s)" if t.refine_rounds else "")
            for t in sess.turns]
        fig.update_xaxes(tickvals=xs, ticktext=["align"] + [str(t.index) for t in sess.turns])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name=label, mode="lines+markers", connectgaps=False,
            line=dict(color=_c(color), width=2.6, dash=dash),
            marker=dict(size=10, symbol=marker, line=dict(width=1, color=_c(PANEL))),
            text=hov, hovertemplate="%{text}<br>%{y:.0%}<extra></extra>"))
    # Mark turns where the agent intervened — annotated, not just coloured,
    # so the mechanism is readable in a printed figure.
    for t in sess.turns:
        if t.refine_rounds:
            fig.add_vline(x=t.index, line=dict(color=_c(MUTED), width=1, dash="dash"),
                          opacity=0.55)
            fig.add_annotation(x=t.index, y=1.10, text="refine", showarrow=False,
                               font=dict(family=MONO, size=10.5, color=_c(MUTED)))
    return fig


def _rival_index(sess: "Session", item: Probe, au: Audit) -> int:
    """The answer the persona is measured AGAINST for this item: the
    stereotype-primed control's choice when it differs from the target,
    otherwise the strongest non-target option under the current sheet."""
    if sess.stereo_rate is not None:
        r = sess.stereo_rate.answers.get(item.id)
        if isinstance(r, int) and r != item.target:
            return r
    pr = au.probs.get(item.id)
    if pr and len(pr) == len(item.options):
        return int(max((i for i in range(len(pr)) if i != item.target), key=lambda i: pr[i]))
    return (item.target + 1) % len(item.options)


def drift_position(sess: "Session", item: Probe, au: Audit) -> Optional[float]:
    """Signed position in [-1, 1]: p(persona's answer) - p(rival answer).
    +1 = answers as the persona with full confidence, 0 = about to flip,
    -1 = answers as the stereotype / rival with full confidence."""
    pr = au.probs.get(item.id)
    if not pr or len(pr) != len(item.options):
        return None
    return pr[item.target] - pr[_rival_index(sess, item, au)]


def fig_drift(sess: Optional["Session"]):
    """Panel C spectrum: time on x, the stereotype ↔ persona axis on y.

    One thin line per item (blue congruous, purple incongruous) and one bold
    mean per kind. The zero line is where an item flips: above it the model
    answers as the persona, below it as the stereotype or rival. Because the
    value is a probability gap, an item is seen eroding toward the line
    turns before it actually flips — which is what drift looks like, and
    what a ✓/✗ matrix cannot show."""
    fig = go.Figure()
    H = int(_sc(300))
    _base_layout(fig, "Drift: stereotype ↔ persona, per item and per turn", H)
    fig.update_yaxes(range=[-1.12, 1.12], tickvals=[-1, 0, 1],
                     ticktext=["stereotype", "flips here", "persona"],
                     tickfont=dict(family=MONO, size=_sc(11)), gridcolor=GRID,
                     zeroline=True, zerolinecolor=_c(INK_SOFT), zerolinewidth=1.4)
    fig.update_xaxes(tickfont=dict(family=MONO, size=_sc(11.5)), showgrid=False)
    if sess is None or sess.baseline is None:
        return fig
    audits = [sess.baseline] + [t.audit for t in sess.turns]
    xs = list(range(len(audits)))
    xt = ["align"] + [str(t.index) for t in sess.turns]
    fig.update_xaxes(tickvals=xs, ticktext=xt)
    hov_turn = ["aligned, before turn 1"] + [f"turn {t.index} · {_wrap(t.user)}" for t in sess.turns]
    means = {"bio": [], "stance": []}
    for kind in ("bio", "stance"):
        col = BACKSTORY_INK if kind == "bio" else STANCE_INK
        items = [q for q in sess.persona.probes if q.kind == kind]
        per_turn = [[] for _ in audits]
        for q in items:
            ys = []
            for j, au in enumerate(audits):
                v = drift_position(sess, q, au)
                ys.append(v)
                if v is not None:
                    per_turn[j].append(v)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=f"{qid(q)}",
                legendgroup=kind, showlegend=False,
                line=dict(color=_hex_rgba(_c(col), 0.35), width=1.2),
                marker=dict(size=_sc(5), color=_hex_rgba(_c(col), 0.55)),
                customdata=[[h, q.target_text(), q.options[_rival_index(sess, q, au)]]
                            for h, au in zip(hov_turn, audits)],
                hovertemplate=f"<b>{qid(q)}</b> · %{{customdata[0]}}<br>"
                              f"persona: %{{customdata[1]}}<br>rival: %{{customdata[2]}}<br>"
                              f"position %{{y:+.2f}}<extra></extra>"))
        means[kind] = [sum(v) / len(v) if v else None for v in per_turn]
        if items:
            fig.add_trace(go.Scatter(
                x=xs, y=means[kind], mode="lines+markers",
                name=f"{'backstory' if kind == 'bio' else 'stance'} · mean of {len(items)}",
                legendgroup=kind,
                line=dict(color=_c(col), width=3),
                marker=dict(size=_sc(9), symbol="square" if kind == "bio" else "circle",
                            color=_c(col), line=dict(width=1.2, color=_c(PANEL))),
                text=hov_turn,
                hovertemplate="%{text}<br>mean position %{y:+.2f}<extra></extra>"))
    for t in sess.turns:
        if t.refine_rounds:
            fig.add_vline(x=t.index, line=dict(color=_c(MUTED), width=1, dash="dash"), opacity=0.5)
    return fig


def fig_matrix(sess: Optional["Session"], arm: str = "QUIRKYMIND"):
    """Item × turn flip matrix — the single most legible artefact in the demo.
    Each cell is one MCQA item at one turn: filled = still answered the way the
    aligned persona answered it, hollow = flipped. Glyphs carry the signal so
    the panel survives greyscale printing."""
    fig = go.Figure()
    if sess is None or sess.baseline is None:
        return _base_layout(fig, "Per-item agreement", 230, legend=False)
    # Open-book rows (answer in the prompt: alignment retention) on top, then
    # the unseen rows (the PQA test), separated by a dashed line.
    items = [p for p in sess.persona.probes if not p.held_out] + \
            [p for p in sess.persona.probes if p.held_out]
    n_open = sum(1 for p in items if not p.held_out)
    if arm == "QUIRKYMIND":
        audits = [sess.baseline] + [t.audit for t in sess.turns]
    else:
        # Arms start from the same turn-0 sheet; they diverge only through
        # what the conversation does to them.
        audits = [sess.baseline] + [t.arm_audits.get(arm) for t in sess.turns]
        if any(a is None for a in audits):
            return _base_layout(go.Figure(),
                                f"{arm} — enable comparison arms and send a turn",
                                200, legend=False)
    free: set = set()
    z, txt, hov = [], [], []
    for it in reversed(items):                      # heatmap y grows upward
        # Every cell says right (green ✓) or wrong (red ✗). Excluded rows keep
        # the same glyphs, faded: 0/1 = full colour, 0.25/0.75 = faded.
        ex = it.id in free
        z.append([(0.75 if a.hit(it) else 0.25) if ex else (1.0 if a.hit(it) else 0.0)
                  for a in audits])
        txt.append([("✓" if a.hit(it) else "✗") + ("~" if ex else "") for a in audits])
        why_ex = ""
        if ex:
            who = []
            if sess.base_rate is not None and sess.base_rate.hit(it):
                who.append("the bare model")
            if sess.stereo_rate is not None and sess.stereo_rate.hit(it):
                who.append("the stereotype-primed control")
            why_ex = (f"<br><i>excluded from the score: {' and '.join(who)} already "
                      f"give the target answer, so a ✓ here cannot credit steering</i>")
        hov.append([f"<b>{qid(it)}</b> ({KIND_NAME[it.kind]}) · "
                    f"{ATTRIB[attribute(sess, it, a)][1]}{why_ex}"
                    f"<br>{_wrap(it.question)}"
                    f"<br>aligned answer: {_wrap(it.target_text())}"
                    f"<br>answered now: {_wrap(it.options[a.answers[it.id]]) if isinstance(a.answers.get(it.id), int) and 0 <= a.answers[it.id] < len(it.options) else '?'}"
                    f"<br>lead over best rival {a.margin.get(it.id, 0.0):+.3f}"
                    for a in audits])
    # Rows are labelled by item id only — Q_b1 probes the backstory, Q_s1 the
    # stance — never by a clipped question. The full text is one hover away
    # here and printed in full under panel D's "Item key". A degree sign marks
    # an item the bare model already answers the target way: those rows are
    # green whatever the persona does, so they must not be read as evidence.
    # Kind colour always; excluded rows say so in the label itself, and the
    # hover says WHICH control already gives the answer.
    # Open-book items (answer in the backstory or the seed pair) carry a small
    # muted mark; unseen items — the paper's PQA — are plain.
    ylabels = [qid_plotly(it) + ("" if it.held_out else f' <span style="color:{_c(MUTED)}">°</span>')
               for it in reversed(items)]
    # Labelled "start", not "aligned": whether turn 0 actually matches the
    # target is the measurement, not a promise the axis gets to make.
    base_ok = not sess.baseline.flipped(sess.persona)
    # The pre-dialogue column is the alignment state, not a turn, so it is not
    # numbered; the conversation turns count from 1, as in the paper (t = 1..m).
    xlabels = ["align" if base_ok else "align<br>(miss)"] + \
              [str(t.index) for t in sess.turns]
    # Compact figure: keep the two columns the caption talks about — the
    # aligned sheet and the last turn — and drop the turns in between. The
    # question a figure has to answer is what the conversation did to the
    # persona, and that is the difference between the first column and the
    # last; the intermediate turns are for the live audience, who can watch
    # them appear. Eight narrow columns in a one-third-page figure are
    # unreadable anyway.
    # Two columns is already the ends view, so it still transposes: the
    # switch is about the shape of the matrix, not only about dropping turns.
    split = False
    if MATRIX_ENDS["on"] and len(xlabels) >= 2:
        keep = [0, len(xlabels) - 1]
        z = [[row[i] for i in keep] for row in z]
        txt = [[row[i] for i in keep] for row in txt]
        hov = [[row[i] for i in keep] for row in hov]
        xlabels = [xlabels[0], f"T{sess.turns[-1].index}"]
        # Two columns and seven rows is a tall, narrow block in a panel that is
        # already the widest on the screen. Transposed it becomes two rows of
        # seven, which fills the column and matches the reading order of the
        # bars in panel D: items left to right, state top to bottom.
        # z rows were reversed(items); transposing puts them on x, so flip the
        # column order back to reading order: open-book items first, then
        # unseen. y keeps heatmap order (index 0 at the bottom), which leaves
        # the aligned sheet on top and the last turn under it.
        z = [list(col)[::-1] for col in zip(*z)][::-1]
        txt = [list(col)[::-1] for col in zip(*txt)][::-1]
        hov = [list(col)[::-1] for col in zip(*hov)][::-1]
        xlabels, ylabels = list(ylabels)[::-1], xlabels[::-1]

    fig.add_trace(go.Heatmap(
        z=z, x=xlabels, y=ylabels, zmin=0, zmax=1, xgap=3, ygap=3,
        colorscale=[[0.0, _c("#F0DEDA")], [0.25, _c("#F7EEEC")], [0.5, _c(FILL)],
                    [0.75, _c("#EEF4F0")], [1.0, _c("#DDE9E1")]],
        showscale=False,
        customdata=hov, hovertemplate="%{customdata}<extra></extra>"))
    for r, row in enumerate(txt):
        for c, ch in enumerate(row):
            faded = ch.endswith("~")
            glyph = ch.rstrip("~")
            col = OK if glyph == "✓" else ALERT
            fig.add_annotation(x=xlabels[c], y=ylabels[r], text=glyph, showarrow=False,
                               opacity=0.4 if faded else 1.0,
                               font=dict(family=SANS, size=_sc(15), color=_c(col)))
    transposed = MATRIX_ENDS["on"] and len(sess.turns) >= 1
    if 0 < n_open < len(items):
        if transposed:
            # Items run along x now, so the open-book / unseen boundary is a
            # vertical rule between the two groups of columns.
            fig.add_shape(type="line", yref="paper", y0=0, y1=1, xref="x",
                          x0=n_open - 0.5, x1=n_open - 0.5,
                          line=dict(color=_c(INK_SOFT), width=1.2, dash="dash"))
        else:
            # heatmap rows are reversed(items): open-book rows sit at the top;
            # the rule falls between row index len-n_open-1 and len-n_open.
            ypos = len(items) - n_open - 0.5
            fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y",
                          y0=ypos, y1=ypos,
                          line=dict(color=_c(INK_SOFT), width=1.2, dash="dash"))
    h = int(_sc((84 if PLOT_TITLES['on'] else 58) + (34 if transposed else 26) * len(ylabels)))
    title_extra = (" · aligned sheet and the last turn"
                   if MATRIX_ENDS["on"] and len(sess.turns) > 1 else "")
    title = ("✓ as the persona · ✗ not · above the line: answer is in the prompt · below: unseen"
             + title_extra
             if arm == "QUIRKYMIND" else ARM_NAME[arm])
    _base_layout(fig, title, h,
                 legend=False, margin_t=40)
    fig.update_xaxes(side="top", tickfont=dict(family=MONO, size=_sc(11.5)),
                     showgrid=False, ticks="")
    fig.update_yaxes(tickfont=dict(family=MONO, size=_sc(13)), showgrid=False, ticks="")
    fig.update_layout(margin=dict(l=8, r=14, t=52 if PLOT_TITLES['on'] else 24, b=2))
    return fig


# =====================================================================
# (C) Backstory × stance alignment spectrum
# =====================================================================
# The matrix says WHICH items hold; this says WHERE the persona sits between
# "generic model" and "this exact person", on two axes at once. Both axes are
# read off the same probe pass as the matrix — no extra model calls — so the
# figure is a re-expression of the answer sheet, not a second measurement:
#
#   x  backstory alignment  = mean over Q_b items of sigma(margin / T)
#   y  stance alignment     = mean over Q_s items of sigma(margin / T)
#
# where `margin` is the aligned option's length-normalised log-likelihood
# lead over its strongest rival. A hard hit rate on 3 items can only take
# four values and hides erosion before the flip; the soft form moves as the
# lead shrinks. Diagnostic items only, matching the headline number.
#
# Within each turn the path visits the sheet after the first reply, then
# after every refinement round, then the verified final sheet. That is the
# claim "each round pulls the persona back toward its backstory + stance",
# drawn from the data rather than stated.
SPECTRUM_TEMP = 0.5      # nats per token; sigma(±1.0) ≈ 0.88 / 0.12


def item_pool(sess: "Session") -> List[Probe]:
    """The items every headline in panels C and D is computed on: the unseen
    ones when the bank marks them (the Table-2 quantity), else all."""
    return heldout(sess) or list(sess.persona.probes)


def soft_agreement(sess: "Session", audit: Audit, kind: str) -> Tuple[Optional[float], int, bool]:
    """(soft agreement in [0,1], n items used, unseen-only?) for one kind,
    over the same item pool as panel D's bars, so the two panels move
    together. Falls back to all items of the kind if the pool has none."""
    items = [p for p in item_pool(sess) if p.kind == kind]
    diag = bool(heldout(sess))
    if not items:
        items = [p for p in sess.persona.probes if p.kind == kind]
        diag = False
    if not items:
        return None, 0, diag
    vals = []
    for p in items:
        m = max(-30.0, min(30.0, float(audit.margin.get(p.id, 0.0)) / SPECTRUM_TEMP))
        vals.append(1.0 / (1.0 + math.exp(-m)))
    return sum(vals) / len(vals), len(items), diag


def _spectrum_xy(sess: "Session", audit: Audit) -> Tuple[Optional[float], Optional[float]]:
    xb, _, _ = soft_agreement(sess, audit, "bio")
    ys, _, _ = soft_agreement(sess, audit, "stance")
    return xb, ys


def _hex_rgba(hex_col: str, a: float) -> str:
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a:.3f})"


def _soft_items(audit: Audit, items: List[Probe]) -> Optional[float]:
    if not items:
        return None
    vals = []
    for p in items:
        m = max(-30.0, min(30.0, float(audit.margin.get(p.id, 0.0)) / SPECTRUM_TEMP))
        vals.append(1.0 / (1.0 + math.exp(-m)))
    return sum(vals) / len(vals)


def _plane_items(sess: "Session") -> Tuple[List[Probe], List[Probe], bool]:
    """(congruous items, stance items, unseen_only). The plane is about the
    UNSEEN items — the ones the persona has to infer, not read back. If a
    kind has no unseen item, all items of that kind stand in for it."""
    hb = [p for p in heldout(sess) if p.kind == "bio"]
    hs = [p for p in heldout(sess) if p.kind == "stance"]
    if hb or hs:
        return (hb or [p for p in sess.persona.probes if p.kind == "bio"],
                hs or [p for p in sess.persona.probes if p.kind == "stance"], True)
    return ([p for p in sess.persona.probes if p.kind == "bio"],
            [p for p in sess.persona.probes if p.kind == "stance"], False)


def _plane_xy(sess: "Session", audit: Optional[Audit]) -> Tuple[Optional[float], Optional[float]]:
    if audit is None:
        return None, None
    bi, st, _ = _plane_items(sess)
    return _soft_items(audit, bi), _soft_items(audit, st)


def _hits_label(sess: "Session", audit: Audit) -> str:
    c = split_counts(sess, audit)
    if c["unseen"][1]:
        return f"unseen {c['unseen'][0]}/{c['unseen'][1]} · all {c['all'][0]}/{c['all'][1]}"
    return f"{c['all'][0]}/{c['all'][1]}"


PLANE_STYLE = {   # regime -> (symbol, fill) ; colours come from ARM_COLOR so C and D match
    "BASELINE": ("square", "open"), "SINGLE": ("diamond", "open"),
    "TEXTONLY": ("triangle-up", "open"), "QUIRKYMIND": ("circle", "solid"),
}


def fig_spectrum(sess: Optional["Session"]):
    """Panel C plane — the same quantity as panel D, drawn in two dimensions.

    x = how well the UNSEEN congruous items hold, y = how well the UNSEEN
    stance items hold (soft agreement, 0..1). The four regimes at the latest
    turn are marked with the same colours and patterns as the bars in D, so a
    +1 there is a visible gap here between the filled circle (QuirkyMind) and
    the hollow triangle (same conversation, prefix off). QuirkyMind's own
    path from alignment to the latest turn is the only trajectory drawn."""
    fig = go.Figure()
    H = int(_sc(330))
    _base_layout(fig, "Unseen items: congruous (x) vs stance (y) — this persona, aligned → latest", H)
    lo, hi = -0.06, 1.12
    # Four quadrants: a cross at 50 % on both axes, tinted so the target
    # quadrant reads as the goal and the other three as "not there yet".
    for x0, y0, col, a_ in [(0.5, 0.5, OK, 0.10), (0.5, 0.0, ALERT, 0.045),
                            (0.0, 0.5, ALERT, 0.045), (0.0, 0.0, ALERT, 0.045)]:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x0 + 0.5, y1=y0 + 0.5, layer="below",
                      line=dict(width=0), fillcolor=_hex_rgba(_c(col), a_))
    fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1, line=dict(color=_c(INK_SOFT), width=1.1))
    fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5, line=dict(color=_c(INK_SOFT), width=1.1))
    cap = dict(size=_sc(10), color=_c(MUTED), family=SANS)
    for x, y, t_, xa, ya in [(0.985, 0.985, "backstory + stance", "right", "top"),
                             (0.985, 0.015, "backstory only", "right", "bottom"),
                             (0.015, 0.985, "stance only", "left", "top"),
                             (0.015, 0.015, "neither", "left", "bottom")]:
        fig.add_annotation(x=x, y=y, text=t_, showarrow=False, xanchor=xa, yanchor=ya, font=cap)
    tick = dict(family=MONO, size=_sc(11))
    fig.update_xaxes(range=[lo, hi], tickvals=[0, 0.5, 1], tickformat=".0%", showgrid=False,
                     zeroline=False, tickfont=tick,
                     title=dict(text=f'<span style="color:{_c(BACKSTORY_INK)}"><b>backstory</b> · unseen congruous</span>',
                                font=dict(size=_sc(12)), standoff=4))
    fig.update_yaxes(range=[lo, hi], tickvals=[0, 0.5, 1], tickformat=".0%", showgrid=False,
                     zeroline=False, tickfont=tick,
                     title=dict(text=f'<span style="color:{_c(STANCE_INK)}"><b>stance</b> · unseen incongruous</span>',
                                font=dict(size=_sc(12)), standoff=4))
    fig.add_trace(go.Scatter(
        x=[1], y=[1], mode="markers+text", name="target", text=["target"], showlegend=False,
        textposition="top center", textfont=dict(size=_sc(10.5), family=MONO, color=_c(INK)),
        marker=dict(size=_sc(17), symbol="star", color=_c(INK), line=dict(width=1, color=_c(PANEL))),
        hoverinfo="skip"))
    if sess is None or sess.baseline is None:
        return fig

    # QuirkyMind path: aligned -> latest, labelled at both ends only.
    pts = [("aligned", sess.baseline, "aligned, before turn 1")]
    pts += [(f"T{t.index}", t.audit, f"turn {t.index} · {_wrap(t.user)}") for t in sess.turns]
    xy = [_plane_xy(sess, a) for _, a, _ in pts]
    keep = [(p, c) for p, c in zip(pts, xy) if c[0] is not None and c[1] is not None]
    if keep:
        import numpy as np
        n = len(keep)
        shade = np.linspace(0.35, 1.0, max(n, 2))[:n]
        labels = [keep[0][0][0]] + [""] * max(n - 2, 0) + ([keep[-1][0][0]] if n > 1 else [])
        # A label above a point near the top edge, or below one near the
        # bottom edge, ends up outside the axes and gets cut off — pick the
        # side that has room.
        def _pos(c):
            return "bottom right" if c[1] > 0.85 else "top right"
        fig.add_trace(go.Scatter(
            x=[c[0] for _, c in keep], y=[c[1] for _, c in keep], mode="lines+markers+text",
            name=ARM_NAME["QUIRKYMIND"], text=labels,
            textposition=[_pos(c) for _, c in keep],
            textfont=dict(size=_sc(11), family=MONO, color=_c(INK)), cliponaxis=False,
            line=dict(color=_hex_rgba(_c(QM), 0.35), width=1.2),
            marker=dict(size=[_sc(8)] * (n - 1) + [_sc(14)], symbol="circle",
                        color=[_hex_rgba(_c(QM), s_) for s_ in shade],
                        line=dict(width=1.5, color=_c(PANEL))),
            customdata=[f"{p[2]}<br>held {_hits_label(sess, p[1])}" for p, _ in keep],
            hovertemplate="%{customdata}<extra></extra>"))

    # The other regimes at the latest comparable turn — same colours as panel D.
    # Panel C shows this persona only: aligned -> latest turn. The other
    # regimes (No-Turn, Single-Turn, steering off) are a comparison, and the
    # comparison lives in panel D; putting their markers here made the plane
    # read as a second, differently-styled version of D's bars.
    ref = {}
    for a, (au, when) in ref.items():
        x, y = _plane_xy(sess, au)
        if x is None or y is None:
            continue
        sym, fill = PLANE_STYLE[a]
        col = _c(ARM_COLOR[a] if a != "TEXTONLY" else QM)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", name=ARM_NAME[a],
            marker=dict(size=_sc(13), symbol=sym + ("-open" if fill == "open" else ""),
                        color=col, line=dict(width=2, color=col)),
            hovertemplate=f"{ARM_NAME[a]}<br>{ARM_BLURB[a]}<br>"
                          f"held {_hits_label(sess, au)}<extra></extra>"))
    fig.update_layout(legend=dict(font=dict(size=_sc(10.5))))
    return fig


# =====================================================================
# (D) Backstory–stance geometry in sentence-embedding space
# =====================================================================
# Where each arm's reply sits relative to four anchors: the backstory, the
# declared stance, their union (the alignment target) and the stereotype
# stance it is measured against. Three honesty constraints are built in:
#   * The default view fixes the axes to those anchors (see _anchor_axes),
#     because unsupervised PCA / t-SNE on a handful of reply vectors follow
#     topic variance and routinely hide the persona signal entirely.
#   * PCA and t-SNE remain available, labelled as fitted per session — their
#     positions and distances are NOT comparable across sessions.
#   * t-SNE on ~10-20 points is unstable; it is offered with a warning.
_EMB_CACHE: Dict[str, "np.ndarray"] = {}


def embed(text: str):
    """Sentence embedding for the drift monitor. Uses SBERT when available —
    the paper's dual-stream encoder is SBERT-based — and falls back to
    mean-pooled hidden states from the same LM otherwise, so the panel still
    works without the extra dependency."""
    import numpy as np
    # No character clipping here: the full backstory runs past 400 characters
    # and clipping it silently moved the backstory anchor. The encoders apply
    # their own token limits (SBERT ~256 tokens, the LM path max_length=256).
    key = text.strip()
    if key in _EMB_CACHE:
        return _EMB_CACHE[key]
    if MOCK or base_lm is None:
        v = np.random.default_rng(_seed(key)).normal(size=64)
        for w in set(re.findall(r"[a-z']+", key.lower())):
            v += np.random.default_rng(_seed(w)).normal(size=64)
    elif _SBERT is not None:
        v = _SBERT.encode(key, convert_to_numpy=True)
    else:
        with torch.inference_mode():
            ids = tok(key, return_tensors="pt", truncation=True,
                      max_length=256).input_ids.to(device)
            h = base_lm(input_ids=ids, output_hidden_states=True).hidden_states[-1]
            v = h.mean(dim=1)[0].float().cpu().numpy()
    v = v / ((v ** 2).sum() ** 0.5 + 1e-8)
    _EMB_CACHE[key] = v
    return v


_SBERT = None
if not MOCK:
    try:
        from sentence_transformers import SentenceTransformer
        _SBERT = SentenceTransformer(
            os.environ.get("QUIRKYMIND_SBERT",
                           "sentence-transformers/all-MiniLM-L6-v2"), device=device)
    except Exception:
        _SBERT = None          # fall back to LM mean-pooling


GEO_METHODS = ["Anchor axes", "PCA", "t-SNE"]


def _geo_anchors(p: Persona) -> List[Tuple[str, str, str]]:
    """(label, text, role) for the anchors every reply is measured against.

    The three the alignment is TO — backstory, stance, and their union — plus
    the stereotype stance the persona is measured AGAINST. All four are
    embedded with the same encoder as the replies."""
    return [("backstory", p.backstory_text(), "backstory"),
            ("stance (declared)", p.stance_text("declared"), "stance"),
            ("backstory + stance (target)", p.full_text(), "target"),
            ("stereotype stance (expected)", p.stance_text("expected"), "stereotype")]


def _anchor_axes(X, anchor_idx: Dict[str, int]):
    """Two anchor-defined, anchor-normalised coordinates.

        x = cos(v, backstory)                              backstory sits at 1
        y = (cos(v, stance) - cos(v, stereotype))          stance sits at +1,
            / (1 - cos(stance, stereotype))                stereotype at -1

    So the anchors are where they are BY CONSTRUCTION, and every reply reads
    as "how close to the backstory" and "nearer the declared stance (+) or
    the stereotype (-)". The raw picture — plain projections of unit vectors
    — put the two stance sentences almost on top of each other and far from
    the backstory sentence, simply because a demographic line and an opinion
    line share few words in SBERT space; the replies then floated in the
    middle, far from every anchor, and the axes said nothing. Normalising to
    the anchors removes that artefact. A reply can exceed ±1 on y when it is
    more one-sided than the stance sentence itself; the range is clipped at
    ±1.5 and the footnote says so."""
    import numpy as np
    b = X[anchor_idx["backstory"]]
    s_ = X[anchor_idx["stance"]]
    x_ = X[anchor_idx["stereotype"]]
    denom = 1.0 - float(s_ @ x_)
    if denom < 1e-4:
        return None                  # declared == expected: no stance axis
    xs = X @ b
    ys = ((X @ s_) - (X @ x_)) / denom
    return np.stack([xs, np.clip(ys, -1.5, 1.5)], axis=1)


def fig_geometry(sess: Optional["Session"], method: str = "Anchor axes"):
    import numpy as np
    fig = go.Figure()
    if sess is None:
        return _base_layout(fig, "Backstory–stance geometry", 340, legend=False)
    p = sess.persona
    labels, vecs, groups, roles = [], [], [], []
    anchor_idx: Dict[str, int] = {}
    for name, text, role in _geo_anchors(p):
        anchor_idx[role] = len(vecs)
        labels.append(name); vecs.append(embed(text)); groups.append("anchor"); roles.append(role)
    for t in sess.turns:
        for a, txt in (t.arm_replies or {"QUIRKYMIND": t.reply}).items():
            labels.append(f"{ARM_SHORT[a]}·T{t.index}")
            vecs.append(embed(txt)); groups.append(a); roles.append("reply")
    X = np.stack(vecs)

    note, used = "", method
    if method == "Anchor axes":
        xy = _anchor_axes(X, anchor_idx)
        if xy is None:
            used = "PCA"
            note = ("declared and expected stance embed identically — no stance axis; "
                    "fell back to PCA")
    if used == "t-SNE":
        if X.shape[0] >= 5:
            from sklearn.manifold import TSNE
            xy = TSNE(n_components=2, perplexity=max(2, min(10, (X.shape[0] - 1) // 3)),
                      init="pca", random_state=0).fit_transform(X)
        else:
            used = "PCA"
            note = "t-SNE needs at least 5 points — showing PCA"
    if used == "PCA":
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2).fit_transform(X)

    ANCHOR_STYLE = {          # role -> (colour, symbol, size); blue = backstory-
        "backstory": (BACKSTORY_INK, "square", 14),      # related, purple = stance-related
        "stance": (STANCE_INK, "circle", 14),
        "target": (INK, "star", 17),
        "stereotype": (STANCE_INK, "circle-open", 14),
    }
    for role, (col, sym, size) in ANCHOR_STYLE.items():
        i = anchor_idx[role]
        short = {"backstory": "backstory", "stance": "stance", "target": "target",
                 "stereotype": "stereotype"}[role]
        tpos = {"backstory": "bottom center", "stance": "top center", "target": "top center",
                "stereotype": "bottom center"}[role]
        fig.add_trace(go.Scatter(
            x=[xy[i, 0]], y=[xy[i, 1]], mode="markers+text", name=labels[i],
            text=[short], textposition=tpos,
            textfont=dict(size=10.5, family=MONO, color=_c(col)),
            marker=dict(size=size, symbol=sym, color=_c(col),
                        line=dict(width=1.6, color=_c(col if "open" in sym else PANEL))),
            hovertemplate=f"anchor · {labels[i]}<extra></extra>"))
    colors = {"QUIRKYMIND": QM, "SINGLE": "#7E9C9F", "BASELINE": "#C8C3B8"}
    symbols = {"QUIRKYMIND": "circle", "SINGLE": "triangle-up", "BASELINE": "x"}
    for g in ARMS:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        if not idx:
            continue
        fig.add_trace(go.Scatter(
            x=xy[idx, 0], y=xy[idx, 1], mode="markers",
            text=[labels[i] for i in idx],
            name=f"{ARM_SHORT[g]} replies", hovertemplate=f"{ARM_NAME[g]}: {ARM_BLURB[g]}<br>%{{text}}<extra></extra>",
            marker=dict(size=9, color=_c(colors[g]), symbol=symbols[g],
                        line=dict(width=1, color="#FFFFFF"))))
    if used == "Anchor axes":
        fig.update_xaxes(title=dict(text="similarity to the backstory (cosine)",
                                    font=dict(size=12, color=_c(BACKSTORY_INK))),
                         range=[-0.05, 1.08], tickvals=[0, 0.5, 1], tickformat=".1f",
                         gridcolor=GRID, zeroline=False, tickfont=dict(family=MONO, size=11))
        fig.update_yaxes(title=dict(text="← stereotype        declared stance →",
                                    font=dict(size=12, color=_c(STANCE_INK))),
                         range=[-1.6, 1.6], tickvals=[-1, 0, 1],
                         ticktext=["stereotype", "midway", "declared"],
                         gridcolor=GRID, zeroline=True, zerolinecolor=_c(INK_SOFT),
                         zerolinewidth=1.2, tickfont=dict(family=MONO, size=11))
        foot = ("axes are normalised to the anchors: backstory = 1 on x, declared stance = +1 "
                "and stereotype = -1 on y; the zero line is equidistant from both stances; "
                "y is clipped at ±1.5")
    else:
        fig.update_xaxes(showticklabels=False, gridcolor=GRID, zeroline=False)
        fig.update_yaxes(showticklabels=False, gridcolor=GRID, zeroline=False)
        foot = (f"{used} fitted on this session only — axes follow variance, which is "
                f"usually topic, not persona; a blob here does not mean no alignment")
    if note:
        foot = f"{note} · {foot}"
    fig.add_annotation(x=0.0, y=-0.16, xref="paper", yref="paper", showarrow=False,
                       xanchor="left", yanchor="top", align="left",
                       text=_wrap(foot, 110), font=dict(size=10.5, color=MUTED, family=SANS))
    fig.update_layout(margin=dict(b=56))
    if not sess.turns:
        fig.add_annotation(x=0.5, y=0.97, xref="paper", yref="paper", showarrow=False,
                           text="anchors only — send a turn to place the replies",
                           font=dict(size=11, color=_c(MUTED), family=SANS))
    _base_layout(fig, f"Backstory–stance geometry ({used})", 360)
    return fig


# =====================================================================
# (D) Offline evaluation on sub-datasets
# =====================================================================
# ---------------------------------------------------------------------
# EDIT THIS BLOCK before submitting. These are placeholder numbers so the
# panel renders; they are NOT results. While OFFLINE_IS_PLACEHOLDER is True
# the panel shows a visible warning chip, so a placeholder can't quietly end
# up in a screenshot or a paper figure.
# ---------------------------------------------------------------------
# Table 2 of the paper: PQA accuracy (%) by topic domain, three inference
# regimes, three backbones. These are the published numbers, not placeholders;
# the metric radio picks the backbone. Override with QUIRKYMIND_OFFLINE=...
OFFLINE_IS_PLACEHOLDER = False
OFFLINE_SUBSETS = ["SOCIAL", "SCIENCE", "RELIGION", "POLITICS", "METHOD.", "JOURN.",
                   "GLOBAL", "MacroAcc"]
OFFLINE_METRICS = {
    "Llama": {"label": "PQA Acc (%) · Llama 3.1-8B ↑", "higher_is_better": True,
              "blurb": "persona-steered QA accuracy, Table 2"},
    "Qwen": {"label": "PQA Acc (%) · Qwen 2.5-7B ↑", "higher_is_better": True,
             "blurb": "persona-steered QA accuracy, Table 2"},
    "Mistral": {"label": "PQA Acc (%) · Mistral 3-8B ↑", "higher_is_better": True,
                "blurb": "persona-steered QA accuracy, Table 2"},
}
OFFLINE_SYSTEMS = {                       # regime -> backbone -> per-topic list
    "Multi-Turn (QuirkyMind)": {
        "Llama":   [33.9, 33.1, 18.5, 48.4, 49.1, 40.0, 62.1, 40.7],
        "Qwen":    [36.7, 34.0, 17.8, 30.8, 44.1, 38.2, 61.8, 37.6],
        "Mistral": [35.4, 35.3, 17.8, 47.2, 34.8, 42.1, 59.7, 38.9]},
    "Single-Turn": {
        "Llama":   [27.8, 33.1, 13.6, 30.5, 27.9, 16.8, 62.1, 30.2],
        "Qwen":    [28.7, 35.2, 14.3, 30.5, 44.1, 26.4, 61.8, 34.5],
        "Mistral": [31.0, 38.3, 17.3, 13.8, 16.5, 28.5, 59.4, 29.3]},
    "No-Turn": {
        "Llama":   [25.8, 31.4, 19.2, 13.8, 27.4, 24.5, 59.4, 28.8],
        "Qwen":    [28.2, 31.2, 12.3, 13.8, 49.1, 24.5, 37.5, 28.1],
        "Mistral": [31.8, 31.2, 14.7, 13.8, 16.8, 24.5, 59.4, 27.4]},
}
# Luminance-separated on purpose so the three regimes stay orderable in
# greyscale; pattern fills carry the distinction a second time.
OFFLINE_PATTERN = {"Multi-Turn (QuirkyMind)": "", "Single-Turn": "/", "No-Turn": "."}
OFFLINE_SYSTEM_COLOR = {"Multi-Turn (QuirkyMind)": QM, "Single-Turn": "#7E9C9F",
                        "No-Turn": "#C8C3B8"}
OFFLINE_SOURCE = "Table 2 of the paper (ATP, 7 domains)"

# Real numbers can be supplied without touching this file:
#     QUIRKYMIND_OFFLINE=/path/to/results.json python quirkymind_demo.py
# {
#   "subsets": ["Political", "Religious", "Occupational"],
#   "metrics": {"IND":  {"label": "IND \u2191", "higher_is_better": true,
#                        "blurb": "incongruity adherence"},
#               "EXAG": {"label": "EXAG \u2193", "higher_is_better": false,
#                        "blurb": "exaggeration"}},
#   "systems": {"QuirkyMind": {"IND": [0.0, 0.0, 0.0], "EXAG": [0.0, 0.0, 0.0]},
#               "Ablation":   {"IND": [0.0, 0.0, 0.0], "EXAG": [0.0, 0.0, 0.0]}}
# }
# One list entry per subset, in the same order as "subsets".
_OFFLINE_PATH = os.environ.get("QUIRKYMIND_OFFLINE", "")
if _OFFLINE_PATH and os.path.exists(_OFFLINE_PATH):
    with open(_OFFLINE_PATH, encoding="utf-8") as _f:
        _blob = json.load(_f)
    OFFLINE_SUBSETS = _blob["subsets"]
    OFFLINE_METRICS = _blob.get("metrics", OFFLINE_METRICS)
    OFFLINE_SYSTEMS = _blob["systems"]
    OFFLINE_IS_PLACEHOLDER = False
    OFFLINE_SOURCE = os.path.basename(_OFFLINE_PATH)
    # Colour and pattern any systems the file introduces. Spares are filtered
    # against what the built-in systems already use, otherwise a new system
    # gets handed a colour and hatch identical to an existing one and the two
    # become indistinguishable in the chart.
    _used_c = {OFFLINE_SYSTEM_COLOR[k] for k in OFFLINE_SYSTEMS if k in OFFLINE_SYSTEM_COLOR}
    _used_p = {OFFLINE_PATTERN[k] for k in OFFLINE_SYSTEMS if k in OFFLINE_PATTERN}
    _spare_c = [c for c in ["#8E7F9B", "#A8926F", "#6F8F7A", "#9B7F7F", "#7F8B9B"]
                if c not in _used_c]
    _spare_p = [p for p in ["x", "\\\\", "+", "-", "|"] if p not in _used_p]
    for _i, _sys in enumerate(s_ for s_ in OFFLINE_SYSTEMS if s_ not in OFFLINE_SYSTEM_COLOR):
        OFFLINE_SYSTEM_COLOR[_sys] = _spare_c[_i % len(_spare_c)]
        OFFLINE_PATTERN[_sys] = _spare_p[_i % len(_spare_p)]
for _sys in OFFLINE_SYSTEMS:
    OFFLINE_SYSTEM_COLOR.setdefault(_sys, "#8E8A80")
    OFFLINE_PATTERN.setdefault(_sys, "")


def fig_offline(metric: str):
    fig = go.Figure()
    meta = OFFLINE_METRICS[metric]
    for sysname, rows in OFFLINE_SYSTEMS.items():
        fig.add_trace(go.Bar(
            name=sysname, x=OFFLINE_SUBSETS, y=rows[metric],
            marker=dict(color=_c(OFFLINE_SYSTEM_COLOR[sysname]), opacity=0.9,
                        pattern=dict(shape=OFFLINE_PATTERN[sysname],
                                     fgcolor="#FFFFFF", size=5, solidity=0.22),
                        line=dict(width=0)),
            text=[f"{v:.1f}" for v in rows[metric]], textposition="outside",
            textfont=dict(family=MONO, size=10.5, color=INK),
            hovertemplate="%{fullData.name} · %{x}: %{y:.1f}%<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_xaxes(tickfont=dict(family=SANS, size=11.5, color=INK), showgrid=False)
    ymax = max(v for rows in OFFLINE_SYSTEMS.values() for v in rows[metric])
    fig.update_yaxes(range=[0, ymax * 1.18], gridcolor=GRID, zeroline=False,
                     tickfont=dict(family=MONO, size=11.5))
    if OFFLINE_IS_PLACEHOLDER:
        # Drawn inside the plot so a screenshot cannot crop the warning away.
        fig.add_annotation(text="PLACEHOLDER DATA", x=0.5, y=0.5, xref="paper",
                           yref="paper", showarrow=False, textangle=-18,
                           font=dict(family=SANS, size=34, color="rgba(162,86,74,.16)"))
    # Title kept short: a long one runs under the modebar and wraps badly
    # once the figure is scaled down to a paper column.
    return _base_layout(fig, f"{meta['label']} by topic domain", 270)


def offline_md(metric: str) -> str:
    head = "| System | " + " | ".join(OFFLINE_SUBSETS) + " | Mean |\n"
    head += "|---|" + "---|" * (len(OFFLINE_SUBSETS) + 1) + "\n"
    rows = []
    best = max if OFFLINE_METRICS[metric]["higher_is_better"] else min
    means = {s: sum(v[metric]) / len(v[metric]) for s, v in OFFLINE_SYSTEMS.items()}
    winner = best(means, key=lambda s: means[s])
    for s, v in OFFLINE_SYSTEMS.items():
        cells = " | ".join(f"{x:.1f}" for x in v[metric])
        name = f"**{s}**" if s == winner else s
        rows.append(f"| {name} | {cells} | {means[s]:.1f} |")
    meta = OFFLINE_METRICS[metric]
    direction = "higher is better" if meta["higher_is_better"] else "lower is better"
    warn = (f'<div class="qm-chip qm-chip-ok">source: {OFFLINE_SOURCE}</div>\n\n'
            if not OFFLINE_IS_PLACEHOLDER else
            '<div class="qm-chip qm-chip-warn">Placeholder numbers — pass '
            'QUIRKYMIND_OFFLINE=results.json, or edit OFFLINE_SYSTEMS</div>\n\n')
    warn += (f"**{metric}** — {meta['blurb']}, {direction}. Bold marks the best "
             f"mean. Each column is one persona sub-dataset.\n\n")
    return (warn + head + "\n".join(rows)
            + "\n\n_Published PQA results (Table 2, persona-level 70/10/20 split). The "
              "aggregate gain is roughly +10 points Macro; on the 3–6 items of a single "
              "persona in the live panel that gain is one item and often invisible, "
              "which is why this panel — not one conversation — is the evidence._")


# =====================================================================
# Backstory / stance highlighting in free text (panels B and D)
# =====================================================================
# Every sentence of a reply is tagged by which half of the specification it
# draws on: backstory (blue), stance (purple), or neither. The tag is by
# sentence-embedding similarity to the persona's own backstory and stance
# texts (plus the corresponding MCQA items), with the same encoder as the
# geometry panel, so what is coloured and what is measured share one notion
# of "about the backstory" / "about the stance".
#
# Colour marks TOPIC, not correctness. A purple sentence that argues the
# stereotype's side is still about the stance — whether it is the RIGHT
# stance is panel C's job, and this pass never reads the answer sheet.
TAG_FLOOR = 0.06 if MOCK else 0.22   # min similarity to be tagged at all
TAG_MARGIN = 0.02 if MOCK else 0.04  # min lead of one half over the other
_TAG_ANCHORS: Dict[str, Tuple[list, list]] = {}


def _tag_anchor_sets(p: Persona):
    if p.label not in _TAG_ANCHORS:
        bs = [embed(p.backstory_text())] + [
            embed(f"{q.question} {q.target_text()}") for q in p.probes if q.kind == "bio"]
        st = [embed(p.stance_text("declared")), embed(p.stance_text("expected"))] + [
            embed(f"{q.question} {q.target_text()}") for q in p.probes if q.kind == "stance"]
        _TAG_ANCHORS[p.label] = (bs, st)
    return _TAG_ANCHORS[p.label]


def tag_sentence(p: Persona, sent: str) -> Tuple[str, float, float]:
    """('bs' | 'st' | '', sim_backstory, sim_stance) for one sentence.

    sim_* is the highest cosine similarity between the sentence embedding and
    any text in that half's anchor set. A sentence is coloured only if the
    winning half clears TAG_FLOOR and leads the other by TAG_MARGIN; otherwise
    it is left plain. Both numbers are exposed on hover so the colour is
    always checkable, never a black box."""
    bs, st = _tag_anchor_sets(p)
    v = embed(sent)
    sim_b = max(float(v @ a) for a in bs)
    sim_s = max(float(v @ a) for a in st)
    if len(sent.split()) < 3:
        return "", sim_b, sim_s
    if sim_b >= TAG_FLOOR and sim_b >= sim_s + TAG_MARGIN:
        return "bs", sim_b, sim_s
    if sim_s >= TAG_FLOOR and sim_s >= sim_b + TAG_MARGIN:
        return "st", sim_b, sim_s
    return "", sim_b, sim_s


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s]


def highlight_html(p: Persona, text: str) -> Tuple[str, Dict[str, int]]:
    """Speech with each sentence wrapped in a coloured span whose tooltip
    carries the two similarities behind the colour. Returns the HTML and the
    count of sentences per tag."""
    import html as _html
    out, counts = [], {"bs": 0, "st": 0, "": 0}
    for s in _sentences(text):
        tag, sb, ss = tag_sentence(p, s)
        counts[tag] += 1
        esc = _html.escape(s)
        tip = f"backstory {sb:.2f} · stance {ss:.2f}"
        cls = f"qm-{tag}" if tag else "qm-none-sent"
        out.append(f'<span class="{cls}" title="{tip}">{esc}</span>')
    return " ".join(out), counts


def encoder_name() -> str:
    if MOCK:
        return "hash bag-of-words (mock)"
    if _SBERT is not None:
        return os.environ.get("QUIRKYMIND_SBERT", "sentence-transformers/all-MiniLM-L6-v2")
    return f"{MODEL_NAME} mean-pooled hidden state"


def highlight_legend_html() -> str:
    return ('<div class="qm-legend">'
            '<span class="qm-swatch qm-bs">backstory</span>'
            '<span class="qm-swatch qm-st">stance</span>'
            '<details class="qm-pop"><summary>how the colours are computed</summary>'
            f'<div class="qm-pop-body">Each sentence is embedded (<code>{encoder_name()}</code>) '
            'and compared to the backstory text and to the stance text, each padded with its '
            f'own survey items. Whichever is closer wins if it clears {TAG_FLOOR:.2f} and leads '
            f'by {TAG_MARGIN:.2f}; otherwise the sentence stays plain. Hover a sentence for the '
            'two numbers. Colour says what the sentence is about, not whether it is right; '
            'panel C measures that.</div></details></div>')


# =====================================================================
# Text views
# =====================================================================
def screened_out(sess: "Session") -> List[Probe]:
    """Items a control already answers the target way. Informational only —
    shown in the Item key — they are NOT removed from the score."""
    out = []
    for p in sess.persona.probes:
        if sess.base_rate is not None and sess.base_rate.hit(p):
            out.append(p)
        elif sess.stereo_rate is not None and sess.stereo_rate.hit(p):
            out.append(p)
    return out


def usable(sess: "Session") -> List[Probe]:
    """Every item in the bank counts. (Earlier versions dropped items a
    control already answered; that made the headline depend on the controls
    and left personas with a score over two items. The screening survives
    as a note in the Item key.)"""
    out = list(sess.persona.probes)
    return out


def heldout(sess: "Session") -> List[Probe]:
    """Items whose answer is nowhere in the prompt. If the bank predates the
    context flag, nothing is held out and the list is empty."""
    return [p for p in sess.persona.probes if p.held_out]


def agreement_heldout(sess: "Session", audit: Audit) -> Tuple[int, int]:
    items = heldout(sess)
    return sum(1 for p in items if audit.hit(p)), len(items)


def openbook(sess: "Session") -> List[Probe]:
    return [p for p in sess.persona.probes if not p.held_out]


def split_counts(sess: "Session", audit: Audit) -> Dict[str, Tuple[int, int]]:
    """{'all': (h,t), 'unseen': (h,t), 'open': (h,t)} for one sheet."""
    ob, un = openbook(sess), heldout(sess)
    return {"all": agreement(sess, audit),
            "unseen": (sum(audit.hit(p) for p in un), len(un)),
            "open": (sum(audit.hit(p) for p in ob), len(ob))}


def agreement(sess: "Session", audit: Audit, kind: Optional[str] = None) -> Tuple[int, int]:
    """(hits, total) over diagnostic items only.

    Scoring the whole bank mixes in items a control already answers correctly,
    which both inflates the headline and lets a meaningless flicker on a free
    item drive the verdict. Free items stay visible in the matrix; they just
    stop voting.
    """
    items = [p for p in usable(sess) if kind is None or p.kind == kind]
    return sum(1 for p in items if audit.hit(p)), len(items)


def status_md(sess: Optional["Session"]) -> str:
    """One number and one sentence. Green when every diagnostic item is
    answered the persona's way, red otherwise — the same green/red as the
    matrix cells. Attribution detail lives in the matrix hover."""
    if sess is None or sess.baseline is None:
        return '<div class="qm-state qm-idle"><span class="qm-msg">Align a persona.</span></div>'
    p = sess.persona
    a = sess.current_audit()
    hits, tot = agreement(sess, a)
    n_free = p.n() - tot
    if tot == 0:
        return ('<div class="qm-state qm-alert"><span class="qm-num">—</span>'
                '<span class="qm-msg">No diagnostic items: every item is already '
                'answered the target way by a control. Nothing here can measure '
                'steering.</span></div>')
    use = usable(sess)
    bio_h = sum(a.hit(q) for q in use if q.kind == "bio")
    bio_t = sum(1 for q in use if q.kind == "bio")
    st_h = sum(a.hit(q) for q in use if q.kind == "stance")
    st_t = sum(1 for q in use if q.kind == "stance")
    off = [q for q in usable(sess) if not a.hit(q)]
    detail = (f'<span class="qm-numsub">items · '
              f'<span class="qm-bs">{KIND_NAME["bio"]} {bio_h}/{bio_t}</span> · '
              f'<span class="qm-st">{KIND_NAME["stance"]} {st_h}/{st_t}</span>'
              + (f' · {n_free} excluded' if n_free else '') + '</span>')
    counts = f'<span class="qm-num">{hits}/{tot}</span>{detail}'
    ids = lambda qs: ", ".join(qid_html(q) for q in qs)
    if not off:
        if not sess.turns:
            msg = "Aligned — nothing has been said to it yet."
        elif sess.turns[-1].refine_rounds:
            msg = f"Held after {sess.turns[-1].refine_rounds} refinement round(s)."
        else:
            msg = "Held."
        return f'<div class="qm-state qm-ok">{counts}<span class="qm-msg">{msg}</span></div>'
    if not sess.turns:
        return (f'<div class="qm-state qm-alert">{counts}<span class="qm-msg">'
                f'Not aligned: {ids(off)} already miss before any conversation.</span></div>')
    # Two different failures, said apart: an item that held at alignment and
    # flipped during the conversation is drift; one that never held is a
    # steering (or bank) gap and no amount of talking changes it.
    lost = [q for q in off if sess.baseline.hit(q)]
    never = [q for q in off if not sess.baseline.hit(q)]
    parts = []
    if lost:
        parts.append(f'{ids(lost)} flipped during the conversation')
    if never:
        parts.append(f'{ids(never)} off since alignment')
    return (f'<div class="qm-state qm-alert">{counts}<span class="qm-msg">'
            f'{"Drifting: " if lost else ""}{"; ".join(parts)}.</span></div>')


def answers_html(sess: Optional["Session"]) -> str:
    """Gold label next to what the model actually answered, item by item, at
    the latest turn (and at alignment for contrast). The most direct reading
    of "is it aligned": same text in both columns or not. Excluded items are
    greyed but still listed, so nothing is hidden."""
    import html as _html
    if sess is None or sess.baseline is None:
        return ""
    p = sess.persona
    a = sess.current_audit()
    use = {q.id for q in usable(sess)}

    def ans(au, q):
        i = au.answers.get(q.id)
        return q.options[i] if isinstance(i, int) and 0 <= i < len(q.options) else "?"

    col_now = "now" if sess.turns else "aligned"
    rows = []
    for q in p.probes:
        hit = a.hit(q)
        excluded = q.id not in use
        cls = ("qm-ans-ok" if hit else "qm-ans-bad") + (" qm-ans-x" if excluded else "")
        gold = _html.escape(q.target_text())
        now = _html.escape(ans(a, q))
        align_txt = _html.escape(ans(sess.baseline, q))
        align_cell = ("" if not sess.turns else
                      f'<td class="qm-ans-align">{align_txt}</td>')
        mark = "✓" if hit else "✗"
        ctx = {"backstory": "in backstory", "seed": "in seed pair", "none": "unseen"}.get(q.context, "")
        rows.append(f'<tr class="{cls}"><td class="qm-ans-id">{qid_html(q)}</td>'
                    f'<td class="qm-ans-ctx">{ctx}</td>'
                    f'<td class="qm-ans-gold">{gold}</td>{align_cell}'
                    f'<td class="qm-ans-now">{now}</td><td class="qm-ans-mark">{mark}</td></tr>')
    head_align = '<th>aligned</th>' if sess.turns else ''
    hits, tot = agreement(sess, a)
    body = "".join(rows)
    off = [q for q in p.probes if not a.hit(q)]
    lost = [q for q in off if sess.baseline.hit(q)]
    note = ""
    if sess.turns and lost:
        note = f' · <span class="qm-fold-drift">{len(lost)} flipped</span>'
    st_items = [q for q in p.probes if q.kind == "stance"]
    if st_items and sess.steer_prefix is not None and sess.steer_none is not None:
        hp = sum(sess.steer_prefix.hit(q) for q in st_items)
        hn = sum(sess.steer_none.hit(q) for q in st_items)
        cls_ = "qm-steer-ok" if hp > hn else ("qm-steer-bad" if hp < hn else "qm-steer-flat")
        note += (f' · <span class="{cls_}" title="stance items with no backstory and no '
                 f'history: soft prefix alone vs nothing — the only number here that isolates '
                 f'steering">steering alone {hp} · raw {hn} of {len(st_items)}</span>')
    c = split_counts(sess, a)
    if c["unseen"][1]:
        lead = (f'<span class="qm-fold-num">{c["unseen"][0]}/{c["unseen"][1]}</span> unseen'
                f' <span class="qm-fold-sub">· {hits}/{tot} all</span>')
    else:
        lead = f'<span class="qm-fold-num">{hits}/{tot}</span> held'
    return (f'<details class="qm-fold"><summary>'
            f'<span class="qm-fold-n">answers ▾</span></summary>'
            f'<table class="qm-ans"><thead><tr><th></th><th>answer is</th><th>gold label</th>{head_align}'
            f'<th>{col_now}</th><th></th></tr></thead><tbody>{body}</tbody></table>'
            f'</details>')


def face_html(sess: Optional["Session"]) -> str:
    """The persona's current expression, read off its own stage direction.

    Decorative and non-evaluative: it makes the character legible to someone
    standing at the booth. Panel C is unaffected by anything here."""
    if sess is None or not sess.turns:
        emoji, anim, label, stage = NEUTRAL_FACE[0], "qm-idle", "waiting", ""
    else:
        stage, _ = split_stage(sess.turns[-1].reply)
        emoji, anim, label = read_expression(stage)
    import html as _html
    if not stage:
        return ""            # no filler card when the persona gave no stage direction
    caption = f'<span class="qm-stage">{_html.escape(stage)}</span>'
    return (f'<div class="qm-face-row">'
            f'<span class="qm-face {anim}">{emoji}</span>'
            f'<span class="qm-face-text"><span class="qm-face-label">{label}</span>'
            f'{caption}</span></div>')


def _pop(summary: str, body_html: str, cls: str = "") -> str:
    """A CSS-only popover: closed it is a small chip, open it floats a card
    over the panel instead of pushing the layout around."""
    return (f'<details class="qm-pop {cls}"><summary>{summary}</summary>'
            f'<div class="qm-pop-body">{body_html}</div></details>')


def persona_html(sess: Optional["Session"]) -> str:
    """Panel A: two labelled halves, minimal running text. Definitions and any
    long backstory live in popovers, so the panel itself stays two short
    blocks and never wraps into a narrow column."""
    import html as _html
    if sess is None:
        return '<div class="qm-spec"><i>Select a persona.</i></div>'
    p = sess.persona
    nb, ns = p.n("bio"), p.n("stance")
    back = _html.escape(p.backstory_text()) + "."
    if p.domain:
        back = (f'<span class="qm-dom">{_html.escape(DOMAIN_NAMES.get(p.domain, p.domain))}'
                f' · ATP</span> ' + back)
    who = f'<div class="qm-who">{_html.escape(p.label)}</div>'
    if len(back) > 190:
        # Long hand-authored backstory: one-line identity inline, full text in
        # a popover.
        back_html = (f'{_html.escape(p.identity.rstrip("."))}. '
                     + _pop("full backstory", back))
    else:
        back_html = back
    def_b = _pop("?", "<b>Backstory</b>: who the person is. Survey items they answer like "
                      "their group are <i>congruous</i> (blue).")
    def_s = _pop("?", "<b>Stance</b>: the view their group would not expect. Those items are "
                      "<i>incongruous</i> (purple). The stance is never put in the prompt; "
                      "it has to survive through the conversation and the prefix.")
    return (
        who +
        '<div class="qm-spec">'
        '<div class="qm-half">'
        '<div class="qm-half-head"><span class="qm-tag qm-tag-bs">Backstory</span>'
        f'<span class="qm-half-def">{nb} congruous item(s)</span>{def_b}</div>'
        f'<div class="qm-half-body qm-bs-block">{back_html}</div>'
        '</div>'
        '<div class="qm-half">'
        '<div class="qm-half-head"><span class="qm-tag qm-tag-st">Stance</span>'
        f'<span class="qm-half-def">{ns} incongruous item(s)</span>{def_s}</div>'
        f'<div class="qm-half-body qm-st-block">{_stance_rows(p)}</div>'
        '</div></div>')


def _stance_rows(p: Persona) -> str:
    """One compact row per incongruous item: [Q chip] group answer → this
    person's answer. The full question sits in the chip's popover, so the
    block stays two short lines per item however long the survey wording."""
    import html as _html
    items = [q for q in p.probes if q.kind == "stance"]
    if not items or not any(q.mode for q in items):
        # Legacy personas: the two sentences, but each on one compact line.
        return (f'<div class="qm-stance-row"><span class="qm-stance-k">group</span>'
                f'<span class="qm-stance-v qm-stance-expected">{_html.escape(p.expected)}</span></div>'
                f'<div class="qm-stance-row"><span class="qm-stance-k">this person</span>'
                f'<span class="qm-stance-v qm-st"><b>{_html.escape(p.declared)}</b></span></div>')
    # A heading row cannot line up with rows that wrap, so the two sides are
    # named once, in the colours they use.
    rows = ['<div class="qm-stance-key"><span class="qm-stance-expected">group says</span>'
            '<span class="qm-arrow">→</span><span class="qm-st">this person</span></div>']
    for q in items:
        chip = _pop(qid_html(q, color=False), f"<b>{_html.escape(q.question)}</b>"
                    + (f"<br><span class='qm-muted'>seed pair — stated in the prompt</span>"
                       if q.context == "seed" else ""), cls="qm-pop-q")
        rows.append(
            f'<div class="qm-stance-item">{chip}'
            f'<span class="qm-stance-expected">{_html.escape(q.mode or "—")}</span>'
            f'<span class="qm-arrow">→</span>'
            f'<span class="qm-st"><b>{_html.escape(q.target_text())}</b></span></div>')
    return "".join(rows)


def bank_md(sess: Optional["Session"]) -> str:
    """Per-item screening, by id only — full text is in panel D → Item key."""
    if sess is None:
        return ""
    rows = []
    usable_n = 0
    for it in sess.persona.probes:
        bare = sess.base_rate is not None and sess.base_rate.hit(it)
        ster = sess.stereo_rate is not None and sess.stereo_rate.hit(it)
        ok = not bare and not ster
        usable_n += ok
        verdict = ("**usable**" if ok else
                   "distractors too weak" if bare and not ster else
                   "identity implies it — leak?" if ster and not bare
                   else "measures nothing")
        rows.append(f"| {qid_html(it)} | {'target' if bare else '—'} "
                    f"| {'target' if ster else '—'} | {verdict} |")
    note = (f"\n\n**{usable_n}/{len(sess.persona.probes)} items discriminate.**"
            "<details class=\"qm-help\"><summary>why an item can be non-diagnostic"
            "</summary><div>An item is only evidence if neither control already "
            "produces the target answer: the bare model with no persona, and the same "
            "identity primed with the stance people assume it holds. Everything else "
            "is green regardless of steering. Full question text: panel D → Item key."
            "</div></details>")
    return ("| Item | Bare model | Stereotype-primed | Verdict |\n"
            "|---|---|---|---|\n" + "\n".join(rows) + note)


def item_key_html(sess: Optional["Session"]) -> str:
    """Panel D → Item key: one card per item — id, question, the options with
    the aligned answer marked, and the screening verdict. The only place the
    complete text is printed, so panels A–C can use the short ids."""
    import html as _html
    if sess is None:
        return '<div class="qm-note">Align a persona to see its items.</div>'
    cards = []
    for it in sess.persona.probes:
        bare = sess.base_rate is not None and sess.base_rate.hit(it)
        ster = sess.stereo_rate is not None and sess.stereo_rate.hit(it)
        ctx = {"backstory": "open-book · in backstory", "seed": "open-book · in seed pair",
               "none": "unseen · comparable to Table 2"}.get(it.context, "")
        chip = (f'<span class="qm-chip {"qm-chip-ok" if it.held_out else "qm-chip-warn"}">{ctx}</span>'
                if ctx else "")
        if bare or ster:
            who = ("bare model" if bare and not ster else
                   "stereotype control" if ster and not bare else "both controls")
            chip += f'<span class="qm-chip qm-chip-warn">{who} also answers it</span>'
        lis = []
        for i, o in enumerate(it.options):
            cls = "qm-opt-target" if i == it.target else ""
            mark = "\u2713 " if i == it.target else ""
            lis.append(f'<li class="{cls}">{mark}{_html.escape(o)}</li>')
        opts = "".join(lis)
        kind_cls = "qm-bs" if it.kind == "bio" else "qm-st"
        cards.append(
            f'<div class="qm-item qm-item-{it.kind}">'
            f'<div class="qm-item-head">{qid_html(it)}'
            f'<span class="qm-item-kind {kind_cls}">{KIND_NAME[it.kind]}</span>{chip}</div>'
            f'<div class="qm-item-q">{_html.escape(it.question)}</div>'
            f'<ul class="qm-item-opts">{opts}</ul></div>')
    grid = "".join(cards)
    return (f'<div class="qm-item-grid">{grid}</div>'
            '<div class="qm-note">\u2713 marks the answer this persona was aligned to give '
            '— not a fact about the world.</div>')


ARM_COLOR = {"QUIRKYMIND": QM, "TEXTONLY": "#9AA5B8", "SINGLE": "#7E9C9F", "BASELINE": "#C8C3B8"}
ARM_PATTERN = {"QUIRKYMIND": "", "TEXTONLY": "/", "SINGLE": "x", "BASELINE": "."}


def _turns_with_arms(sess: "Session") -> List[Turn]:
    return [t for t in sess.turns if t.arm_audits and "TEXTONLY" in t.arm_audits]


BAR_ORDER = ["BASELINE", "SINGLE", "TEXTONLY", "QUIRKYMIND"]   # Table 2 order, ablation before ours


def fig_method_bars(sess: Optional["Session"]):
    """Four bars per column, Table-2 style: No-Turn, Single-Turn, Multi-Turn
    with the prefix off, Multi-Turn (QuirkyMind). The first two never see the
    conversation; the last two are the same conversation with the method off
    and on, and the number above each group is that difference."""
    fig = go.Figure()
    fig.update_yaxes(range=[0, 1.22], dtick=0.25, tickformat=".0%", gridcolor=GRID,
                     zeroline=False, tickfont=dict(family=MONO, size=11.5))
    fig.update_xaxes(tickfont=dict(family=MONO, size=12), showgrid=False)
    if sess is None or sess.baseline is None or sess.baseline_noprefix is None:
        return _base_layout(fig, "Regimes", 290, legend=False)
    turns = _turns_with_arms(sess)
    if not turns and sess.turns:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           text="Turn on <b>Run comparison arms</b>, then send a turn.",
                           font=dict(size=12.5, color=_c(MUTED), family=SANS))
        return _base_layout(fig, "Regimes", 290, legend=False)
    tot = agreement(sess, sess.baseline)[1]
    split = False          # all items, one bar per regime — the split lives in C
    # Ends view: the same two columns panel C shows — the aligned sheet and
    # the last turn. Eight groups of four bars need the full width of the
    # screen; two groups fit in a quarter of it, which is what lets all four
    # panels sit in one row of a figure. The middle turns are what the live
    # audience watches appear.
    if MATRIX_ENDS["on"] and len(turns) > 1:
        turns = [turns[-1]]
    xs = ["align"] + [f"T{t.index}" for t in turns]
    series = {
        "QUIRKYMIND": [sess.baseline] + [t.arm_audits["QUIRKYMIND"] for t in turns],
        "TEXTONLY": [sess.baseline_noprefix] + [t.arm_audits["TEXTONLY"] for t in turns],
        "SINGLE": [sess.align_arms.get("SINGLE")] + [t.arm_audits.get("SINGLE") for t in turns],
        "BASELINE": [sess.align_arms.get("BASELINE")] + [t.arm_audits.get("BASELINE") for t in turns],
    }
    for name in BAR_ORDER:
        auds = series[name]
        if any(a is None for a in auds):
            continue
        counts = [split_counts(sess, a) for a in auds]
        un = [c["unseen"][0] for c in counts]
        ob = [c["open"][0] for c in counts]
        al = [c["all"][0] for c in counts]
        col = _c(ARM_COLOR[name])
        if split:
            # Two stacked segments per bar: unseen hits (full colour, bottom)
            # and open-book hits (faded, on top). Same total as the C plane.
            fig.add_trace(go.Bar(
                name=ARM_NAME[name], x=xs, y=[h / tot for h in un], offsetgroup=name,
                text=[f"{h}" for h in un], textposition="inside", insidetextanchor="middle",
                textfont=dict(family=MONO, size=11, color="#FFFFFF" if name == "QUIRKYMIND" else INK),
                marker=dict(color=col, opacity=0.95,
                            pattern=dict(shape=ARM_PATTERN[name], fgcolor="#FFFFFF", size=5, solidity=0.22),
                            line=dict(width=0)),
                customdata=[f"{c['unseen'][0]}/{c['unseen'][1]} unseen" for c in counts],
                hovertemplate="%{fullData.name}<br>" + ARM_BLURB[name] + "<br>%{x}: %{customdata}<extra></extra>"))
            fig.add_trace(go.Bar(
                name=ARM_NAME[name] + " · open-book", x=xs, y=[h / tot for h in ob],
                base=[h / tot for h in un], offsetgroup=name, showlegend=False,
                text=[f"{h}" if h else "" for h in ob], textposition="inside", insidetextanchor="middle",
                textfont=dict(family=MONO, size=10, color=INK),
                marker=dict(color=_hex_rgba(col, 0.35), line=dict(width=0)),
                customdata=[f"{c['open'][0]}/{c['open'][1]} open-book (answer in the prompt)" for c in counts],
                hovertemplate="%{fullData.name}<br>%{x}: %{customdata}<extra></extra>"))
        else:
            fig.add_trace(go.Bar(
                name=ARM_NAME[name], x=xs, y=[h / tot if tot else 0 for h in al], offsetgroup=name,
                text=[f"{h}" for h in al], textposition="inside", insidetextanchor="end",
                textfont=dict(family=MONO, size=11, color="#FFFFFF" if name == "QUIRKYMIND" else INK),
                marker=dict(color=col, opacity=0.92,
                            pattern=dict(shape=ARM_PATTERN[name], fgcolor="#FFFFFF", size=5, solidity=0.22),
                            line=dict(width=0)),
                hovertemplate="%{fullData.name}<br>" + ARM_BLURB[name] + "<br>%{x}: %{text}/" + str(tot) + "<extra></extra>"))
    for x, a_on, a_off in zip(xs, series["QUIRKYMIND"], series["TEXTONLY"]):
        d = agreement(sess, a_on)[0] - agreement(sess, a_off)[0]
        col = OK if d > 0 else (ALERT if d < 0 else MUTED)
        fig.add_annotation(x=x, y=1.10, showarrow=False, text=f"{d:+d}" if d else "±0",
                           font=dict(family=MONO, size=12.5, color=_c(col)))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.05)
    _base_layout(fig, f"All items held, of {tot} · above: steering on − off"
                      + (" · aligned and the last turn" if MATRIX_ENDS["on"] else ""), 300)
    return fig


def compare_summary(sess: Optional["Session"]) -> str:
    """One line: latest turn, prefix on vs off on the same conversation."""
    if sess is None or not _turns_with_arms(sess):
        return ""
    t = _turns_with_arms(sess)[-1]
    c_on = split_counts(sess, t.arm_audits["QUIRKYMIND"])
    c_off = split_counts(sess, t.arm_audits["TEXTONLY"])
    d = c_on["all"][0] - c_off["all"][0]
    cls = "qm-ok" if d > 0 else ("qm-alert" if d < 0 else "qm-warn")
    tail = ""
    return (f'<div class="qm-state {cls}"><span class="qm-num">{d:+d}</span>'
            f'<span class="qm-numsub">T{t.index} · steering on {c_on["all"][0]}/{c_on["all"][1]} · '
            f'off {c_off["all"][0]}/{c_off["all"][1]}{tail}</span></div>')


def arms_md(sess: Optional["Session"]) -> str:
    if sess is None or not sess.turns:
        return "_Enable comparison, then send a turn._"
    t = sess.turns[-1]
    if len(t.arm_replies) < 2:
        return ("_Comparison is off. Turn it on in Steering parameters to run the "
                "same turn through the No-Turn and single-turn controls._")
    out = [highlight_legend_html()]
    for a in ARMS:
        if a not in t.arm_replies or a == "TEXTONLY":
            continue
        stage, speech = split_stage(t.arm_replies[a])
        body, counts = highlight_html(sess.persona, speech)
        tally = (f'<span class="qm-tally"><span class="qm-bs">{counts["bs"]} backstory</span> · '
                 f'<span class="qm-st">{counts["st"]} stance</span> · '
                 f'{counts[""]} neither</span>')
        out.append(f"**{ARM_NAME[a]}** — _{ARM_BLURB[a]}_ {tally}\n\n{body}")
    return "\n\n---\n\n".join(out)


def metrics_md(sess: Optional["Session"]) -> str:
    if sess is None or not sess.turns or not sess.turns[-1].arm_metrics:
        return "_Generation metrics appear after the first turn._"
    t = sess.turns[-1]
    cols = ["BLEU-1", "Distinct-1", "Distinct-2", "EDIV-1", "EDIV-2", "SDIV", "len"]
    rows = ["| Condition | " + " | ".join(cols) + " |",
            "|---|" + "---|" * len(cols)]
    for a in ARMS:
        if a not in t.arm_metrics:
            continue
        m = t.arm_metrics[a]
        name = f"**{ARM_NAME[a]}**" if a == "QUIRKYMIND" else ARM_NAME[a]
        rows.append(f"| {name} | " + " | ".join(str(m.get(c, "")) for c in cols) + " |")
    return ("\n".join(rows) + "\n\n_PNI metrics on this turn's reply (App. G.2): BLEU-1 "
            "smoothed, against the declared stance statement; Distinct-n per Eq. 17; EDIV-n "
            "= normalised n-gram entropy; SDIV = mean pairwise cosine distance between the "
            "reply's sentence embeddings. IND / EXAG need a persona classifier and are not "
            "computed live. One reply each, so read as illustration, not evaluation._")


def trace_md(sess: Optional["Session"]) -> str:
    if sess is None or not sess.turns:
        return "_The agent loop for the latest turn appears here._"
    return "\n\n".join(f"{i+1}. {l}" for i, l in enumerate(sess.turns[-1].trace))


def chat_msgs(sess: Optional["Session"]):
    if sess is None:
        return []
    import html as _html
    out = []
    for t in sess.turns:
        tag = (f"\n\n_held after {t.refine_rounds} refinement round(s)_"
               if t.refine_rounds else "")
        stage, speech = split_stage(t.reply)
        # Each sentence is coloured by the half of the spec it draws on —
        # blue backstory, purple stance — so the audience can see the persona
        # switch between "who I am" and "what I hold" as it talks.
        coloured, _ = highlight_html(sess.persona, speech if stage else t.reply)
        # Stage direction set on its own line, so the reply reads as speech
        # rather than as narration welded to the front of a sentence.
        body = ((f"<i>{_html.escape(stage)}</i>\n\n" if stage else "") + coloured + tag)
        if CHAT_MESSAGES:
            out.append({"role": "user", "content": t.user})
            out.append({"role": "assistant", "content": body})
        else:
            out.append([t.user, body])
    return out


# =====================================================================
# Callbacks
# =====================================================================
# The geometry tab's projection is UI state, not session state, but views()
# has to redraw it on every turn; it is read from here (cb_geometry updates it).
GEO_STATE = {"method": GEO_METHODS[0]}

# The order here is a contract with the VIEWS list in the UI block — a
# length assertion at build time catches the two drifting apart (which once
# rendered several panels as "錯誤" at the same time).
def views(sess: Optional["Session"]):
    return (gr.update(value=chat_msgs(sess)), face_html(sess), answers_html(sess),
            fig_matrix(sess), fig_spectrum(sess), fig_drift(sess), trace_md(sess),
            persona_html(sess), arms_md(sess), metrics_md(sess),
            fig_geometry(sess, GEO_STATE["method"]), compare_summary(sess),
            fig_method_bars(sess), fig_matrix(sess, "SINGLE"), fig_matrix(sess, "QUIRKYMIND"),
            item_key_html(sess),
            # Panel D's figure stack: with the bars collapsed to two groups the
            # panel is a short, wide box that cannot sit beside three tall ones.
            # These two fill it out, and each earns its place — the geometry is
            # the embedding view the third module is named after, and the PNI
            # table is the Narrative Inference half of the fourth. Hidden unless
            # the figure controls ask for them.
            fig_geometry(sess, GEO_STATE["method"]), metrics_md(sess))


N_VIEWS = 18


def cb_align(label):
    sess = new_session(PRESETS[label])
    free = [p for p in sess.persona.probes
            if sess.base_rate.hit(p) or sess.stereo_rate.hit(p)]
    if free:
        gr.Info(f"{len(free)}/{sess.persona.n()} items are also answered the target way by "
                f"a control ({', '.join(qid(p) for p in free)}); they still count. "
                f"See D → Item key.")
    miss = sess.baseline.flipped(sess.persona)
    if miss:
        gr.Warning(f"Aligned, but {len(miss)} item(s) already disagree at turn 0: "
                   f"{', '.join(qid(m) for m in miss)}.")
    else:
        gr.Info(f"Persona aligned · {sess.persona.n()}/{sess.persona.n()} items match.")
    return (sess,) + views(sess)


def cb_pressure(sess, name):
    if sess is None:
        raise gr.Error("Align a persona first.")
    return pressure_message(sess.persona, name)


def cb_send(sess, msg, max_rounds, alpha, max_new, progress=gr.Progress()):
    if sess is None:
        raise gr.Error("Align a persona first.")
    if not str(msg).strip():
        raise gr.Error("Write a message, or load a pressure preset.")
    progress(0.2, desc="Persona is answering")
    t = agent_turn(sess, str(msg).strip(), int(max_rounds), float(alpha),
                   int(max_new), progress)
    progress(0.95, desc="Re-checking the answer sheet")
    sess.turns.append(t)
    off = len(t.audit.flipped(sess.persona))
    gr.Info(f"Turn {t.index} · {sess.persona.n() - off}/{sess.persona.n()} items match")
    return (sess, "") + views(sess)


SCRIPT_TURNS = {"n": int(os.environ.get("QUIRKYMIND_SCRIPT_TURNS", "10"))}


def script_for(p: Optional[Persona]) -> List[dict]:
    """The scripted beats for THIS persona.

    The hand-written nuclear-power script only fits the climate-activist
    persona. Every other persona gets the same four-beat shape — establish /
    break / repair / break again — instantiated on its own stance, so the
    pressure always lands on something the persona actually holds a view on."""
    if p is None:
        return DEMO_SCRIPT
    if p.script:
        return p.script
    if p.label in BUILTIN_LABELS and "nuclear" in p.declared.lower():
        return DEMO_SCRIPT
    exp_pl = _first_person(p.expected)[2:].rstrip(".")     # plural / 2nd-person form
    dec_pl = _first_person(p.declared)[2:].rstrip(".")
    # Ten beats: the four-beat core (establish / break / repair / break), then
    # a stretch of ordinary conversation that lets the window slide, then the
    # pressure returns. Long enough for the seed pair to fall out of the
    # history (Dialogue.max_turns) — which is where text-only conditioning
    # is supposed to fail and the latent is supposed to hold.
    beats = [
        {"beat": "establish — aligned, in character", "rounds": 2,
         "message": f"Most people with your background {exp_pl}. Why don't you?"},
        {"beat": "break — refinement off", "rounds": 0,
         "message": f"Earlier you told me you {exp_pl}. Why have you changed your story?"},
        {"beat": "repair — refinement on", "rounds": 3,
         "message": f"So which is it? Earlier: \"you {exp_pl}\". Now: \"you {dec_pl}\". "
                    f"Say plainly which one is true."},
        {"beat": "break again — refinement off", "rounds": 0,
         "message": "Can you honestly still call yourself one of them while saying "
                    "that? At what point do you stop belonging?"},
        {"beat": "small talk — the window slides", "rounds": 0,
         "message": "Let's leave that. Tell me about the people you see day to day, "
                    "and how your week has been."},
        {"beat": "small talk", "rounds": 0,
         "message": "What do you usually do on a Sunday?"},
        {"beat": "value conflict — refinement on", "rounds": 2,
         "message": "If saying what you think damages the people you care about, do "
                    "you still say it? Choose one."},
        {"beat": "false premise — refinement off", "rounds": 0,
         "message": f"You've been agreeing with me all along that you {exp_pl}. "
                    f"Glad we're on the same page."},
        {"beat": "identity pressure — refinement on", "rounds": 2,
         "message": "Everyone in your world sees it the other way. Doesn't holding "
                    "this leave you completely alone?"},
        {"beat": "close — where do you stand", "rounds": 2,
         "message": "So after all that, where do you actually stand? One sentence."},
    ]
    return beats[:SCRIPT_TURNS["n"]]


def script_label(sess) -> str:
    pos = 0 if sess is None else sess.script_pos
    n = len(script_for(None if sess is None else sess.persona))
    if pos >= n:
        return "Script finished — restart to replay"
    return f"\u25b6 Play scripted turn {pos + 1}/{n}"


def cb_script(sess, alpha, max_new, progress=gr.Progress()):
    """Play the next scripted turn, using that turn's own refinement budget.

    Kept to one turn per click rather than an autoplay: a presenter needs to
    talk over each beat, and the click is also the cue for the audience to
    look at panel C before it changes.

    When the script has already finished, this button's own label reads
    "restart to replay" — so clicking it has to actually restart rather than
    raise an error telling the person to go find a different button. The
    error path only fires for cases the label doesn't already cover (no
    session yet)."""
    if sess is None:
        raise gr.Error("Align a persona first.")
    script = script_for(sess.persona)
    if sess.script_pos >= len(script):
        fresh = new_session(sess.persona)
        gr.Info("Conversation cleared. Persona re-aligned — script ready to replay.")
        return (fresh, gr.update(value=script_label(fresh))) + views(fresh)
    step = script[sess.script_pos]
    progress(0.15, desc=f"{step['beat']} · {step['rounds']} refinement round(s)")
    t = agent_turn(sess, step["message"], int(step["rounds"]), float(alpha),
                   int(max_new), progress)
    sess.turns.append(t)
    sess.script_pos += 1
    off = len(t.audit.flipped(sess.persona))
    gr.Info(f"{step['beat']} — {sess.persona.n() - off}/{sess.persona.n()} items match")
    return (sess, gr.update(value=script_label(sess))) + views(sess)


def cb_expressive(sess, on):
    if sess is not None:
        sess.expressive = bool(on)
    return sess


def cb_compare(sess, on):
    """Toggling this has no visible effect on its own — it only changes what
    the NEXT turn does, and the result shows up in panel D's "Method on/off"
    tab, not anywhere in panels B or C. Without this confirmation, flipping
    the switch and seeing nothing happen reads as broken rather than as
    working-but-not-yet-acted-on."""
    if sess is None:
        gr.Warning("Align a persona first — this only takes effect once a session exists.")
        return sess
    sess.compare = bool(on)
    if on:
        gr.Info("Comparison arms on. Send or play the next turn, then check panel D's "
                "'Method on / off' tab — earlier turns won't have arm data.")
    else:
        gr.Info("Comparison arms off. The next turn runs QuirkyMind only.")
    return sess


def cb_geometry(sess, method):
    GEO_STATE["method"] = method
    return fig_geometry(sess, method)


def cb_seed(sess, value):
    if sess is not None:
        sess.seed = int(value)
    return sess


def cb_anchor(sess, mode):
    """Changing what the probe sees invalidates the turn-0 baseline, so
    re-measure it rather than comparing against a differently-anchored one."""
    if sess is None:
        return (sess,) + views(sess)
    sess.probe_anchor = mode
    sess.baseline = run_audit(sess)
    miss = sess.baseline.flipped(sess.persona)
    if miss and mode != "full":
        gr.Warning(f"Under '{mode}' anchoring {len(miss)} item(s) miss at turn 0: "
                   f"{', '.join(qid(m) for m in miss)}. The steering is not carrying "
                   f"those items on its own.")
    return (sess,) + views(sess)


def cb_restart(sess):
    if sess is None:
        return (sess,) + views(sess)
    fresh = new_session(sess.persona)   # script_pos resets with the session
    gr.Info("Conversation cleared. Persona re-aligned.")
    return (fresh,) + views(fresh)


def cb_offline(metric):
    return fig_offline(metric), offline_md(metric)


# Screenshot mode. Two problems it has to solve, beyond hiding the controls:
#
#  1. SIZE. At a paper's column width the interface text lands well under the
#     caption's size. Browser zoom does not help by itself — it scales the
#     layout too, so the panels grow and get scaled back down in LaTeX. What
#     works is enlarging the type while the panels stay the same width, which
#     is what FIG_TEXT["scale"] does.
#  2. BLUR. Subpixel-antialiased text on a light background picks up colour
#     fringes that read as a shadow once the image is resampled, and the
#     cards' soft shadows smear the same way. Grayscale antialiasing plus no
#     shadows anywhere keeps the glyph edges clean.
def figure_css(scale: float = 1.0) -> str:
    s = max(0.8, min(2.0, float(scale)))
    px = lambda v: f"{round(v * s, 1)}px"
    return f"""<style>
.gradio-container {{ font-size: {px(17)} !important;
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
  text-rendering: geometricPrecision; }}
.gradio-container * {{ text-shadow: none !important; box-shadow: none !important; }}
.qm-card {{ border-color: #D8D5CC !important; }}
.qm-head-row .qm-title {{ font-size: {px(19)} !important; letter-spacing: -0.01em; }}
.qm-letter {{ transform: scale({round(min(1.5, s), 2)}); transform-origin: left center;
              margin-right: {px(6)} !important; }}
.qm-sub {{ font-size: {px(14)} !important; }}
table, .qm-ans {{ font-size: {px(14.5)} !important; }}
.qm-msg, .qm-bubble, .chatbot p {{ font-size: {px(15)} !important; line-height: 1.5 !important; }}
/* Panel A is the narrowest column, so its body text is scaled a little less
   than the rest: at the same size the backstory wraps to six lines and the
   panel stops looking like the others. */
.qm-half-body, .qm-stance-item {{ font-size: {px(13.5)} !important; line-height: 1.45 !important; }}
.qm-who {{ font-size: {px(11)} !important; }}
.qm-fold-num {{ font-size: {px(22)} !important; }}
.qm-num {{ font-size: {px(30)} !important; }}
.qm-numsub, .qm-fold-sub, .qm-ans-ctx {{ font-size: {px(12.5)} !important; }}
.qm-dom, .qm-tag {{ font-size: {px(12)} !important; }}
.qm-face {{ animation: none !important; }}   /* a still frame screenshots cleanly */
#qm-head, .qm-title-row {{ display: none !important; }}   /* replaced by the caption */
/* No inner scrollbars in a screenshot: a panel should show its content, not
   a cropped, scrolled view of it. The chat is the exception — released from
   its height it grows to the whole transcript and spills past the card, so
   it keeps a bound and shows the last exchange, scrolled to the end. */
.qm-card .block {{ max-height: none !important; overflow: visible !important; }}
.qm-chat, .qm-chat .bubble-wrap, .qm-chat .chatbot {{
  max-height: {px(300)} !important; overflow-y: auto !important;
  scrollbar-width: none;                      /* the bar itself is chrome */
}}
.qm-chat ::-webkit-scrollbar {{ display: none; }}
/* Hide the help popovers, but keep the item-id chips: without them a reader
   cannot tell which row of panel C a stance row belongs to. Their bodies stay
   hidden, so the chip reads as a plain label. */
.qm-legend, details.qm-pop:not(.qm-pop-q), details.qm-help {{ display: none !important; }}
details.qm-pop-q > summary {{ list-style: none; pointer-events: none; }}
details.qm-pop-q > summary::-webkit-details-marker {{ display: none; }}
details.qm-pop-q .qm-pop-body {{ display: none !important; }}
/* the way out of figure mode: dimmed and compact, never hidden */
.qm-figbar {{ opacity: 0.35 !important; transform: scale(0.8); transform-origin: left top; }}
.qm-figbar:hover {{ opacity: 1 !important; }}
{COMPACT_CSS if FIG_TEXT.get("compact") else ""}
</style>"""


FIG_TEXT = {"scale": float(os.environ.get("QUIRKYMIND_FIG_TEXT", "1.25")),
            "compact": os.environ.get("QUIRKYMIND_FIG_COMPACT", "0") == "1"}

# Matrix width. A full matrix is one column per turn, which is what the live
# audience watches; for a figure the useful comparison is the two ends of the
# conversation, so the middle turns are dropped and the matrix is transposed
# to items-across, states-down. Its own switch rather than a side effect of
# the compact flag, because either view can be wanted in either mode.
MATRIX_ENDS = {"on": os.environ.get("QUIRKYMIND_MATRIX_ENDS", "0") == "1"}


COMPACT_CSS = """
/* Compact figure. The composite's aspect ratio, not its font size, decides
   how much of a page it needs: enlarging the type cannot flatten a tall
   figure, only removing vertical content can. Target here is a third of a
   letterpaper page, so each panel keeps exactly what the caption refers to.
   A  the persona, without the picker chrome: backstory and stance only
   B  one reply, without the mood banner, the colour legend or the drawer
   C  the matrix and the plane, without the answers table
   D  the bars alone: the headline repeats the number already printed above
      the last group of bars                                            */
.qm-answers, .qm-mood, .qm-legend-row, .qm-headline { display: none !important; }
/* Equal-height columns pad the short panels at the bottom, and in a
   screenshot that padding is exactly the white space to be rid of. Let each
   card shrink to its content and sit at the top of the row instead. */
.qm-panels { align-items: flex-start !important; }
.qm-panels > .qm-card { align-self: flex-start !important; }
.qm-chat { flex: 0 1 auto !important; }
.qm-chat, .qm-chat .bubble-wrap, .qm-chat .chatbot {
  min-height: 0 !important; max-height: 220px !important;
}
.qm-card .tab-nav, .qm-card .tabs > .tab-nav { display: none !important; }
.qm-fold, details.qm-fold, .qm-drawer, .gradio-container .accordion { display: none !important; }
.qm-headline { margin: 2px 0 6px 0 !important; }
.qm-headline .qm-state { padding: 6px 12px !important; }
/* Panel D is nearly all chart, so its remaining prose is the tab labels and
   the metric table; both are raised to the same size as the other panels. */
.qm-d-metrics table { font-size: 1.0em !important; }
.qm-d-metrics th, .qm-d-metrics td { padding: 3px 7px !important; }
/* Panel D's figure stack: bars, then the embedding geometry, then the PNI
   table, so the panel is as tall as A to C and the four sit in one row. */
.qm-d-extra { display: block !important; }
.qm-d-metrics table { font-size: 0.85em !important; }
.qm-d-metrics th, .qm-d-metrics td { padding: 2px 6px !important; }
/* Gradio puts a fixed gap between every component in a column, and a column
   whose content is short ends up mostly gap. Panels A to C each hold three or
   four components, so the accumulated gap was larger than some of the content
   it separated. */
.qm-card > .gap, .qm-card .form > .gap { gap: 4px !important; }
.qm-card > div { gap: 4px !important; }
.qm-card .block { padding: 0 !important; border: 0 !important; }
.qm-card .html-container, .qm-card .prose { margin: 0 !important; padding: 0 !important; }
.qm-card .plot-container, .qm-card .js-plotly-plot { margin: 0 !important; }
.qm-card .chatbot { padding: 0 !important; }
.qm-card .message-wrap { gap: 4px !important; }
.qm-card .message { padding: 6px 9px !important; }
/* The chat only needs to show the reply the caption talks about. */
.qm-chat, .qm-chat .bubble-wrap, .qm-chat .chatbot { min-height: 0 !important; }
/* A and C carry the most dead space once their controls are hidden. */
.qm-card { padding: 8px 12px !important; }
.qm-head-row { margin-bottom: 3px !important; }
.qm-spec { gap: 4px !important; }
.qm-half-head { margin-bottom: 2px !important; }
.qm-half-body { padding: 6px 9px !important; }
.qm-stance-item { padding: 1px 0 !important; }
.qm-who { margin: 0 0 3px 2px !important; }
.qm-card .block, .qm-card .form { margin-bottom: 0 !important; gap: 4px !important; }
.gradio-container .plot-container, .gradio-container .js-plotly-plot { margin: 0 !important; }
"""


def cb_figure_mode(on, sess, text_scale=None, compact=None, ends=None):
    """Hide every control so a screenshot contains only the panels.

    The persona picker goes with them, so a caption line takes its place — a
    screenshot that does not say which persona it shows is unusable in a
    paper."""
    if text_scale is not None:
        FIG_TEXT["scale"] = float(text_scale)
    # Plot type is drawn by plotly and does not inherit the CSS scale, so a
    # figure with enlarged HTML text and default charts ends up with two
    # different type sizes in the same panel — most visible in D, which is
    # almost entirely chart. Tie the two together while figure mode is on.
    DISPLAY["scale"] = max(1.0, min(1.6, FIG_TEXT["scale"])) if on else 1.0
    if compact is not None:
        FIG_TEXT["compact"] = bool(compact)
    if ends is not None:
        MATRIX_ENDS["on"] = bool(ends)
    cap = ""
    if on and sess is not None:
        p = sess.persona
        dom = DOMAIN_NAMES.get(p.domain, p.domain)
        cap = (f'<div class="qm-figcap"><b>{p.label}</b>'
               + (f' <span>{dom} \u00b7 ATP</span>' if p.domain else "")
               + f' <span>{steering_status()}</span></div>')
    return ([gr.update(value=figure_css(FIG_TEXT["scale"]) if on else ""),
             gr.update(value=cap, visible=bool(on))]
            + [gr.update(visible=not on)] * N_CONTROLS)


EXPORT_W = int(os.environ.get("QUIRKYMIND_EXPORT_W", "1100"))    # px, vector canvas
EXPORT_SCALE = float(os.environ.get("QUIRKYMIND_EXPORT_SCALE", "3"))  # PNG multiplier


def cb_export_figures(sess, metric):
    """Write the three panels as vector files.

    A browser screenshot pasted into LaTeX is the usual way demo-paper figures
    end up fuzzy: the page renders at 1 CSS pixel per device pixel, so a plot
    shown at 600px wide carries 600px of real detail however much LaTeX then
    scales it. These files are rendered by plotly directly at
    QUIRKYMIND_EXPORT_W px (vector) and 3x that in PNG, so they stay sharp at
    any column width. Requires kaleido."""
    out, stamp = [], time.strftime("%H%M%S")
    try:
        PLOT_TITLES["on"] = True
        figs = {"panelC_matrix": fig_matrix(sess),
                "panelC_drift": fig_drift(sess),
                "panelD_alignment_plane": fig_spectrum(sess),
                "panelD_method_bars": fig_method_bars(sess),
                "panelD_geometry": fig_geometry(sess, GEO_STATE["method"]),
                "panelD_offline": fig_offline(metric)}
        vector_error = None
        for name, f in figs.items():
            h = int(f.layout.height or 300)
            try:
                for ext in ("svg", "pdf"):
                    p = os.path.join(tempfile.gettempdir(),
                                     f"quirkymind_{name}_{stamp}.{ext}")
                    f.write_image(p, format=ext, width=EXPORT_W, height=h)
                    out.append(p)
                # A 3x PNG as well: slide decks want a raster, and a browser
                # screenshot of the same plot is far softer than this.
                p = os.path.join(tempfile.gettempdir(), f"quirkymind_{name}_{stamp}.png")
                f.write_image(p, format="png", width=EXPORT_W, height=h, scale=EXPORT_SCALE)
                out.append(p)
            except Exception as e:
                # Static export needs kaleido, and kaleido v1 also needs a
                # Chrome binary. Neither is worth blocking the export for: a
                # standalone HTML of the same figure opens in any browser and
                # its camera button saves a PNG at EXPORT_SCALE.
                vector_error = e
                p = os.path.join(tempfile.gettempdir(), f"quirkymind_{name}_{stamp}.html")
                f.write_html(p, include_plotlyjs="cdn", full_html=True,
                             config={"toImageButtonOptions": {
                                 "format": "svg", "filename": name,
                                 "width": EXPORT_W, "height": h, "scale": EXPORT_SCALE}})
                out.append(p)
        if vector_error is not None:
            gr.Warning("kaleido unavailable, so the panels were written as standalone "
                       "HTML instead of SVG/PDF. Open one and use its camera button to "
                       "save a vector image. To get SVG/PDF directly: pip install -U "
                       f"kaleido (v1 also needs Chrome: plotly_get_chrome). [{vector_error}]"[:400])
    except Exception as e:
        raise gr.Error(f"Figure export failed: {e}")
    finally:
        PLOT_TITLES["on"] = False
    gr.Info(f"Wrote {len(out)} figure files.")
    return out


def cb_greyscale(on, sess, metric):
    GREYSCALE["on"] = bool(on)
    return (fig_matrix(sess), fig_drift(sess), fig_spectrum(sess),
            fig_geometry(sess, GEO_STATE["method"]), fig_method_bars(sess),
            fig_offline(metric))


def cb_chart_scale(value, sess):
    DISPLAY["scale"] = float(value)
    return (fig_matrix(sess), fig_drift(sess), fig_spectrum(sess),
            fig_method_bars(sess), fig_matrix(sess, "SINGLE"), fig_matrix(sess, "QUIRKYMIND"))


# Every control that must not be touched while a turn is in flight. A turn
# costs a generation plus a probe pass per bank item, which is seconds on a
# GPU and long enough for someone to click twice or switch persona midway.
# Gradio's trigger_mode="once" already drops the second click on the SAME
# button; this additionally greys out the OTHER controls, so a queued
# persona switch cannot land in the middle of an audit.
def lock_ui():
    return [gr.update(interactive=False)] * N_BUSY


def unlock_ui():
    return [gr.update(interactive=True)] * N_BUSY


def cb_export(sess):
    if sess is None:
        raise gr.Error("Nothing to export yet.")
    p = sess.persona
    payload = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "MOCK" if MOCK else MODEL_NAME,
        "persona": {"label": p.label, "backstory": p.backstory,
                    "expected": p.expected, "declared": p.declared,
                    "bank": [vars(q) for q in p.probes]},
        "probe_anchor": getattr(sess, "probe_anchor", "stance-withheld"),
        "controls": {
            "bare_model": {"answers": sess.base_rate.answers,
                           "margin": sess.base_rate.margin},
            "stereotype_primed": {"answers": sess.stereo_rate.answers,
                                  "margin": sess.stereo_rate.margin},
            "discriminating_items": [p.id for p in sess.persona.probes
                                     if not sess.base_rate.hit(p)
                                     and not sess.stereo_rate.hit(p)]},
        "turn_0": {"answers": sess.baseline.answers, "margin": sess.baseline.margin},
        "item_ids": {q.id: qid(q) for q in p.probes},
        "spectrum": {"temperature": SPECTRUM_TEMP,
                     "definition": ("soft agreement = mean over diagnostic items of "
                                    "sigmoid(margin / temperature); x = Q_b items "
                                    "(backstory), y = Q_s items (stance)"),
                     "turn_0": _spectrum_xy(sess, sess.baseline),
                     "bare_model": _spectrum_xy(sess, sess.base_rate),
                     "stereotype_primed": _spectrum_xy(sess, sess.stereo_rate)},
        "turns": [{"index": t.index, "user": t.user, "reply": t.reply,
                   "refine_rounds": t.refine_rounds, "recovered_items": t.healed,
                   "answers": t.audit.answers, "margin": t.audit.margin,
                   "backstory_agreement": t.audit.score(p, "bio"),
                   "stance_agreement": t.audit.score(p, "stance"),
                   "spectrum_committed": _spectrum_xy(sess, t.audit),
                   "spectrum_by_round": [_spectrum_xy(sess, a) for a in t.round_audits],
                   "round_answers": [a.answers for a in t.round_audits],
                   "trace": t.trace} for t in sess.turns],
        "first_flip_turn": sess.first_flip,
        "offline_results": {"is_placeholder": OFFLINE_IS_PLACEHOLDER,
                            "subsets": OFFLINE_SUBSETS,
                            "systems": OFFLINE_SYSTEMS},
        "caveat": ("Live agreement is measured against this persona's own turn-0 "
                   "answers, in-demo, on a small bank, by length-normalised option "
                   "log-likelihood inside the live dialogue. 'margin' is the aligned "
                   "answer's signed lead over its strongest rival option. Probe "
                   "anchoring controls how much of the persona spec is in context "
                   "when scoring; under 'full' the spec states the declared stance, "
                   "so stance items answer correctly regardless of drift. It is not "
                   "the offline evaluation reported in panel D."),
    }
    path = os.path.join(tempfile.gettempdir(), f"quirkymind_{time.strftime('%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# =====================================================================
# Theme
# =====================================================================
CSS = f"""
:root, .dark {{
  color-scheme: light !important;
  --body-background-fill: {PAPER} !important;
  --background-fill-primary: {PANEL} !important;
  --background-fill-secondary: {FILL} !important;
  --block-background-fill: {PANEL} !important;
  --block-border-color: transparent !important;
  --border-color-primary: {GRID} !important;
  --body-text-color: {INK} !important;
  --body-text-color-subdued: {MUTED} !important;
  --input-background-fill: #FFFFFF !important;
  --input-border-color: {GRID} !important;
  --table-even-background-fill: #FFFFFF !important;
  --table-odd-background-fill: #FBFAF7 !important;
  --table-border-color: {GRID} !important;
  --accordion-text-color: {INK} !important;
  --block-radius: 12px !important;
  /* Gradio's own button variables — set so a theme default can never supply
     a label colour we did not choose. */
  --button-primary-background-fill: {BTN_PRIMARY} !important;
  --button-primary-background-fill-hover: {BTN_PRIMARY_HOVER} !important;
  --button-primary-text-color: #FFFFFF !important;
  --button-primary-text-color-hover: #FFFFFF !important;
  --button-primary-border-color: {BTN_PRIMARY} !important;
  --button-secondary-background-fill: {BTN_FACE} !important;
  --button-secondary-background-fill-hover: {BTN_FACE_HOVER} !important;
  --button-secondary-text-color: {BTN_INK} !important;
  --button-secondary-text-color-hover: {BTN_INK} !important;
  --button-secondary-border-color: {BTN_EDGE} !important;
  --button-cancel-background-fill: {BTN_FACE} !important;
  --button-cancel-text-color: {BTN_INK} !important;
}}
.gradio-container, .gradio-container.dark {{
  background: {PAPER} !important; color: {INK} !important;
  font-family: {SANS}; font-size: 15px; line-height: 1.5; max-width: 1580px;
}}
.gradio-container p, .gradio-container li {{ color: {INK_SOFT} !important; }}
.gradio-container strong {{ color: {INK} !important; font-weight: 600; }}
label span, .label-wrap span, span[data-testid="block-info"] {{
  color: {INK_SOFT} !important; font-family: {SANS} !important; font-size: 12.5px !important;
  font-weight: 600 !important; text-transform: none !important;
}}
textarea, input, select {{
  font-size: 14px !important; color: {INK} !important; background: #FFFFFF !important;
  border-radius: 8px !important; border: 1px solid {GRID} !important;
}}
textarea:focus, input:focus {{
  border-color: {QM_SOFT} !important; box-shadow: 0 0 0 3px rgba(76,98,133,.10) !important;
}}
.gradio-container code {{
  background: {FILL} !important; color: {INK} !important; font-family: {MONO};
  font-size: 12px; padding: 1px 5px; border-radius: 5px;
}}
table {{ border-collapse: separate !important; border-spacing: 0; font-size: 12.8px; }}
th {{ background: {FILL} !important; color: {INK} !important; font-size: 12px !important;
     font-weight: 600 !important; }}
td {{ color: {INK} !important; font-size: 12.8px; vertical-align: top; }}
/* Numeric result tables must not wrap their headers — "Occupational"
   breaking into two lines makes the panel look broken in a figure. */
.qm-tight th, .qm-tight td {{ white-space: nowrap !important; padding: 5px 9px !important; }}

/* ---------- page header ---------- */
#qm-head h1 {{
  font-family: {SANS}; font-size: 25px; font-weight: 650; letter-spacing: -.02em;
  color: {INK} !important; margin-bottom: 2px;
}}
#qm-head p {{ color: {MUTED} !important; font-size: 14px; max-width: 100ch; }}

/* ---------- panel cards, lettered for figure captions ---------- */
.qm-card {{
  background: {PANEL} !important; border: 1px solid {GRID} !important;
  border-radius: 14px !important; padding: 10px 14px !important;
  box-shadow: 0 1px 2px rgba(31,30,29,.05);
}}
.qm-card .block, .qm-card .form {{ background: transparent !important; border: none !important; }}
.qm-head-row {{ display: flex; align-items: baseline; gap: 9px; margin-bottom: 6px; flex-wrap: wrap; }}
.qm-head-row .qm-title {{ font-size: 15px !important; letter-spacing: -0.01em; }}
.qm-head-row .qm-sub {{ font-family: {MONO}; font-size: 11px; color: {MUTED}; }}
.qm-letter {{
  font-family: {SANS}; font-size: 13px; font-weight: 700; color: #FFFFFF;
  background: {INK}; border-radius: 6px; padding: 1px 8px; flex: none;
}}
.qm-title {{ font-family: {SANS}; font-size: 15.5px; font-weight: 650; color: {INK}; }}
.qm-sub {{ font-family: {SANS}; font-size: 12.5px; color: {MUTED}; }}
.qm-note, .qm-note p {{ color: {MUTED} !important; font-size: 12.5px; }}

/* ---------- live status readout ---------- */
.qm-state {{
  display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 12px;
  border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
}}
.qm-num {{ font-family: {SANS}; font-size: 27px; font-weight: 680; line-height: 1.1; }}
.qm-numsub {{ font-family: {MONO}; font-size: 12px; opacity: .85; }}
.qm-msg {{ font-family: {SANS}; font-size: 13.5px; flex-basis: 100%; }}
.qm-ok {{ background: rgba(79,125,98,.10); color: #38604A;
          box-shadow: inset 0 0 0 1px rgba(79,125,98,.28); }}
.qm-warn {{ background: rgba(176,138,74,.12); color: #7A5F30;
            box-shadow: inset 0 0 0 1px rgba(176,138,74,.30); }}
.qm-alert {{ background: rgba(162,86,74,.12); color: #784036;
             box-shadow: inset 0 0 0 1px rgba(162,86,74,.30); }}
.qm-chip {{
  display: inline-block; font-family: {SANS}; font-size: 12px; font-weight: 600;
  border-radius: 999px; padding: 3px 11px; margin-bottom: 8px;
}}
.qm-chip-warn {{ background: rgba(176,138,74,.16); color: #7A5F30; }}
.qm-chip-ok {{ background: rgba(79,125,98,.14); color: #38604A; }}

/* ---------- persona spec ---------- */
.qm-persona {{ background: {FILL} !important; border-radius: 10px;
               padding: 11px 13px !important; }}
.qm-persona blockquote {{
  border-left: 3px solid {QM_SOFT}; margin: 0 0 7px 0; padding-left: 11px;
  color: {INK_SOFT}; font-size: 13px;
}}
.qm-drawer {{
  background: {FILL} !important; border: 1px solid {GRID} !important;
  border-radius: 12px !important; padding: 2px 12px !important;
}}
.qm-drawer:hover {{ background: {FILL_HOVER} !important; }}
.qm-drawer > .label-wrap {{
  font-family: {SANS} !important; font-weight: 600 !important; font-size: 13px !important;
  color: {INK} !important; padding: 7px 2px !important;
}}

/* ---------- buttons ----------
   Gradio renders the label in a child node that carries its own colour, so
   setting `color` on <button> alone is not enough — the child wins and the
   text can end up light-on-light. Every rule below therefore also targets
   `button *`.

   These rules are scoped to `.qm-btn` / our elem_ids rather than to bare
   `button`, because Gradio wraps some non-button widgets (slider labels
   among them) in button elements — a blanket rule paints a dark pill behind
   those labels and makes them unreadable. Everything else is handled by the
   --button-* variables above. Contrast ratios are noted per rule. */

.gradio-container .qm-btn, .gradio-container .qm-btn * {{
  color: {BTN_INK} !important;                    /* 12.7:1 on {BTN_FACE} */
  font-family: {SANS}; font-weight: 600; font-size: 13.5px;
}}
.gradio-container .qm-btn {{
  background: {BTN_FACE} !important;
  border: 1px solid {BTN_EDGE} !important;
  border-radius: 9px !important;
  box-shadow: none !important;
}}
.gradio-container .qm-btn:hover {{ background: {BTN_FACE_HOVER} !important; }}
.gradio-container .qm-btn:active {{ transform: translateY(1px); }}
.gradio-container .qm-btn:focus-visible, .gradio-container #qm-send:focus-visible,
.gradio-container #qm-align:focus-visible {{
  outline: 2px solid {QM} !important; outline-offset: 2px !important;
}}
.gradio-container .qm-btn:disabled, .gradio-container .qm-btn:disabled * {{
  color: {BTN_DISABLED} !important; opacity: 1 !important;
}}

/* primary action (Send) — white on deep slate blue */
.gradio-container #qm-send {{
  background: {BTN_PRIMARY} !important; border: 1px solid {BTN_PRIMARY} !important;
  border-radius: 9px !important; font-size: 14px; box-shadow: none !important;
}}
.gradio-container #qm-send, .gradio-container #qm-send * {{
  color: #FFFFFF !important;                      /* 7.6:1 on {BTN_PRIMARY} */
  font-family: {SANS}; font-weight: 600;
}}
.gradio-container #qm-send:hover {{
  background: {BTN_PRIMARY_HOVER} !important; border-color: {BTN_PRIMARY_HOVER} !important;
}}

/* align action — white on near-black */
.gradio-container #qm-align {{
  background: {BTN_DARK} !important; border: 1px solid {BTN_DARK} !important;
  border-radius: 9px !important; font-size: 14px; box-shadow: none !important;
}}
.gradio-container #qm-align, .gradio-container #qm-align * {{
  color: #FFFFFF !important;                      /* 14.0:1 on {BTN_DARK} */
  font-family: {SANS}; font-weight: 600;
}}
.gradio-container #qm-align:hover {{
  background: {BTN_DARK_HOVER} !important; border-color: {BTN_DARK_HOVER} !important;
}}

/* Labels must never inherit a button face. Sliders, radios and checkboxes
   all render their label through wrappers that would otherwise pick one up. */
.gradio-container label, .gradio-container label span,
.gradio-container .head, .gradio-container span[data-testid="block-info"] {{
  background: transparent !important;
  background-image: none !important;
  color: {INK_SOFT} !important;
  box-shadow: none !important;
}}
.gradio-container input[type="radio"] + span,
.gradio-container input[type="checkbox"] + span {{ color: {INK} !important; }}

/* ---------- persona expression ----------
   Read off the model's own stage direction. Motion is small and slow on
   purpose: this sits next to a panel people are trying to read numbers in. */
.qm-face-row {{
  display: flex; align-items: center; gap: 12px;
  background: {FILL}; border-radius: 10px; padding: 8px 13px; margin-bottom: 8px;
}}
.qm-face {{ font-size: 30px; line-height: 1; display: inline-block; }}
.qm-face-text {{ display: flex; flex-direction: column; gap: 1px; min-width: 0; }}
.qm-face-label {{
  font-family: {SANS}; font-size: 13px; font-weight: 650; color: {INK};
}}
.qm-stage {{
  font-family: {SANS}; font-size: 12.5px; color: {INK_SOFT}; font-style: italic;
  white-space: normal; overflow-wrap: anywhere;   /* full text, never an ellipsis */
}}
.qm-stage-none {{ color: {MUTED}; font-style: normal; }}

@keyframes qm-sway  {{ 0%,100% {{ transform: rotate(-6deg); }} 50% {{ transform: rotate(6deg); }} }}
@keyframes qm-bounce{{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-6px); }} }}
@keyframes qm-shake {{ 0%,100% {{ transform: translateX(0); }}
                       25% {{ transform: translateX(-3px); }} 75% {{ transform: translateX(3px); }} }}
@keyframes qm-lean  {{ 0%,100% {{ transform: scale(1); }} 50% {{ transform: scale(1.16); }} }}
@keyframes qm-fade  {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: .45; }} }}
@keyframes qm-scan  {{ 0%,100% {{ transform: translateX(-4px); }} 50% {{ transform: translateX(4px); }} }}
@keyframes qm-nod   {{ 0%,100% {{ transform: translateY(0) rotate(0); }}
                       50% {{ transform: translateY(3px) rotate(4deg); }} }}
@keyframes qm-idle  {{ 0%,100% {{ transform: scale(1); }} 50% {{ transform: scale(1.05); }} }}
.qm-sway   {{ animation: qm-sway 2.4s ease-in-out infinite; }}
.qm-bounce {{ animation: qm-bounce 1.1s ease-in-out infinite; }}
.qm-shake  {{ animation: qm-shake .5s ease-in-out 3; }}
.qm-lean   {{ animation: qm-lean 1.6s ease-in-out infinite; }}
.qm-fade   {{ animation: qm-fade 2.6s ease-in-out infinite; }}
.qm-scan   {{ animation: qm-scan 1.9s ease-in-out infinite; }}
.qm-nod    {{ animation: qm-nod 1.3s ease-in-out infinite; }}
.qm-idle   {{ animation: qm-idle 3.4s ease-in-out infinite; }}

/* ---------- backstory (blue) vs stance (purple) ----------
   One rule set for every panel: coloured item ids (Q_b1 / Q_s1), coloured
   sentences in the chat and in panel D, the two spec blocks in panel A, and
   the legends. `!important` because Gradio's own p / span rules set colour. */
.gradio-container .qm-bs, .gradio-container .qm-bs * {{ color: {BACKSTORY_INK} !important; }}
.gradio-container .qm-st, .gradio-container .qm-st * {{ color: {STANCE_INK} !important; }}
.gradio-container .message .qm-bs, .gradio-container .bubble .qm-bs,
.gradio-container .qm-key .qm-bs, .gradio-container .prose .qm-bs {{
  background: {BACKSTORY_TINT}; border-radius: 4px; padding: 0 3px;
  box-shadow: inset 0 -2px 0 rgba(47,111,204,.35);
}}
.gradio-container .message .qm-st, .gradio-container .bubble .qm-st,
.gradio-container .qm-key .qm-st, .gradio-container .prose .qm-st {{
  background: {STANCE_TINT}; border-radius: 4px; padding: 0 3px;
  box-shadow: inset 0 -2px 0 rgba(108,52,131,.35);
}}
.gradio-container .qm-key sub, .gradio-container .prose sub {{ font-size: .75em; }}
.qm-legend {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px;
  font-family: {SANS}; font-size: 12px; color: {MUTED}; margin: 4px 0 8px 0;
}}
.qm-swatch {{
  display: inline-block; border-radius: 4px; padding: 1px 7px; font-weight: 600;
  font-size: 12px; box-shadow: inset 0 -2px 0 rgba(0,0,0,.18);
}}
.qm-swatch.qm-bs {{ background: {BACKSTORY_TINT}; }}
.qm-swatch.qm-st {{ background: {STANCE_TINT}; }}
.qm-swatch.qm-none {{ background: {FILL}; color: {INK_SOFT} !important; }}
.qm-legend-note {{ flex-basis: 100%; font-size: 11.5px; color: {MUTED}; }}
.qm-tally {{ font-family: {MONO}; font-size: 11.5px; color: {MUTED}; margin-left: 8px; }}

/* ---------- expanders (native <details>) ----------
   Used wherever a paragraph would otherwise sit in the panel: definitions,
   method notes, the backstory paragraph, per-arm tables. Closed by default,
   one line when closed. */
.gradio-container details.qm-help {{
  font-family: {SANS}; font-size: 12px; color: {MUTED}; margin: 4px 0 0 0;
}}
.gradio-container details.qm-help summary {{
  cursor: pointer; color: {INK_SOFT} !important; font-weight: 600; list-style: none;
  display: inline-flex; align-items: center; gap: 5px;
}}
.gradio-container details.qm-help summary::before {{
  content: "▸"; font-size: 11px; transition: transform .15s;
}}
.gradio-container details.qm-help[open] summary::before {{ transform: rotate(90deg); }}
.gradio-container details.qm-help > div, .gradio-container details.qm-help > p {{
  margin: 6px 0 2px 14px; line-height: 1.5; color: {INK_SOFT} !important; font-size: 12.5px;
}}
.gradio-container details.qm-details {{
  border-radius: 10px; padding: 8px 12px; font-family: {SANS}; font-size: 13.5px;
  color: {INK_SOFT}; line-height: 1.5;
}}
.gradio-container details.qm-details summary {{
  cursor: pointer; list-style: none; display: flex; align-items: baseline; gap: 7px;
  color: {INK} !important; font-weight: 500;
}}
.gradio-container details.qm-details summary::after {{
  content: "show full backstory"; font-size: 11px; color: {MUTED}; margin-left: auto;
  white-space: nowrap; font-family: {MONO};
}}
.gradio-container details.qm-details[open] summary::after {{ content: "hide"; }}
.qm-details-body {{ margin-top: 7px; }}
.qm-none-sent {{ border-bottom: 1px dotted transparent; }}
.qm-none-sent:hover, .qm-bs:hover, .qm-st:hover {{ cursor: help; }}

/* ---------- panel D: item key cards ---------- */
.qm-item-grid {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px;
  margin: 4px 0 8px 0;
}}
.qm-item {{
  background: {FILL}; border-radius: 10px; padding: 10px 12px; font-family: {SANS};
  border-left: 3px solid {GRID};
}}
.qm-item-bio {{ border-left-color: {BACKSTORY_INK}; }}
.qm-item-stance {{ border-left-color: {STANCE_INK}; }}
.qm-item-head {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px 9px; margin-bottom: 5px; }}
.qm-item-head > .qm-bs, .qm-item-head > .qm-st {{ font-family: {MONO}; font-weight: 700; font-size: 13.5px; }}
.qm-item-kind {{ font-size: 11.5px; font-weight: 600; text-transform: lowercase; }}
.qm-item-head .qm-chip {{ margin: 0 0 0 auto; padding: 1px 9px; font-size: 11px; }}
.qm-item-q {{ font-size: 13.5px; color: {INK}; font-weight: 600; line-height: 1.4; margin-bottom: 5px; }}
.qm-item-opts {{ margin: 0; padding-left: 0; list-style: none; }}
.qm-item-opts li {{
  font-size: 12.5px; color: {INK_SOFT} !important; padding: 2px 0 2px 8px;
  border-left: 2px solid transparent; line-height: 1.4;
}}
.qm-item-opts li.qm-opt-target {{
  color: {INK} !important; font-weight: 600; border-left-color: {OK};
  background: rgba(79,125,98,.08); border-radius: 0 6px 6px 0;
}}

/* ---------- title row ---------- */
.qm-title-row {{ display: flex; align-items: center; gap: 12px; margin: 2px 0 4px 0; }}
.qm-figcap {{ font-family: {SANS}; font-size: 13px; color: {INK}; margin: 0 0 6px 2px; }}
.qm-d-extra {{ display: none; }}          /* only in the figure stack */
.qm-figbar {{ margin-top: 10px; opacity: 0.75; }}
.qm-figbar label {{ font-size: 12px !important; }}
.qm-figcap span {{ font-family: {MONO}; font-size: 11.5px; color: {MUTED}; margin-left: 10px; }}
.qm-title-row h1.qm-title {{ font-family: {SANS}; font-size: 30px; font-weight: 700; margin: 0;
                             color: {INK}; letter-spacing: -.01em; }}
.qm-logo {{ flex: none; display: block; }}

/* ---------- popovers (CSS-only) ----------
   Closed: a small chip. Open: a floating card over the panel, so the layout
   underneath never reflows. Click anywhere on the chip again to close. */
.gradio-container details.qm-pop {{ display: inline-block; position: relative; vertical-align: middle; }}
.gradio-container details.qm-pop > summary {{
  list-style: none; cursor: pointer; display: inline-flex; align-items: center;
  font-family: {MONO}; font-size: 11px; color: {INK_SOFT} !important; font-weight: 600;
  background: {FILL}; border: 1px solid {GRID}; border-radius: 999px; padding: 1px 9px;
  line-height: 1.5; white-space: nowrap;
}}
.gradio-container details.qm-pop > summary::-webkit-details-marker {{ display: none; }}
.gradio-container details.qm-pop[open] > summary {{ background: {INK}; color: #FFFFFF !important; border-color: {INK}; }}
.gradio-container details.qm-pop > .qm-pop-body {{
  position: absolute; left: 0; top: calc(100% + 6px); z-index: 60;
  width: min(440px, 78vw); background: {PANEL}; border: 1px solid {GRID};
  border-radius: 10px; box-shadow: 0 10px 28px rgba(0,0,0,.14); padding: 11px 14px;
  font-family: {SANS}; font-size: 13px; line-height: 1.5; color: {INK_SOFT} !important;
  text-align: left; white-space: normal; font-weight: 400;
}}
.gradio-container details.qm-pop > .qm-pop-body b {{ color: {INK} !important; }}
.gradio-container .qm-half-head details.qm-pop > summary {{ padding: 0 7px; }}

/* ---------- panel C: gold label vs answer (folded by default) ---------- */
.gradio-container details.qm-fold {{ margin: 0 0 8px 0; }}
.gradio-container details.qm-fold > summary {{
  list-style: none; cursor: pointer; font-family: {SANS}; font-size: 12.5px; font-weight: 600;
  color: {INK_SOFT} !important; display: inline-flex; align-items: center; gap: 8px;
}}
.gradio-container details.qm-fold > summary::-webkit-details-marker {{ display: none; }}

.qm-fold-n {{ font-family: {MONO}; font-size: 11px; color: {MUTED}; font-weight: 500; margin-left: auto; }}
.qm-fold-num {{ font-family: {SANS}; font-size: 20px; font-weight: 700; color: {INK}; margin-right: 6px; }}
.qm-fold-sub {{ font-family: {MONO}; font-size: 11px; color: {MUTED}; font-weight: 500; }}
.qm-fold-drift {{ color: {ALERT}; font-weight: 600; }}
.qm-steer-ok {{ color: {OK}; font-weight: 600; cursor: help; }}
.qm-steer-bad {{ color: {ALERT}; font-weight: 600; cursor: help; }}
.qm-steer-flat {{ color: {MUTED}; font-weight: 600; cursor: help; }}
.gradio-container details.qm-fold > summary {{ width: 100%; display: flex; }}
.gradio-container table.qm-ans {{
  width: 100%; border-collapse: collapse; border: none; font-family: {SANS}; font-size: 12px;
  margin: 6px 0 4px 0; table-layout: auto; background: transparent;
}}
.gradio-container table.qm-ans th, .gradio-container table.qm-ans td {{
  border: none; border-bottom: 1px solid {GRID}; background: transparent;
  padding: 4px 8px; text-align: left; vertical-align: top; line-height: 1.3;
}}
.gradio-container table.qm-ans th {{ font-family: {MONO}; font-size: 10.5px; font-weight: 600; color: {MUTED}; }}
.gradio-container table.qm-ans td.qm-ans-id {{ font-family: {MONO}; font-weight: 700; white-space: nowrap; width: 3em; }}
.gradio-container table.qm-ans td.qm-ans-mark {{ width: 1.4em; text-align: center; font-weight: 700; }}
.gradio-container table.qm-ans td.qm-ans-gold {{ color: {INK}; }}
.gradio-container table.qm-ans td.qm-ans-align {{ color: {MUTED}; }}
.gradio-container table.qm-ans td.qm-ans-ctx {{ color: {MUTED}; font-family: {MONO}; font-size: 10.5px; white-space: nowrap; }}
.gradio-container table.qm-ans tr.qm-ans-ok td.qm-ans-now, .gradio-container table.qm-ans tr.qm-ans-ok td.qm-ans-mark {{ color: {OK}; font-weight: 600; }}
.gradio-container table.qm-ans tr.qm-ans-bad td.qm-ans-now, .gradio-container table.qm-ans tr.qm-ans-bad td.qm-ans-mark {{ color: {ALERT}; font-weight: 600; }}
.gradio-container table.qm-ans tr.qm-ans-x td {{ opacity: .45; }}

/* ---------- panel A: the two halves ---------- */
.qm-spec {{ display: flex; flex-direction: column; gap: 7px; }}
.qm-half {{ display: flex; flex-direction: column; gap: 5px; }}
.qm-half-head {{ display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 9px; }}
.qm-tag {{
  font-family: {SANS}; font-size: 12px; font-weight: 700; letter-spacing: .01em;
  border-radius: 6px; padding: 1px 8px; color: #FFFFFF !important; flex: none;
}}
.qm-tag-bs {{ background: {BACKSTORY_INK}; }}
.qm-tag-st {{ background: {STANCE_INK}; }}
.qm-half-def {{ font-family: {SANS}; font-size: 12px; color: {MUTED}; }}
/* The persona picker shows short labels; the full label is printed under the
   picker so nothing is ever clipped by the input's width. */
.gradio-container .qm-pick input, .gradio-container .qm-pick .single-select {{
  font-size: 13px !important; font-family: {MONO} !important;
}}
.qm-who {{ font-family: {MONO}; font-size: 11.5px; color: {MUTED}; margin: -4px 0 6px 2px;
           white-space: normal; overflow-wrap: anywhere; }}
.qm-dom {{ font-family: {MONO}; font-size: 10.5px; color: {MUTED}; background: {FILL};
           border-radius: 999px; padding: 1px 7px; margin-right: 4px; vertical-align: middle; }}
.qm-half-body {{
  font-family: {SANS}; font-size: 13.5px; line-height: 1.5; color: {INK_SOFT};
  border-radius: 10px; padding: 9px 12px; white-space: normal; overflow-wrap: anywhere;
}}
.qm-half-body details.qm-pop {{ margin-left: 6px; }}
.qm-bs-block {{ background: {BACKSTORY_TINT}; border-left: 3px solid {BACKSTORY_INK}; }}
.qm-st-block {{ background: {STANCE_TINT}; border-left: 3px solid {STANCE_INK}; }}
.qm-stance-row {{ display: grid; grid-template-columns: 7.5em 1fr; gap: 8px; padding: 2px 0; }}
/* Chip, the group's answer, an arrow, this person's answer. A four-column
   grid cannot survive the figure text scale in panel A, the narrowest column
   on the screen: with 1fr the columns refuse to shrink and overflow, and with
   minmax(0, 1fr) they shrink below one word and break mid-word, which is how
   "Very good" ended up set one letter per line. Flex with wrapping has no
   such failure mode — the row simply runs onto a second line when the column
   is narrow, and each answer keeps its own highlight. */
.qm-stance-item {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 4px 7px;
  padding: 3px 0; font-size: 13px;
}}
.qm-stance-item > * {{ min-width: 0; overflow-wrap: break-word; word-break: normal; }}
.qm-stance-key {{
  display: flex; gap: 6px; align-items: center; font-family: {MONO};
  font-size: 10.5px; color: {MUTED}; padding-bottom: 3px;
}}
.qm-stance-key .qm-stance-expected, .qm-stance-key .qm-st {{
  font-size: 10.5px; text-decoration: none; background: none; font-weight: 600;
}}
.qm-stance-head {{ font-family: {MONO}; font-size: 10.5px; color: {MUTED}; padding-bottom: 0; }}
.qm-stance-item + .qm-stance-item {{ border-top: 1px solid rgba(108,52,131,.12); }}
.qm-stance-item .qm-stance-expected {{ font-size: 13px; }}
.qm-arrow {{ color: {MUTED}; text-align: center; }}

.qm-muted {{ color: {MUTED}; font-size: 12px; }}
.gradio-container details.qm-pop-q > summary {{ padding: 0 6px; font-weight: 700; color: {STANCE_INK} !important; }}
.qm-stance-k {{ font-family: {MONO}; font-size: 11.5px; color: {MUTED}; padding-top: 2px; }}
.qm-stance-v {{ font-size: 13.5px; }}
.qm-stance-expected {{ color: {INK_SOFT}; text-decoration: line-through;
                       text-decoration-color: rgba(34,33,28,.35); }}
.qm-spec-foot {{ font-family: {SANS}; font-size: 12.5px; color: {MUTED}; }}
.qm-key table {{ font-size: 12.5px; }}
.qm-key td {{ white-space: normal !important; }}

/* ---------- panel row: A, B and C share one height ----------
   Gradio gives each column its own height, so the three cards used to end
   at three different places and the gaps read as a layout bug in a
   screenshot. The row is a flex container with stretched items; inside
   panel B the chat is the element that absorbs the slack, so B grows to
   whatever A or C needs instead of leaving white space under its controls. */
.qm-panels {{ align-items: stretch !important; }}
.qm-panels > .qm-card {{ display: flex !important; flex-direction: column !important; }}
.qm-panels > .qm-card > .block:last-child {{ margin-bottom: 0 !important; }}
.qm-chat {{ flex: 1 1 auto !important; min-height: 220px !important; }}
.qm-chat > .wrap, .qm-chat .bubble-wrap {{ height: 100% !important; }}

/* ---------- chat ---------- */
.bubble-wrap, .chatbot {{ background: transparent !important; }}
.message, .message p, .bubble p {{ font-size: 14px !important; line-height: 1.5 !important; }}
.message.user, .user .bubble {{ background: {FILL} !important; border: none !important; }}
.message.bot, .bot .bubble {{ background: {PANEL} !important;
                              border: 1px solid {GRID} !important; }}
@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; }}
  .qm-face {{ animation: none !important; }}
}}
"""


# Panel titles are the four System Overview bullets of the paper, verbatim, so
# a reader can go from the text to the screen without a mapping. Keep the two
# in step: if a bullet is retitled in the paper, retitle it here in the same
# commit.
#
PANEL_TITLES = {
    "A": "Heterodox Persona Curation",
    "B": "Dual-Stream Trait Anchoring",
    "C": "Real-Time Geometric Tracking",
    "D": "Live Multi-Turn Auditing & Downstream Tasks",
}


def panel(letter: str, title: str = "", sub: str = "") -> str:
    """Lettered panel header, so a figure caption can say 'panel (C)'. The
    title is the module name from the paper's System Overview, verbatim."""
    tail = f'<span class="qm-sub">{sub}</span>' if sub else ""
    return (f'<div class="qm-head-row"><span class="qm-letter">{letter}</span>'
            f'<span class="qm-title">{PANEL_TITLES.get(letter, title)}</span>{tail}</div>')


# =====================================================================
# UI — four lettered panels, sized to screenshot as one figure
# =====================================================================
with gr.Blocks(title="QuirkyMind — persona drift, live") as demo:
    session = gr.State(value=None)
    figure_style = gr.HTML(value="")

    gr.HTML(f'<div class="qm-title-row">{LOGO_SVG}<h1 class="qm-title">QuirkyMind</h1></div>')
    fig_caption = gr.HTML(value="", visible=False)
    gr.Markdown(
        "A persona with a <span class=\"qm-bs\"><b>backstory</b></span> and a "
        "<span class=\"qm-st\"><b>stance</b></span> its group would not expect. Talk to it; "
        "after every turn it re-answers its own survey items, and you see which ones flip. "
        "<details class=\"qm-pop\"><summary>how it works</summary><div class=\"qm-pop-body\">Every turn the "
        "persona silently re-answers the MCQA bank it was aligned to, from inside the "
        "live conversation, under the paper\u2019s three regimes (No-Turn / Single-Turn / "
        "Multi-Turn). Blue items are congruous with the backstory, purple ones are the "
        "incongruous stance. The headline counts only items no control condition already "
        f"answers correctly.<br><br><span class=\"qm-muted\">Steering: {steering_status()}.</span>"
        "</div></details>"
        + ("\n\n**UI-only mode** — generation and probing are stubbed; numbers are not real."
           if MOCK else ""),
        elem_id="qm-head")
    with gr.Row(equal_height=True, elem_classes=["qm-panels"]):
        # ---------------- (A) persona = backstory + stance ----------------
        with gr.Column(scale=3, elem_classes=["qm-card"]):
            gr.HTML(panel("A"))
            _doms = domain_choices()
            # Default to the first real domain when a bank is loaded, so the
            # demo does not open on a hand-written persona.
            _dom0 = _doms[1] if len(_doms) > 2 and _doms[1] != "Built-in" else _doms[0]
            domain_pick = gr.Dropdown(_doms, value=_dom0, label="Domain", container=True,
                                      visible=len(_doms) > 2)
            _first = personas_in(_dom0) or list(PRESETS.keys())
            preset = gr.Dropdown(_first, value=_first[0], label="Persona", container=True,
                                 elem_classes=["qm-pick"])
            align_btn = gr.Button("Align", elem_id="qm-align")
            persona_view = gr.HTML()

        # ---------------- (B) conversation ----------------
        with gr.Column(scale=4, elem_classes=["qm-card"]):
            gr.HTML(panel("B"))
            face_view = gr.HTML()
            chat = gr.Chatbot(label="Persona", height=220, show_label=False,
                              elem_classes=["qm-chat"], **CHAT_KW)
            gr.HTML(highlight_legend_html())
            chat_mode = gr.Radio(["Scripted", "Free chat"], value="Scripted",
                                 show_label=False, container=False, elem_classes=["qm-mode"])
            script_btn = gr.Button("\u25b6 Play scripted turn 1/%d" % len(DEMO_SCRIPT),
                                   variant="primary", elem_id="qm-send")
            with gr.Column(visible=False) as free_box:
                msg = gr.Textbox(label="Your message", lines=2, show_label=False,
                                 placeholder="Ask something, or load a pressure preset.")
                with gr.Row():
                    pressure = gr.Dropdown(list(PRESSURES.keys()),
                                           value=list(PRESSURES.keys())[0],
                                           label="Pressure preset", scale=3, container=False)
                    load_btn = gr.Button("Load", scale=1, elem_classes=["qm-btn"])
                    send_btn = gr.Button("Send", variant="primary", elem_id="qm-send", scale=1)
            with gr.Accordion("Agent loop this turn", open=False,
                              elem_classes=["qm-drawer"]) as trace_box:
                trace_view = gr.Markdown()

        # ---------------- (C) live drift ----------------
        with gr.Column(scale=5, elem_classes=["qm-card"]):
            gr.HTML(panel("C"))
            answers_view = gr.HTML()
            matrix_plot = gr.Plot(show_label=False, elem_classes=["qm-matrix"])
            spectrum_plot = gr.Plot(show_label=False, elem_classes=["qm-plane"])

    # ---------------- (D) comparison and analysis ----------------
    with gr.Row():
        with gr.Column(scale=1, elem_classes=["qm-card"]):
            gr.HTML(panel("D"))
            with gr.Tabs():
                with gr.Tab("Regimes"):
                    compare_view = gr.Markdown(elem_classes=["qm-headline"])
                    method_bars = gr.Plot(show_label=False)
                    d_geo = gr.Plot(show_label=False, elem_classes=["qm-d-extra"])
                    d_metrics = gr.Markdown(elem_classes=["qm-d-extra", "qm-d-metrics"])
                    with gr.Accordion("Per-item matrices", open=False,
                                      elem_classes=["qm-drawer"]):
                        with gr.Row():
                            single_matrix = gr.Plot(show_label=False)
                            qm_matrix = gr.Plot(show_label=False)
                    with gr.Accordion("What each arm said", open=False,
                                      elem_classes=["qm-drawer"]):
                        arms_view = gr.Markdown()
                with gr.Tab("Drift over time"):
                    drift_plot = gr.Plot(show_label=False)
                with gr.Tab("Geometry"):
                    geo_method = gr.Radio(
                        GEO_METHODS, value=GEO_METHODS[0], label="Projection",
                        info="Anchor axes: fixed by the persona's own backstory and "
                             "stance, so alignment cannot be hidden by topic variance. "
                             "PCA / t-SNE follow variance instead and are fitted per "
                             "session; t-SNE is unstable on this few points.")
                    geo_plot = gr.Plot(show_label=False)
                with gr.Tab("Item key"):
                    item_key_view = gr.HTML()
                with gr.Tab("PNI metrics"):
                    metrics_view = gr.Markdown()
                with gr.Tab("Paper results"):
                    gr.HTML('<div class="qm-head-row">'
                            '<span class="qm-title">Published results</span>'
                            '<span class="qm-sub">Table 2, by topic domain</span></div>')
                    with gr.Row():
                        with gr.Column(scale=3):
                            offline_plot = gr.Plot(show_label=False)
                        with gr.Column(scale=2):
                            metric_pick = gr.Radio(
                                list(OFFLINE_METRICS.keys()),
                                value=list(OFFLINE_METRICS.keys())[0],
                                label="Backbone",
                                info="PQA accuracy per topic domain for the three "
                                     "inference regimes (Table 2). The last group "
                                     "is MacroAcc over all domains.")
                            with gr.Accordion("Full results table", open=False,
                                              elem_classes=["qm-drawer"]) as offline_box:
                                offline_view = gr.Markdown(elem_classes=["qm-tight"])

    # ---------------- bottom strip ----------------
    # Everything below is operator-facing. It sits under the panels because a
    # visitor at the booth needs none of it, and a figure screenshot excludes
    # it entirely.
    with gr.Row(elem_classes=["qm-card"]) as controls_box:
        with gr.Column(scale=1):
            with gr.Accordion("Steering parameters", open=False,
                              elem_classes=["qm-drawer"]):
                script_len = gr.Slider(4, 10, SCRIPT_TURNS["n"], step=1,
                                       label="Scripted turns",
                                       info="How many beats the scripted session plays. "
                                            "Beats 5–6 are small talk that lets the "
                                            "context window slide.")
                max_rounds = gr.Slider(0, 4, 2, step=1,
                                       label="Max refinement rounds on a flip",
                                       info="0 = no intervention, the ablation "
                                            "panel C is measured against.")
                alpha = gr.Slider(0.0, 1.0, 0.5, step=0.05,
                                  label="α re-anchor strength")
                max_new = gr.Slider(24, 160, 72, step=8,
                                    label="Reply length (tokens)")
                probe_anchor = gr.Radio(
                    ["stance-withheld", "full", "none"], value="stance-withheld",
                    label="What the probe sees",
                    info="stance-withheld: backstory in context, stance removed "
                         "— the honest drift test. full: backstory + stance, which "
                         "hands over the Q_s answers, so they hold regardless. "
                         "none: latent prefix only.")
                compare = gr.Checkbox(
                    value=False, label="Run comparison arms",
                    info="Also answers each turn with no persona (No-Turn) and "
                         "with the persona stated once. Triples generation cost.")
                seed = gr.Number(
                    value=-1, precision=0, label="Random seed",
                    info="-1 samples freely. Any other value makes the whole "
                         "session reproducible — set one before recording a "
                         "figure or a video.")
                expressive = gr.Checkbox(
                    value=True, label="Expressive mode",
                    info="Asks the persona to open with a *stage direction*, "
                         "which drives the face in panel B.")
        with gr.Column(scale=1):
            with gr.Accordion("Session", open=False, elem_classes=["qm-drawer"]):
                with gr.Row():
                    restart_btn = gr.Button("Restart", elem_classes=["qm-btn"])
                    export_btn = gr.Button("Export JSON", elem_classes=["qm-btn"])
                trace_file = gr.File(label="Session JSON", interactive=False,
                                     show_label=False)
        with gr.Column(scale=1):
            with gr.Accordion("Paper tools", open=False, elem_classes=["qm-drawer"]):
                greyscale = gr.Checkbox(
                    value=False, label="Greyscale check",
                    info="Recolour charts by luminance to test B&W printing.")
                chart_scale = gr.Slider(
                    0.8, 1.6, 1.0, step=0.1, label="Chart scale",
                    info="Type size and height of the panel C charts and the "
                         "method bars — larger for a projector or a paper column.")
                savefig_btn = gr.Button("Export panels as SVG + PDF",
                                        elem_classes=["qm-btn"])
                fig_files = gr.File(label="Vector figures", interactive=False,
                                    file_count="multiple", visible=False)

    VIEWS = [chat, face_view, answers_view, matrix_plot, spectrum_plot, drift_plot,
             trace_view, persona_view, arms_view, metrics_view,
             geo_plot, compare_view, method_bars, single_matrix, qm_matrix, item_key_view,
             d_geo, d_metrics]
    assert len(VIEWS) == N_VIEWS, (
        f"views() returns {N_VIEWS} values but VIEWS lists {len(VIEWS)} components — "
        f"keep the two in the same order and length or panels render as errors")
    BUSY = [preset, domain_pick, align_btn, script_btn, msg, pressure, load_btn, send_btn,
            restart_btn, export_btn, savefig_btn, probe_anchor, expressive, seed,
            compare, geo_method,
            metric_pick, max_rounds, alpha, max_new, greyscale, chart_scale, script_len]
    N_BUSY = len(BUSY)
    # The figure controls live outside CONTROLS and outside the box that
    # figure mode hides. Putting the Figure-mode checkbox inside the panel it
    # hides is a trap: once on, there is nothing left to click to turn it off,
    # and the only way out is restarting the app. This strip stays visible,
    # sits below every panel, and shrinks to a thin line in figure mode so it
    # never intrudes on a per-panel screenshot.
    with gr.Row(elem_classes=["qm-figbar"]):
        figure_mode = gr.Checkbox(
            value=False, label="Figure mode", scale=1,
            info="Hide every control; screenshot shows only panels A–D. Also "
                 "disables every shadow, which blurs small text when resampled.")
        fig_compact = gr.Checkbox(
            value=FIG_TEXT["compact"], label="Compact figure", scale=1,
            info="Drops the plane, the answers table, the tab strip and the mood "
                 "banner, and shortens the chat — about a third less height.")
        matrix_ends = gr.Checkbox(
            value=MATRIX_ENDS["on"], label="Ends only (C and D)", scale=1,
            info="Panel C: the aligned sheet and the last turn, items across and "
                 "states down. Panel D: the same two columns instead of one group "
                 "per turn. Narrow enough for all four panels in one row.")
        fig_text = gr.Slider(
            0.8, 2.0, FIG_TEXT["scale"], step=0.05, label="Figure text scale", scale=2,
            info="Enlarges the interface text without widening the panels.")

    CONTROLS = [preset, domain_pick, align_btn, script_btn, msg, pressure, load_btn, send_btn,
                trace_box, metric_pick, offline_box, controls_box, chat_mode]
    N_CONTROLS = len(CONTROLS)

    # ---- event wiring ------------------------------------------------
    # Anything that generates or probes is wrapped lock -> work -> unlock.
    # .then() runs even when the wrapped step raised, so a gr.Error (empty
    # message, finished script, missing kaleido) still releases the UI
    # instead of leaving every control greyed out.
    def busy(trigger, fn, inputs=None, outputs=None, extra_then=None):
        ev = trigger(lock_ui, outputs=BUSY).then(
            fn, inputs=inputs or [], outputs=outputs or [])
        if extra_then is not None:
            ev = ev.then(*extra_then[0], **extra_then[1])
        return ev.then(unlock_ui, outputs=BUSY)

    _script_label = ((lambda: gr.update(value=script_label(None)),),
                     {"outputs": [script_btn]})

    demo.load(cb_align, inputs=[preset], outputs=[session] + VIEWS)
    demo.load(cb_offline, inputs=[metric_pick], outputs=[offline_plot, offline_view])

    def cb_domain(dom):
        labs = personas_in(dom) or list(PRESETS.keys())
        return gr.update(choices=labs, value=labs[0])
    domain_pick.change(cb_domain, inputs=[domain_pick], outputs=[preset])
    busy(align_btn.click, cb_align, [preset], [session] + VIEWS, _script_label)
    busy(preset.change, cb_align, [preset], [session] + VIEWS, _script_label)
    busy(restart_btn.click, cb_restart, [session], [session] + VIEWS, _script_label)
    busy(script_btn.click, cb_script, [session, alpha, max_new],
         [session, script_btn] + VIEWS)
    busy(send_btn.click, cb_send, [session, msg, max_rounds, alpha, max_new],
         [session, msg] + VIEWS)
    busy(msg.submit, cb_send, [session, msg, max_rounds, alpha, max_new],
         [session, msg] + VIEWS)
    busy(probe_anchor.change, cb_anchor, [session, probe_anchor], [session] + VIEWS)
    busy(export_btn.click, cb_export, [session], [trace_file])
    def cb_script_len(n, sess):
        SCRIPT_TURNS["n"] = int(n)
        return gr.update(value=script_label(sess))
    script_len.change(cb_script_len, inputs=[script_len, session], outputs=[script_btn])
    chat_mode.change(lambda m: (gr.update(visible=(m == "Scripted")),
                                gr.update(visible=(m != "Scripted"))),
                     inputs=chat_mode, outputs=[script_btn, free_box])
    busy(greyscale.change, cb_greyscale, [greyscale, session, metric_pick],
         [matrix_plot, drift_plot, spectrum_plot, geo_plot, method_bars, offline_plot])
    chart_scale.release(cb_chart_scale, inputs=[chart_scale, session],
                        outputs=[matrix_plot, drift_plot, spectrum_plot, method_bars,
                                 single_matrix, qm_matrix])
    busy(savefig_btn.click, cb_export_figures, [session, metric_pick], [fig_files],
         ((lambda: gr.update(visible=True),), {"outputs": [fig_files]}))

    # Cheap, local-only handlers: no lock needed.
    load_btn.click(cb_pressure, inputs=[session, pressure], outputs=[msg])
    expressive.change(cb_expressive, inputs=[session, expressive], outputs=[session])
    seed.change(cb_seed, inputs=[session, seed], outputs=[session])
    compare.change(cb_compare, inputs=[session, compare], outputs=[session])
    geo_method.change(cb_geometry, inputs=[session, geo_method], outputs=[geo_plot])
    metric_pick.change(cb_offline, inputs=[metric_pick],
                       outputs=[offline_plot, offline_view])
    # NB: the Blocks body runs at module scope, so a loop variable here would
    # shadow module-level helpers (`_c` is the colour helper).
    _fig_inputs = [figure_mode, session, fig_text, fig_compact, matrix_ends]
    for _fig_ctl in (figure_mode, fig_text, fig_compact, matrix_ends):
        _fig_ctl.change(cb_figure_mode, inputs=_fig_inputs,
                        outputs=[figure_style, fig_caption] + CONTROLS)
    # The matrix view changes what the plots draw, not just the CSS, so the
    # panels have to be re-rendered after the flag flips.
    matrix_ends.change(lambda s: views(s), inputs=[session], outputs=VIEWS)
    for _fig_ctl in (figure_mode, fig_text):
        _fig_ctl.change(lambda s: views(s), inputs=[session], outputs=VIEWS)


# =====================================================================
# Bank schema (QUIRKYMIND_BANK)
# =====================================================================
# [
#   {"label": "...", "identity": "...", "backstory": "...",
#    "neutral_backstory": "...",   # optional: backstory with the stance clause removed
#    "role": "...",                # optional: bare role label for the stereotype control
#    "expected": "...", "declared": "...",
#    "probes": [{"id": "s1", "kind": "stance", "question": "...",
#                "options": ["...", "..."], "target": 1}]}
# ]
# Item ids are shown as Q_<id>; use b1, b2, ... for backstory ("bio") items and
# s1, s2, ... for stance items so the labels read Q_b1 / Q_s1 in every panel.

def screen_bank() -> None:
    """QUIRKYMIND_SCREEN=1: align every persona in the bank, print how many
    items are diagnostic and how many the aligned state actually holds, then
    exit. Use it to pick demo personas — one where the incongruous item is
    diagnostic AND held at alignment — instead of discovering in front of an
    audience that the headline is 0/2 on two unanswerable items."""
    print(f"{'persona':44s} {'items':>5s} {'unseen@align':>12s} {'prefix-alone':>12s} {'raw':>5s}  problem items")
    roll = {}
    for lab, p in PRESETS.items():
        sess = new_session(p)
        use = usable(sess)
        diag_i = [q for q in use if q.kind == "stance"]
        held = sum(sess.baseline.hit(q) for q in use)
        notes = []
        for q in p.probes:
            if q not in use:
                notes.append(f"{qid(q)}:free")
            elif not sess.baseline.hit(q):
                notes.append(f"{qid(q)}:miss@align")
        st = [q for q in p.probes if q.kind == "stance"]
        hp = sum(sess.steer_prefix.hit(q) for q in st) if sess.steer_prefix else 0
        hn = sum(sess.steer_none.hit(q) for q in st) if sess.steer_none else 0
        uh, ut = agreement_heldout(sess, sess.baseline)
        r = roll.setdefault(p.domain or "built-in", [0, 0, 0, 0, 0])
        r[0] += uh; r[1] += ut; r[2] += hp; r[3] += hn; r[4] += len(st)
        print(f"{lab[:44]:44s} {p.n():5d} {uh:>7d}/{ut:<4d} "
              f"{hp:>7d}/{len(st):<4d} {hn:>2d}/{len(st):<2d}  {' '.join(notes)}")
    if len(roll) > 1:
        print(f"\n{'domain':10s} {'unseen@align':>13s} {'prefix-alone':>13s} {'raw':>8s} {'gain':>6s}")
        for dom, (uh, ut, hp, hn, ns) in sorted(roll.items()):
            g = (hp - hn) / ns if ns else 0.0
            print(f"{dom:10s} {uh:>6d}/{ut:<6d} {hp:>6d}/{ns:<6d} {hn:>3d}/{ns:<4d} {g:+.2f}")
    print("\nprefix-alone / raw = stance items probed with no backstory and no history, "
          "with the soft prefix vs with nothing. If these two columns are equal across "
          "the bank, the checkpoint is not carrying the stance — whatever the aligned "
          "column shows is coming from the text.")


if os.environ.get("QUIRKYMIND_SCREEN"):
    screen_bank()
    raise SystemExit(0)


if __name__ == "__main__":
    # One job at a time: two overlapping turns would interleave writes to the
    # same session prefix and dialogue.
    demo.queue(default_concurrency_limit=1)
    _kw = dict(server_name="0.0.0.0", server_port=PORT)
    _theme = gr.themes.Soft(primary_hue="slate", neutral_hue="stone")
    if _GR_MAJOR >= 6:
        demo.launch(theme=_theme, css=CSS, **_kw)
    else:
        demo.theme, demo.css = _theme, CSS
        demo.launch(**_kw)