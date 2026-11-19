# QuirkyMind: Geometric Tracking of Persona Drift under Identity Incongruity

Interactive demonstration accompanying our AAAI-27 Demonstration Track paper.

People are often *incongruous*: a stable demographic profile coexists with an
atypical stance (for example, a conservative who is satisfied with the state of
democracy). Prompted LLM personas tend to lose these stances over a
conversation and revert to what their demographic group typically says.
QuirkyMind lets you talk to such a persona and watch, turn by turn, which
stances it keeps, which it loses, and whether the trained steering module
makes a difference.

The underlying method is described in
[Faithful Persona Steering under Incongruity via Dual-Stream Refinement](https://aclanthology.org/2026.findings-acl.1205/)
(Findings of ACL 2026).

## What the interface shows

| Panel | Name | What it does |
|---|---|---|
| A | Heterodox Persona Curation | Pick a persona; its backstory and declared stance are shown next to the stance its group typically holds. |
| B | Dual-Stream Trait Anchoring | Chat with the persona. Its stage directions drive a small expression readout. |
| C | Real-Time Geometric Tracking | After every reply, a probe re-answers the persona's survey items inside the live conversation. An item-by-turn matrix shows which answers hold or flip; a plane places each turn by backstory retention (x) and stance retention (y). |
| D | Live Multi-Turn Auditing & Downstream Tasks | The same conversation scored under four settings, plus reply-level metrics (BLEU-1, Distinct-1/2, EDIV, SDIV). |

The four settings in panel D:

| Setting | Prompt | Sees the conversation | Trained module |
|---|---|---|---|
| No-Turn | backstory only | no | off |
| Single-Turn | backstory + one example pair | no | off |
| Multi-Turn, steering off | backstory + conversation | yes | off |
| Multi-Turn (QuirkyMind) | backstory + conversation | yes | on |

No-Turn and Single-Turn never see the conversation, so they cannot drift and
are drawn as reference levels. The two Multi-Turn settings differ only in the
trained module, so the gap between them is the effect of the method.

The probe never sees the declared stance in its prompt. Each survey item is
marked *open-book* (its answer is stated in the backstory or the example pair)
or *held-out* (its answer appears nowhere in the prompt); the plane in panel C
uses held-out items.

## Repository contents

| File | Purpose |
|---|---|
| `quirkymind_demo.py` | The interactive interface (Gradio). |
| `train_quirkymind.py` | Trains the soft-prompt adapter the interface loads. |
| `build_banks.py` | Builds the paper's persona banks and training data from the seven ATP wave CSVs. |
| `atp_to_quirkymind.py` | Builds a bank from a single long-format ATP export, grouping by ideology only. |
| `table1_stats.py` | Reproduces the incongruity rates reported in the paper (85.8% at 30%, 64.2% at 20%). |
| `incongruity_stats.py` | The same rate over every respondent rather than the paper's sample (81.8%). |
| `make_fig1.py` | Composes panel screenshots into one figure. |
| `banks/` | The persona banks used in the paper, per domain and pooled (`all_bank.json`). |
| `CITATION.bib`, `CITATION.cff` | Citation for the demo paper and the method paper. |

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

The default backbone is `meta-llama/Llama-3.1-8B-Instruct`, a gated model.
Request access on Hugging Face, then log in:

```bash
huggingface-cli login
```

## Quick start

Run the interface without a GPU or model download. Generation and scoring are
stubbed, so the numbers are not real, but every panel works:

```bash
QUIRKYMIND_MOCK=1 python quirkymind_demo.py
```

Run it for real (a single GPU with about 20 GB free; add `QUIRKYMIND_4BIT=1`
to fit in about 8 GB):

```bash
python quirkymind_demo.py
```

Then open `http://localhost:7860`.

## Trained adapter

The trained adapter is not included. Without it the interface still runs end
to end, but the Multi-Turn (QuirkyMind) setting uses a placeholder prefix and
the page header says so. To train your own, see the next section.

## Building personas and training

### 1. Get the survey data

The personas come from Pew Research Center's American Trends Panel (ATP).
Download the waves from [Pew](https://www.pewresearch.org/american-trends-panel-datasets/)
(free account required). The data cannot be redistributed here.

`atp_to_quirkymind.py` expects a CSV with one row per (respondent, question):

| Column | Content |
|---|---|
| `persona_id` | respondent id |
| `MCQ` | question text |
| `options` | answer options as a Python list literal, e.g. `"['Yes', 'No']"` |
| `label` | index of the respondent's answer in `options` |
| `backstory` | one-line demographic description of the respondent |

### 2a. Reproduce the paper's banks and statistics

Put the seven wave CSVs (`GL_test.csv`, `JN_test.csv`, `MT_test.csv`, `PL_test.csv`,
`RL_test.csv`, `SC_test.csv`, `SO_test.csv`) in one directory, then:

```
python build_banks.py --csv-dir . --out banks --per-domain 40 --bank-per-domain 6
python table1_stats.py --csv-dir .
```

An answer counts as incongruous when at most 30% of the respondent's group gives it and
it is not the group's most common answer. The group is the respondent's political
ideology where the wave asks it (GL), otherwise respondents with the same backstory
line (JN, MT, SO), otherwise the whole population (PL, RL, SC). `banks/report.txt`
records the counts per domain.

### 2b. Build a bank from a single ATP export

```bash
python atp_to_quirkymind.py --csv atp_export.csv \
    --out-bank atp_bank.json --out-train atp_training.jsonl
```

An answer counts as incongruous when at most 30% of the respondent's ideology
group gives it and it is not the group's most common answer
(`--incongruity-threshold` changes the 30%). Personas are split at the
respondent level: those placed in the bank are never used for training.

### 3. Train

```bash
python train_quirkymind.py --data atp_training.jsonl --check-data
python train_quirkymind.py --data atp_training.jsonl --bank atp_bank.json \
    --gphi-input declared --soft-tokens 32 --epochs 8 \
    --out checkpoints/quirkymind
```

After each epoch the script scores the bank with and without the trained
prefix and prints the gain on stance items. On a shared GPU add
`--load-4bit --grad-checkpoint`.

### 4. Run the interface on your personas

```bash
QUIRKYMIND_BANK=atp_bank.json \
QUIRKYMIND_CKPT=checkpoints/quirkymind/quirkymind.pt \
python quirkymind_demo.py
```

The terminal prints a `[QuirkyMind]` line for each of these confirming that
the file was found and loaded.

To check the whole bank without opening the interface:

```bash
QUIRKYMIND_SCREEN=1 QUIRKYMIND_BANK=atp_bank.json \
QUIRKYMIND_CKPT=checkpoints/quirkymind/quirkymind.pt python quirkymind_demo.py
```

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `QUIRKYMIND_MOCK` | `0` | `1` stubs the model for a UI-only run. |
| `QUIRKYMIND_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Backbone. |
| `QUIRKYMIND_PORT` | `7860` | Server port. |
| `QUIRKYMIND_BANK` | unset | Persona bank (JSON). Unset uses the built-in personas. |
| `QUIRKYMIND_CKPT` | unset | Trained adapter from `train_quirkymind.py`. |
| `QUIRKYMIND_LORA` | unset | Optional LoRA adapter directory (`train_quirkymind.py --lora`). |
| `QUIRKYMIND_4BIT` | `0` | `1` loads the backbone in 4-bit. |
| `QUIRKYMIND_SBERT` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence encoder. |
| `QUIRKYMIND_OFFLINE` | unset | JSON of offline results for panel D. |
| `QUIRKYMIND_SCRIPT` | unset | JSON of scripted turns for the replay button. |
| `QUIRKYMIND_SCREEN` | unset | `1` scores the whole bank in the terminal and exits. |

The adapter records which text it was trained on (`--gphi-input`), and the
interface reads that from the checkpoint, so the two cannot drift apart.

### Persona bank format

```json
[
  {
    "label": "Shown in the persona menu",
    "identity": "One-line description",
    "backstory": "Full backstory (without the declared stance)",
    "neutral_backstory": "Optional: backstory with the stance clause removed",
    "role": "Optional: bare role label used by the stereotype control",
    "expected": "the stance the group typically holds",
    "declared": "the stance this persona actually holds",
    "seed_question": "Optional: question of the one example pair",
    "seed_statement": "Optional: the stance stated in that pair",
    "probes": [
      {"id": "s1", "kind": "stance", "question": "...",
       "options": ["...", "..."], "target": 1, "context": "none"}
    ]
  }
]
```

`kind` is `bio` for congruous items and `stance` for incongruous ones.
`context` is `backstory` or `seed` when the answer is stated in the prompt and
`none` for held-out items.

## Citation

```bibtex
@inproceedings{chu2027quirkymind,
  title     = {{QuirkyMind}: Geometric Tracking of Persona Drift under Identity Incongruity},
  author    = {Chu, Yu-An and Chiang, Meng-Fen},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027},
  note      = {Demonstration track, to appear}
}

@inproceedings{chu2026faithful,
  title     = {Faithful Persona Steering under Incongruity via Dual-Stream Refinement},
  author    = {Chu, Yu-An and Pong, Jen-Ren and Yeh, Chia-Yao and Chiang, Meng-Fen},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {24078--24095},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

## Acknowledgments

This work is supported by the National Science and Technology Council (NSTC),
Taiwan, under grants NSTC 115-2221-E-A49-120-MY3 and NSTC 115-2639-E-A49-001-ASP.

## License

This code is released under the [MIT License](LICENSE). The ATP survey data
is not included and remains subject to Pew Research Center's terms of use.