# QuirkyMind

Steering an LLM to hold a real survey respondent's heterodox stance — and
auditing, item by item, whether it still does.

A persona here is one American Trends Panel respondent: a **backstory** the
prompt carries, and a **stance** the prompt never states, which reaches the
model only through a trained soft prefix. After every conversational turn the
system silently re-answers that respondent's own survey items from inside the
live dialogue, so drift shows up as specific items flipping rather than as a
falling score.

## Quick start

No GPU needed — this runs the whole interface with generation stubbed:

```bash
pip install -r requirements.txt
QUIRKYMIND_MOCK=1 QUIRKYMIND_BANK=banks/all_bank.json python quirkymind_demo.py
```

With a trained prefix:

```bash
QUIRKYMIND_BANK=banks/all_bank.json \
QUIRKYMIND_CKPT=checkpoints/quirkymind_v5/quirkymind.pt \
python quirkymind_demo.py
```

Add `QUIRKYMIND_4BIT=1` to load the backbone in NF4 (~6 GB instead of ~16 GB).

## Files

| File | What it does |
|---|---|
| `quirkymind_demo.py` | the interface: persona, conversation, per-turn audit, four-condition comparison |
| `build_banks.py` | turns the seven ATP CSVs into persona banks and training data |
| `train_quirkymind.py` | trains the soft prefix (and optional LoRA) |
| `incongruity_stats.py` | reproduces the incongruity base rate reported in the paper |
| `make_fig1.py` | composes panel screenshots into one figure |

## Data

The ATP files are Pew Research Center's to distribute and are not included.
Download the waves from
<https://www.pewresearch.org/american-trends-panel-datasets/>, put the seven
`*_test.csv` files beside the scripts, then:

```bash
python build_banks.py --csv-dir . --out banks --per-domain 40 --bank-per-domain 6
cat banks/report.txt
```

An item counts as **incongruous** when at most 30% of the respondent's own
group gives that answer and it is not the group's mode; the group is the
respondent's political ideology where the wave asks it, otherwise respondents
sharing the same demographic line, otherwise the population.

## Training

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_quirkymind.py \
  --data banks/all_train.jsonl --bank banks/all_bank.json \
  --gphi-input declared --soft-tokens 32 --epochs 8 \
  --load-4bit --grad-checkpoint --batch 1 --grad-accum 32 --max-len 640 \
  --out checkpoints/quirkymind_v5
```

Each epoch prints the prefix gain on stance items, per domain. That number,
not the training loss, is what says whether the prefix carries the stance.

## Attribution

Backbone: Llama-3.1-8B-Instruct, frozen. Survey data: Pew Research Center,
American Trends Panel; the Center bears no responsibility for the analyses
presented here.

## Code release

The interface and the training script are withheld during review and will be
released with the camera-ready. The data pipeline, the persona banks and the
statistics script are here so that every number in the paper can be checked.
