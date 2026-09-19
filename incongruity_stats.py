"""Compute the incongruity base rate: what fraction of ATP respondents hold at
least one position that is a minority view (<= threshold) within their own
group? Uses the same grouping and thresholds as build_banks.py."""
import sys, importlib.util, argparse, os
spec = importlib.util.spec_from_file_location("bb", "build_banks.py")
bb = importlib.util.module_from_spec(spec); sys.modules["bb"] = bb; spec.loader.exec_module(bb)
import pandas as pd
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--csv-dir", default="/mnt/user-data/uploads")
ap.add_argument("--minority-max", type=float, default=0.30)
args = ap.parse_args()

rows = []
tot_any = tot_all = 0
for code in ["GL", "JN", "MT", "PL", "RL", "SC", "SO"]:
    path = os.path.join(args.csv_dir, f"{code}_test.csv")
    df = bb.load(path)
    wide = df.pivot_table(index="persona_id", columns="MCQ", values="answer", aggfunc="first")
    opts = bb.option_lists(df)
    back = df.drop_duplicates("persona_id").set_index("persona_id")["backstory"].reindex(wide.index)
    has_ideo = bb.IDEO in wide.columns
    bcounts = Counter(back.dropna())
    groups = pd.Series({pid: bb.group_of(wide.loc[pid], back[pid] or "", has_ideo, bcounts)
                        for pid in wide.index})
    dist = bb.group_dists(wide, groups)
    n = n_any = 0; per = []
    for pid in wide.index:
        b = back[pid] or ""
        if not b: continue
        st, inc, con = bb.persona_items(wide.loc[pid], b, opts, dist, groups[pid], args.minority_max)
        if st or inc or con:
            n += 1; per.append(len(inc)); n_any += int(len(inc) >= 1)
    tot_any += n_any; tot_all += n
    rows.append((code, n, n_any, n_any / n, sum(per)/n))
    print(f"{code}  n={n:5d}  >=1 incongruous: {n_any:5d} ({n_any/n:5.1%})  mean incongruous items {sum(per)/n:.2f}")
print(f"\nPOOLED  n={tot_all}  >=1 incongruous: {tot_any} ({tot_any/tot_all:.1%})  "
      f"(minority threshold <= {args.minority_max:.0%} of the respondent's group)")
