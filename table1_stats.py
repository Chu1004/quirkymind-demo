"""Table 1 of the demo paper, restricted to the paper's sampled sizes."""
import sys, importlib.util, random, argparse
from collections import Counter
spec = importlib.util.spec_from_file_location("bb","build_banks.py"); bb = importlib.util.module_from_spec(spec); sys.modules["bb"]=bb; spec.loader.exec_module(bb)
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--csv-dir", default="."); a = ap.parse_args()
NAME={"SO":"Social","SC":"Science","RL":"Religion","PL":"Politics","MT":"Methodology","JN":"Journalism","GL":"Global"}
PAPER_N={"SO":400,"SC":200,"RL":200,"PL":200,"MT":400,"JN":400,"GL":400}
rng=random.Random(a.seed); rows=[]; tn=t30=t20=ti=0
for c in ["GL","JN","MT","PL","RL","SC","SO"]:
    df = bb.load(f"{a.csv_dir}/{c}_test.csv")
    wide = df.pivot_table(index="persona_id", columns="MCQ", values="answer", aggfunc="first")
    opts = bb.option_lists(df)
    back = df.drop_duplicates("persona_id").set_index("persona_id")["backstory"].reindex(wide.index)
    has_ideo = bb.IDEO in wide.columns; bcounts = Counter(back.dropna())
    groups = pd.Series({pid: bb.group_of(wide.loc[pid], back[pid] or "", has_ideo, bcounts) for pid in wide.index})
    grouping = "ideology" if has_ideo else ("profile" if any(k.startswith("back:") for k in groups.unique()) else "population")
    dist = bb.group_dists(wide, groups)                    # priors from the FULL wave
    per_persona_q = int(round(len(df)/df.persona_id.nunique()))
    ids=list(wide.index); rng.shuffle(ids); ids=ids[:PAPER_N[c]]
    n=a30=a20=inc_tot=0
    for pid in ids:
        b=back[pid] or ""
        if not b: continue
        st,inc,con = bb.persona_items(wide.loc[pid], b, opts, dist, groups[pid], 0.30)
        _,i20,_    = bb.persona_items(wide.loc[pid], b, opts, dist, groups[pid], 0.20)
        if st or inc or con:
            n+=1; a30+= len(inc)>=1; a20+= len(i20)>=1; inc_tot+=len(inc)
    ng = groups.nunique()
    gb = grouping + (f" ({ng})" if grouping!="population" else "")
    rows.append((NAME[c],c,n,per_persona_q,gb,a30/n*100,a20/n*100,inc_tot/n))
    tn+=n; t30+=a30; t20+=a20; ti+=inc_tot
for nm,c,n,q,gb,p30,p20,mi in rows:
    print(f"{nm} ({c}) & {n} & {q} & {gb} & {p30:.1f} & {mi:.2f} \\\\")
print("\\midrule")
print(f"\\textbf{{All}} & \\textbf{{{tn}}} & --- & --- & \\textbf{{{t30/tn*100:.1f}}} & \\textbf{{{ti/tn:.2f}}} \\\\")
print(f"% 20% threshold, pooled: {t20/tn*100:.1f}%")
