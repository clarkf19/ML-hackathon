# Cloud-session guide (Hack4Hack R2 "Signal Hunters")
Read `SUMMARY.md` first (problem, what was tried, numbers).

## Data
Raw data is NOT on main (too big) but IS on the `data` branch of this repo (Google Drive is blocked in the cloud network):
```
pip install lightgbm scikit-learn pandas numpy scipy --quiet
git fetch origin data && git worktree add /tmp/h4hdata origin/data && bash /tmp/h4hdata/get_data.sh "$PWD/data/signal-hunters-round2"
```
-> `data/signal-hunters-round2/{train,test}_cadences.npy` (uint8, N x 6 x 16 x 128), `{train,test}_meta.csv`, `PROBLEM_STATEMENT.md`.
torch may not be installable (download.pytorch.org blocked): use numpy/scipy/LightGBM, or sklearn MLPs.

## Bundle (in git)
`bundle/feats_{clean,test,deg2,deg4}.npz` -> X (rows x 427 features; column names in `bundle/meta.npz['cols']`).
clean = GBT train, test = PKS test, deg2/deg4 = train re-rendered as fake-PKS (row i = same cadence as train row i).
`bundle/meta.npz`: y (label index, order in `labels`), folds (5 fixed folds — ALWAYS use them), oofc/oofd/pte = current best
model's probabilities on clean val / fake-PKS val / test.

## Rules
- Score every idea with `python src/bundle_eval.py your_feats.npz` (baseline first, same run) and report BOTH numbers:
  clean-val and fakePKS-val macro-F1. fakePKS-val tracks the public leaderboard (it ran ~0.01 above it).
- New raw-data features must be computed for train, test AND a fake-PKS rendering of train (`src/degrade.py` + `src/qmap.py`
  logic; Windows paths in src/*.py — adapt them), otherwise the robustness number is meaningless.
- Never use test labels (there are none) or hand-label test. Fixed seeds.
- Deliverable: commit to your branch `results/REPORT.md` (idea, numbers vs baseline, verdict) + the scripts + any small .npz.
