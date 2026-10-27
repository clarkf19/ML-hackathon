# Overnight guide (do this when Clark says "end of day")

## Step 1 — send Claude everything that finished
From Google Drive `h4h_r2/out/`, download every file starting with `oofpart_`, `oofdpart_`, `ptepart_`
that is not yet in `C:\Users\clark\hack4hack-r2\data\` and put it there. Tell Claude "uploaded".
Claude then builds the newest submission and the round-4 label file `pl_test4.npy` (takes ~10 min).

## Step 2 — upload the round-4 label file (owner account clark.fernadnes24@spit.ac.in)
Upload `C:\Users\clark\hack4hack-r2\colab\pl_test4.npy` into the `h4h_r2` folder.
(If you skip this, the overnight runs automatically fall back to round-3 labels — still fine.)

## Step 3 — set up each NEW overnight account (4 accounts: 2 for Clark, 2 for teammate)
For EACH new account:
1. Owner account: right-click `h4h_r2` → Share → add the new email as **Editor**.
2. New account (Incognito window, signed in only to it): Drive → **Shared with me** → right-click `h4h_r2`
   → Organize → **Add shortcut → My Drive**.
3. colab.research.google.com → File → **Upload notebook** → the NEWEST
   `C:\Users\clark\hack4hack-r2\colab\signal_hunters_gpu.ipynb`.
4. Runtime → Change runtime type → **T4 GPU** → Save.
5. In the LAST cell change only the `CFGS` line:

| Account | `CFGS = ...` | what it trains |
|---|---|---|
| Clark #1 | `CFGS = ['P4', 'E2']` | ResNet-34 pretrained + round-4 labels, then EfficientNet-B2 |
| Clark #2 | `CFGS = ['C1', 'P6']` | ConvNeXt-Tiny, then ResNet-34 + mixup |
| Teammate #1 | `CFGS = ['E3', 'R50']` | EfficientNet-B0 + round-4 labels, then ResNet-50 |
| Teammate #2 | `CFGS = ['P5']` | ResNet-34 at 384 px (slow, high resolution) |

   If you have fewer accounts, use them in this order: Clark #1, Teammate #1, Clark #2, Teammate #2.
6. Runtime → **Run all** → choose the NEW account for Drive → tick all boxes → Continue.
7. Check the output shows: `cuda True`, `pl_test4.npy copied` (or the fallback message), `patched`,
   `=== running P4 ...` and then `ImageNet weights loaded` (ResNet) or `torchvision ... pretrained` (others).

## Step 4 — before sleeping
- Laptops: plugged in, lid open, Windows sleep = Never, Colab tab left open (not minimised).
- Leave this Claude window open on the main laptop.

## Step 5 — in the morning
1. For any Colab tab that disconnected: Runtime → **Run all** again (it resumes from the last saved epoch).
2. Download all new `oofpart_ / oofdpart_ / ptepart_` files from `h4h_r2/out/` into `data\`.
3. Tell Claude "uploaded" → Claude runs `src/auto_final.py`, ranks candidate blends and tells you which
   `submissions\cand_*.csv` to upload. 10 uploads are available before the 10:00 deadline.
4. Before 10:00: add the GitHub link `https://github.com/clarkf19/ML-hackathon` in the Submission tab.
