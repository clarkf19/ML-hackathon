# Generator: reverse-engineered cadence renderer (GBT / PKS) — DRAFT, numbers being filled in

**Status: work in progress.** Sections marked TBD are being computed.

## 0. Constraint: raw data unavailable in this session
`gdown` of the organisers' Drive file failed: the cloud environment's network policy denies `drive.google.com`
(proxy 403). Per CLOUD.md the work is therefore done from `bundle/` only:
* pixel-level facts are taken from numbers already logged in the repo (SUMMARY.md, src/degrade.py, src/lvl.py), and
* everything else is estimated / validated in the **427-feature space**: generated cadences are rendered as raw uint8
  bytes and pushed through the *unchanged* feature code (`src/featurize.py`), then compared with the bundle features of
  real train (per class) and real test.
PROBLEM_STATEMENT.md could not be read.

## 1. Instrument model (estimates with numbers)
| effect | GBT (train) | PKS (test) | evidence |
|---|---|---|---|
| dropped rows | Poisson 0.147 / cadence | Poisson 0.355 / cadence | counts 0/1/2/3 = 15535/2297/156/10 vs Poisson 15539/2284/168/8; test 7030/2448/468/53 vs 7012/2489/442/52 |
| glitch rows | each panel glitched w.p. 0.10 (1 row, 2 rows w.p. 0.07) | w.p. 0.18 | #panels with bb0>10 = 9529/6403/1773/273/22 vs Binomial(6,0.10) 9566/6377/1771/262/22; test 3037/4000/2193/640/121/9 vs Binomial(6,0.18) 3040/4004/2197/643/106/9 (Poisson per cadence fits worse: 9491/6074/1944/415/66) |
| glitch amplitude | U(18,46) bytes, 30% partial band (40-63 ch) | U(18,48) bytes | bb0 given >10: median 32.9 (GBT) / 27.3 (PKS) z-units -> ~35-38 bytes |
| noise | log-chi2, dof TBD | log-chi2, dof TBD | dof sweep below |
| panel gain jitter | TBD bytes | TBD bytes | lvl_off_spread median 2.06 (train) / 3.55 (test) |
| pixel drift scale | 1 | 0.88 | strong ON-track drift q75/q90: test 0.219/0.387 vs fake-PKS 0.262/0.45 |
| slew gap | uniform {3,4,5,6} | uniform {3,4,5,6} | 4479/4447/4483/4591 and 2536/2474/2532/2458 |

## 2. Signal rendering (per class) — TBD

## 3. Co-occurrence and priority labelling — TBD

## 4. Validation — TBD

## 5. PKS augmentation in bundle_eval — TBD
