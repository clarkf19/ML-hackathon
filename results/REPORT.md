# Cloud session 4: scintillating_star (draft, numbers pending)

## Constraint: no raw data in this session
`gdown` failed: the cloud environment's network policy returns **403 for drive.google.com** (and for
download.pytorch.org, so no torch either). CLOUD.md says to fall back to `bundle/`, so every number below comes from the
427 committed bundle features (clean / test / deg2 / deg4) and the harness `src/bundle_eval.py`. The raw-byte detector
(task step 3) is written, ported to portable paths, and smoke-tested on synthetic bytes, but it has **not** been run on
the competition data. See "Raw-data detector (ready to run)".

PENDING
