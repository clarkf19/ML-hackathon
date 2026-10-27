| Variant | idea | clean-val | Δ | fakePKS-val | Δ | fPKS folds better |
|---|---|---|---|---|---|---|
| H argmax (harness baseline) | ref | 0.9279 | +0.0000 | 0.8931 | +0.0000 | 0/5 |
| H + bias | d | 0.9289 | +0.0010 | 0.8946 | +0.0016 | 3/5 |
| H cond-prob cascade (pure) | a | 0.9288 | +0.0008 | 0.8931 | +0.0000 | 4/5 |
| H cond-prob cascade (upward overrides of argmax) | a | 0.9294 | +0.0014 | 0.8936 | +0.0005 | 4/5 |
| H + bias -> cond-prob upward overrides | a+d | 0.9291 | +0.0012 | 0.8950 | +0.0019 | 3/5 |
| cond detectors: pure cascade | a | 0.9283 | +0.0004 | 0.8903 | -0.0027 | 1/5 |
| cond detectors: chain-rule joint argmax | a | 0.9229 | -0.0050 | 0.8869 | -0.0061 | 0/5 |
| cond detectors: chain-rule joint + bias | a+d | 0.9272 | -0.0008 | 0.8912 | -0.0019 | 2/5 |
| H argmax + cond-detector upward overrides | a | 0.9300 | +0.0021 | 0.8953 | +0.0022 | 4/5 |
| H + bias + cond-detector upward overrides | a+d | 0.9303 | +0.0024 | 0.8954 | +0.0024 | 4/5 |
| avg(H, cond joint) + bias | a+d | 0.9305 | +0.0025 | 0.8946 | +0.0015 | 3/5 |
| H argmax + pair overrides | c | 0.9282 | +0.0003 | 0.8930 | -0.0000 | 4/5 |
| H + bias + pair overrides | c+d | 0.9289 | +0.0010 | 0.8936 | +0.0005 | 3/5 |
| H + bias + cond-prob overrides + pair overrides | a+c+d | 0.9292 | +0.0012 | 0.8938 | +0.0007 | 3/5 |
| H + bias + cond-detector overrides + pair overrides | a+c+d | 0.9301 | +0.0021 | 0.8949 | +0.0018 | 3/5 |
| xgb argmax | e | 0.9267 | -0.0012 | 0.8923 | -0.0008 | 1/5 |
| avg(H, xgb) | e | 0.9273 | -0.0006 | 0.8936 | +0.0006 | 3/5 |
| avg(H, xgb) + bias | e+d | 0.9289 | +0.0010 | 0.8931 | +0.0001 | 3/5 |
| meta argmax (submission2 model) | meta | 0.9292 | +0.0012 | 0.8972 | +0.0041 | 4/5 |
| meta + bias | meta+d | 0.9313 | +0.0033 | 0.8980 | +0.0050 | 5/5 |
| meta cond-prob cascade (pure) | meta+a | 0.9305 | +0.0026 | 0.8999 | +0.0068 | 5/5 |
| meta + bias -> cond-prob upward overrides | meta+a+d | 0.9310 | +0.0031 | 0.8990 | +0.0059 | 5/5 |
| meta + bias + pair overrides | meta+c+d | 0.9315 | +0.0036 | 0.8984 | +0.0054 | 5/5 |
| meta argmax + cond-detector upward overrides | meta+a | 0.9306 | +0.0027 | 0.8983 | +0.0052 | 5/5 |
| meta + bias + cond-detector upward overrides | meta+a+d | 0.9322 | +0.0043 | 0.8981 | +0.0050 | 5/5 |
| meta + bias + cond-detector overrides + pair overrides | meta+a+c+d | 0.9325 | +0.0045 | 0.8981 | +0.0051 | 5/5 |
| avg(meta, H) + bias | meta+e+d | 0.9292 | +0.0013 | 0.8966 | +0.0036 | 5/5 |
| H + bias + cond-detector upward overrides [fPKS-tuned] | a+d | 0.9296 | +0.0017 | 0.8947 | +0.0016 | 4/5 |
| meta cond-prob cascade (pure) [fPKS-tuned] | meta+a | 0.9312 | +0.0032 | 0.9000 | +0.0069 | 5/5 |
| meta + bias [fPKS-tuned] | meta+d | 0.9302 | +0.0023 | 0.8987 | +0.0056 | 5/5 |
