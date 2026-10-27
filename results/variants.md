| Variant | idea | clean-val | Δ | fakePKS-val | Δ | fPKS folds better |
|---|---|---|---|---|---|---|
| H argmax (harness baseline) | ref | 0.9279 | +0.0000 | 0.8931 | +0.0000 | 0/5 |
| H + bias | d | 0.9289 | +0.0010 | 0.8946 | +0.0016 | 3/5 |
| H cond-prob cascade (pure) | a | 0.9288 | +0.0008 | 0.8931 | +0.0000 | 4/5 |
| H cond-prob cascade (upward overrides of argmax) | a | 0.9294 | +0.0014 | 0.8936 | +0.0005 | 4/5 |
| H + bias -> cond-prob upward overrides | a+d | 0.9291 | +0.0012 | 0.8950 | +0.0019 | 3/5 |
| meta argmax (submission2 model) | meta | 0.9292 | +0.0012 | 0.8972 | +0.0041 | 4/5 |
| meta + bias | meta+d | 0.9313 | +0.0033 | 0.8980 | +0.0050 | 5/5 |
| meta cond-prob cascade (pure) | meta+a | 0.9305 | +0.0026 | 0.8999 | +0.0068 | 5/5 |
| meta + bias -> cond-prob upward overrides | meta+a+d | 0.9310 | +0.0031 | 0.8990 | +0.0059 | 5/5 |
