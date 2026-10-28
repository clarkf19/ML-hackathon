PY="/c/Users/clark/AppData/Local/Programs/Python/Python311/python.exe -X utf8"
cd /c/Users/clark/hack4hack-r2
KEEPGAIN=1 $PY src/prep.py train test inj1 && \
OUT=lv_d2 SEED=123 GAINSD=1.9 $PY src/degrade.py && KEEPGAIN=1 $PY src/prep.py lv_d2 && ZPFX=zr $PY src/qmap.py lv_d2 train_deg2 && rm data/lv_d2_cadences.npy data/zr_lv_d2.npy && echo DONE
