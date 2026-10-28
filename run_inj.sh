PY="/c/Users/clark/AppData/Local/Programs/Python/Python311/python.exe -X utf8"
cd /c/Users/clark/hack4hack-r2
$PY src/inject.py 4000 7 inj1 1.25 && $PY src/prep.py inj1 && NT=4 $PY src/phys.py inj1 && $PY src/tex.py inj1 && NT=4 $PY src/track.py inj1 && NT=4 $PY src/rep.py inj1 && NT=4 $PY src/rep2.py inj1 && echo DONE
