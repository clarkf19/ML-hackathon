#!/bin/bash
# Noise-model calibration: noise-only cadences under several settings, compared with the REAL noise class
# (KS over phys,tex,rep,lvl features).  usage: src/gen_dof.sh OUTTXT 'json1' 'json2' ...   (TEL=PKS env for PKS)
cd "$(dirname "$0")/.."
TEL=${TEL:-GBT}; out=$1; shift; : > $out
for cfg in "$@"; do
  GEN_OVR="$cfg" python src/gen_make.py 270 $TEL 11 data/gen/calib_noise_$TEL.npz --label=8 --sets=phys,tex,rep,lvl > /dev/null 2>&1
  echo "=== $cfg" >> $out
  python src/gen_compare.py data/gen/calib_noise_$TEL.npz $([ $TEL = PKS ] && echo --pks) --cols=phys_best_p0,phys_pers_min4,phys_spec_max,rep_rep_k1,tex_v2x4_p0,phys_tex_max_p0,lvl_lv_off_spread,lvl_lv_tblk_std_off,phys_bb0_max 2>/dev/null >> $out
done
# first sweep (results/calib_dof.txt): '{"TEL.GBT.dof": 10}' '{"TEL.GBT.dof": 14}' '{"TEL.GBT.dof": 18}' '{"TEL.GBT.dof": 26}' '{"TEL.GBT.dof": 40}'
