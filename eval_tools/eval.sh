ckpt=results/phase1_ckpt
phase=1
python eval_tools/eval.py $ckpt $phase

ckpt=results/phase2_ckpt
phase=2
python eval_tools/eval.py $ckpt $phase
