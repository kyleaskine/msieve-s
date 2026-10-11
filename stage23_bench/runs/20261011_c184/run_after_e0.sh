#!/bin/bash
# The GPU path after GPU sopt effort 0 (sopt/run_e0.sh), mirroring full_optimization_pipeline.sh:
# top 1000 by exp_E -> GPU sopt effort 50 -> top 300 -> seeds (+ content) -> breadth ropt.
# Phase wall times go to timing.log.
set -e
cd ~/msieve-s
G=pipeline_results/gpu_c184
T() { date +%s.%N; }
lap() { echo "$1 $(echo "$(T) - $2" | bc) s" >> $G/timing.log; }
grep -q '^rc=0 ' $G/sopt/e0.log
echo "e0 $(grep '^rc=0' $G/sopt/e0.log)" >> $G/timing.log
a=$(T)
python3 utils/sort_cado_by_expe.py $G/sopt/e0_unsorted.txt $G/sopt/e0_sorted.txt
python3 utils/extract_input_polys_sorted.py $G/sopt/e0_sorted.txt $G/sopt/e0_unsorted.txt $G/sopt/top1000_input.ms 1000
lap "P1 sort+top1000" $a
a=$(T)
$G/bin/s23_sopt -sopteffort 50 -inputpolys $G/sopt/top1000_input.ms > $G/sopt/e50_output.txt 2> $G/sopt/e50.log
lap "P2 GPU sopt e50" $a
a=$(T)
python3 utils/sort_cado_by_expe.py $G/sopt/e50_output.txt $G/sopt/e50_sorted.txt
python3 utils/cado_to_msieve.py $G/sopt/e50_output.txt $G/sopt/e50_msieve.ms
sort -k10,10n $G/sopt/e50_msieve.ms > $G/sopt/e50_msieve_sorted.ms
head -n 300 $G/sopt/e50_msieve_sorted.ms > $G/sopt/best300_msieve.ms
python3 $G/prepare_gpu.py $G pipeline_results
lap "P3 top300+seeds" $a
for f in top1000_input.ms; do cmp -s $G/sopt/$f pipeline_work/$f && echo "same as CPU: $f" || echo "DIFFERS from CPU: $f"; done >> $G/timing.log
cmp -s $G/sopt/best300_msieve.ms pipeline_results/best300_msieve.ms && echo "same as CPU: best300_msieve.ms" >> $G/timing.log || echo "DIFFERS from CPU: best300_msieve.ms" >> $G/timing.log
a=$(T)
bash $G/run_breadth.sh $G
lap "P4 breadth (bin/s23_ropt, 4 workers x -t 4)" $a
