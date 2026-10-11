#!/bin/bash
# GPU sopt effort 0 on every deduplicated raw poly (CADO took 1837 s on 8 workers).
cd ~/msieve-s
G=pipeline_results/gpu_c184
a=$(date +%s.%N); echo "START $(date)" > $G/sopt/e0.log
$G/bin/s23_sopt -sopteffort 0 -inputpolys msieve.dat.deduped.ms > $G/sopt/e0_unsorted.txt 2>> $G/sopt/e0.log
rc=$?
echo "rc=$rc wall $(echo "$(date +%s.%N) - $a" | bc) s END $(date)" >> $G/sopt/e0.log
