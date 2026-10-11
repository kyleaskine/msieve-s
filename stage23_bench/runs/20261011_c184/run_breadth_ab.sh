#!/bin/bash
# run_breadth_ab.sh DIR BIN OUT NW T [EXTRA]: run_breadth.sh's breadth pass (cap 2e11, budget 3, aw 1.3,
# -rerank 4096, -top 3) with another binary, NW workers x -t T, into DIR/OUT. Resumable.
G=$(cd "$1" && pwd)
BIN=$G/$2
out=$3 nw=$4 th=$5 EXTRA=$6
cd $G && mkdir -p $out/out
todo=todo_breadth.txt log=$out/log.txt cap=2e11
worker() {
  awk -v w=$1 -v n=$nw 'NR % n == w' $todo | while read s; do
    grep -q "^$s " $log 2>/dev/null && continue
    a=$(date +%s.%N)
    $BIN -poly seeds/$s.poly -search -budget 3.0 -aw 1.3 -maxcells $cap -rerank 4096 -top 3 -t $th $EXTRA -out $out/out/$s.poly > $out/out/$s.txt 2>&1
    rc=$?
    b=$(date +%s.%N)
    echo "$s rc=$rc wall=$(echo "$b - $a" | bc) $(grep -m1 '^# s23_ropt' $out/out/$s.txt | sed 's/.*search: //') | $(grep -m1 '^# size model' $out/out/$s.txt | sed 's/.*minimum //')" >> $log
  done
}
start=$(date +%s.%N)
echo "START $(date) cap $cap bin $2 workers $nw x -t $th $EXTRA" >> $log
for w in $(seq 0 $((nw - 1))); do worker $w & done
wait
echo "TOTAL $(echo "$(date +%s.%N) - $start" | bc) s" >> $log
