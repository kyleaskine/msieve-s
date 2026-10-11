#!/bin/bash
# run_breadth.sh DIR: the C181/C205 breadth pass (cap 2e11, budget 3, aw 1.3, -rerank 4096,
# -top 3, 4 workers x -t 4) over DIR/todo_breadth.txt. Resumable: skips searches already logged.
G=$(cd "$1" && pwd)
BIN=$G/bin/s23_ropt
cd $G && mkdir -p breadth/out
todo=todo_breadth.txt out=breadth/out log=breadth/log.txt cap=2e11 nw=4
worker() {
  awk -v w=$1 -v n=$nw 'NR % n == w' $todo | while read s; do
    grep -q "^$s " $log 2>/dev/null && continue
    a=$(date +%s.%N)
    $BIN -poly seeds/$s.poly -search -budget 3.0 -aw 1.3 -maxcells $cap -rerank 4096 -top 3 -t 4 -out $out/$s.poly > $out/$s.txt 2>&1
    rc=$?
    b=$(date +%s.%N)
    echo "$s rc=$rc wall=$(echo "$b - $a" | bc) $(grep -m1 '^# s23_ropt' $out/$s.txt | sed 's/.*search: //') | $(grep -m1 '^# size model' $out/$s.txt | sed 's/.*minimum //')" >> $log
  done
}
start=$(date +%s.%N)
echo "START $(date) cap $cap" >> $log
for w in $(seq 0 $((nw - 1))); do worker $w & done
wait
echo "TOTAL $(echo "$(date +%s.%N) - $start" | bc) s" >> $log
