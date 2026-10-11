#!/bin/bash
# run_pass2.sh: pass 2 on the C184's 16 leading families (by the breadth pass's lattice score):
# cap 6e12 at budget 3, the first 4 also with -murphy (C185-size MurphyE bounds). 2 workers x
# -t 8, -launch 30 (desktop-friendly), pulling from one queue. Resumable: skips logged searches.
G=$(cd "$(dirname "$0")" && pwd)
cd $G && mkdir -p pass2/out
BIN=$G/bin/s23_ropt_new log=pass2/log.txt q=pass2/queue
MURPHY="-murphy -Bf 4294967296 -Bg 2147483648 -area 2.68435456e16 -K 16000 -Keval 64000 -maxeval 384"
LEAD="s0004 s0037 s0013 s0012"
[ -f $q ] || cp pass2/todo.txt $q
worker() {
  while :; do
    s=$(flock $q.lock sh -c "head -1 $q; sed -i 1d $q")
    [ -z "$s" ] && break
    grep -q "^$s " $log 2>/dev/null && continue
    X=; case " $LEAD " in *" $s "*) X=$MURPHY;; esac
    a=$(date +%s.%N)
    $BIN -poly seeds/$s.poly -search -budget 3.0 -aw 1.3 -maxcells 6e12 -rerank 4096 -top 3 -t 8 -launch 30 $X \
      -out pass2/out/$s.poly > pass2/out/$s.txt 2> pass2/out/$s.err
    rc=$?
    echo "$s rc=$rc wall=$(echo "$(date +%s.%N) - $a" | bc) murphy=${X:+yes} $(grep -m1 '^# s23_ropt' pass2/out/$s.txt | sed 's/.*search: //') | $(grep -m1 '^# size model' pass2/out/$s.txt | sed 's/.*minimum //')" >> $log
  done
}
echo "START $(date) cap 6e12, 2 workers x -t 8, -launch 30" >> $log
start=$(date +%s.%N)
worker & worker & wait
echo "TOTAL $(echo "$(date +%s.%N) - $start" | bc) s" >> $log
