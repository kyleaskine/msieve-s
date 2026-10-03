c146 test sieve, 2026-10-03: seven finalists, CADO las (0574bc39d), CADO's params.c145.
Same command for every poly (factor base from makefb -lim 23000000 -maxbits 14 -side 1):

  las -poly X.poly -fb1 fb1_X.gz -I 14 -lim0 15000000 -lim1 23000000 -lpb0 30 -lpb1 30
      -mfb0 58 -mfb1 58 -lambda0 1.935 -lambda1 1.94 -ncurves0 13 -ncurves1 14 -sqside 1
      -q0 Q -q1 Q+4000 -t 8          for Q = 3000000, 8000000, 16000000

ms_rNNNN.poly = best msieve -npr result of the seed at effort-50 exp_E rank NNNN;
cado_orig_r0084.poly = CADO polyselect_ropt (orig) best on the winning seed, at its
skewopt skew. results.tsv: per poly and q range, special-q count, relations,
relations per special-q (+ standard error), CPU seconds per relation.

This is a stand-in for the real job: these numbers are not sieved with GGNFS lasieve
or the job file you would actually use. For that, use ~/code/test-sieve.

Second run (results_lpbr29_lpba30.tsv): the user's c145-style settings, lpb0 (rational)
29, lpb1 (algebraic) 30, I = 14, q from 10M (lims, mfb 58/58 unchanged; lambda0 2.0,
lambda1 1.94), at q0 = 10M, 15M, 20M, 30M (+4000 each). Its CPU s/rel overlapped another
CPU job and should not be compared; relations per special-q are unaffected.
