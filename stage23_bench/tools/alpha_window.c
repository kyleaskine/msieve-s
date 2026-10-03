/* Exhaustive affine-alpha line sieve over one window of a rotation u-line.
 *
 * Reads a spec on stdin (written by prefilter_window.py):
 *   W <u> <v_lo> <n>                  window: cells v = v_lo .. v_lo + n - 1 on line u
 *   S <m> <s_0> ... <s_{m-1}>         small-prime class score per v mod m (the prefilter)
 *   THR <k> <t_1> ... <t_k>           class-score cutoffs (top 0.01%, 0.1%, ...)
 *   T <m> <w_0> ... <w_{m-1}>         one prime power: score added to cells with v = j mod m
 *   ...
 *   END
 * A cell's score is sum_T w[v mod m]; affine alpha = const - score, so higher is better.
 * Prints the best cells overall and the best cell whose class passes each cutoff.
 *
 * Build: gcc -O3 -march=native -o alpha_window alpha_window.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUNK 65536
#define TOPN 10

typedef struct { int m; float *w; } table_t;

static long long mod(long long a, long long m) { a %= m; return a < 0 ? a + m : a; }

int main(void)
{
    char tag[8];
    long long u = 0, v_lo = 0, n = 0;
    int sm = 0, nthr = 0, ntab = 0, cap = 0;
    float *s = NULL, thr[16];
    table_t *tabs = NULL;

    while (scanf("%7s", tag) == 1) {
        if (!strcmp(tag, "W")) {
            if (scanf("%lld %lld %lld", &u, &v_lo, &n) != 3) return 1;
        } else if (!strcmp(tag, "S")) {
            if (scanf("%d", &sm) != 1) return 1;
            s = malloc(sm * sizeof(float));
            for (int i = 0; i < sm; i++) if (scanf("%f", &s[i]) != 1) return 1;
        } else if (!strcmp(tag, "THR")) {
            if (scanf("%d", &nthr) != 1 || nthr > 16) return 1;
            for (int i = 0; i < nthr; i++) if (scanf("%f", &thr[i]) != 1) return 1;
        } else if (!strcmp(tag, "T")) {
            if (ntab == cap) { cap = cap ? 2 * cap : 512; tabs = realloc(tabs, cap * sizeof(table_t)); }
            int m;
            if (scanf("%d", &m) != 1) return 1;
            tabs[ntab].m = m;
            tabs[ntab].w = malloc(m * sizeof(float));
            for (int i = 0; i < m; i++) if (scanf("%f", &tabs[ntab].w[i]) != 1) return 1;
            ntab++;
        } else if (!strcmp(tag, "END")) {
            break;
        }
    }

    float *buf = malloc(CHUNK * sizeof(float));
    float top_s[TOPN]; long long top_v[TOPN]; int ntop = 0;
    float best_s[16]; long long best_v[16]; long long cnt_pass[16];
    for (int i = 0; i < nthr; i++) { best_s[i] = -1e30f; best_v[i] = 0; cnt_pass[i] = 0; }

    for (long long off = 0; off < n; off += CHUNK) {
        int len = (int)(n - off < CHUNK ? n - off : CHUNK);
        long long vb = v_lo + off;
        memset(buf, 0, len * sizeof(float));
        for (int t = 0; t < ntab; t++) {
            int m = tabs[t].m;
            const float *w = tabs[t].w;
            int idx = (int)mod(vb, m), j = 0;
            while (j < len) {                      /* contiguous runs, so the inner loop vectorizes */
                int run = m - idx;
                if (run > len - j) run = len - j;
                for (int k = 0; k < run; k++) buf[j + k] += w[idx + k];
                j += run;
                idx = 0;
            }
        }
        for (int j = 0; j < len; j++) {
            float sc = buf[j];
            long long v = vb + j;
            float cls = s[mod(v, sm)];
            for (int i = 0; i < nthr; i++)
                if (cls >= thr[i]) {
                    cnt_pass[i]++;
                    if (sc > best_s[i]) { best_s[i] = sc; best_v[i] = v; }
                }
            if (ntop < TOPN || sc > top_s[ntop - 1]) {
                int p = ntop < TOPN ? ntop++ : TOPN - 1;
                while (p > 0 && top_s[p - 1] < sc) { top_s[p] = top_s[p - 1]; top_v[p] = top_v[p - 1]; p--; }
                top_s[p] = sc; top_v[p] = v;
            }
        }
    }
    for (int i = 0; i < ntop; i++)
        printf("TOP %lld %lld %.6f %.6f\n", u, top_v[i], top_s[i], s[mod(top_v[i], sm)]);
    for (int i = 0; i < nthr; i++)
        printf("THR %d %lld %lld %.6f %lld\n", i, u, best_v[i], best_s[i], cnt_pass[i]);
    return 0;
}
