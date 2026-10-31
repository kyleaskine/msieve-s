/* Host-only, opt-in phase snapshots for one s23_sopt input. No device printf or
 * changes to the numerical kernels. Integers are strings to retain all bits;
 * unavailable/nonfinite objectives are null, never invalid JSON. */
#pragma once

#include <cmath>
#include <cstdio>
#include "gmp_bridge.h"
#include "sopt.h"

namespace s23 {

enum SoptPhase { PH_PREPARE, PH_LLL, PH_DEDUPE, PH_DESCENT, PH_REDUCE, PH_COUNT };
static const char *phase_name[PH_COUNT] = {"prepare", "lll", "dedupe", "descent", "reduce"};

template <int L> static void trace_integer(FILE *out, const Int<L> &x)
{
    mpz_t z;
    mpz_init(z);
    to_mpz(z, x);
    gmp_fprintf(out, "\"%Zd\"", z);
    mpz_clear(z);
}

template <int L, int DEG>
static void trace_pair(FILE *out, const Int<L> *f, const Int<L> *g)
{
    fprintf(out, "\"f\":[");
    for (int i = 0; i <= DEG; i++) {
        if (i) fputc(',', out);
        trace_integer(out, f[i]);
    }
    fprintf(out, "],\"g\":[");
    trace_integer(out, g[0]);
    fputc(',', out);
    trace_integer(out, g[1]);
    fputc(']', out);
}

static void trace_objective(FILE *out, double obj)
{
    if (std::isfinite(obj)) fprintf(out, "%.17g", obj);
    else fprintf(out, "null");
}

template <int L, int DEG>
static void sopt_trace(FILE *out, long index, const char *backend, int effort,
                       SoptPhase phase, const SoptPoly<L, DEG> &p, const SoptCand<L, DEG> *c)
{
    fprintf(out, "{\"schema\":1,\"index\":%ld,\"backend\":\"%s\",\"effort\":%d,"
                 "\"phase\":\"%s\",\"status\":%d,\"ncand\":%d,\"objective\":",
            index, backend, effort, phase_name[phase], p.status, p.ncand);
    if (p.status == SOPT_OK) trace_objective(out, p.obj);
    else fprintf(out, "null");
    if (phase == PH_PREPARE && p.status == SOPT_OK) {
        fprintf(out, ",\"skew\":");
        trace_integer(out, p.skew);
        fputc(',', out);
        trace_pair<L, DEG>(out, p.f, p.g);
    }
    if (phase == PH_REDUCE && p.status == SOPT_OK) {
        fputc(',', out);
        trace_pair<L, DEG>(out, p.f_opt, p.g_opt);
    }
    fprintf(out, ",\"candidates\":[");
    for (int j = 0; j < p.ncand; j++) {
        if (j) fputc(',', out);
        fprintf(out, "{\"index\":%d,\"k\":%lld,\"state\":%d", j, (long long)c[j].k, c[j].state);
        if (c[j].state == CAND_LLL || c[j].state == CAND_DUP || c[j].state == CAND_DONE) {
            fprintf(out, ",\"kv\":");
            trace_integer(out, c[j].kv);
            fputc(',', out);
            trace_pair<L, DEG>(out, c[j].f, c[j].g);
        }
        if (c[j].state == CAND_DONE) {
            fprintf(out, ",\"objective\":");
            trace_objective(out, c[j].obj);
        }
        fputc('}', out);
    }
    fprintf(out, "]}\n");
    fflush(out);
}

} // namespace s23
