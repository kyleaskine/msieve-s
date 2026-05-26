#include "collision_common.cuh"
#include "bench_gpu.cuh"
#include "bench_cpu.h"
#include "verify_qsort.h"
#include "data_gen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

struct cell_t {
    uint32 n;
    uint32 bits;
};

struct result_t {
    uint32 n;
    uint32 bits;
    uint32 cpu_hits;
    uint32 gpu_hits;
    uint32 qsort_hits;
    uint32 gpu_msieve_hits;
    uint32 qsort_msieve_hits;
    double cpu_wall;
    double gpu_wall;
    gpu_timings_t gpu_t;
    int ok;
};

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --n N            number of keys (default 1000000)\n"
        "  --bits B         key width in bits (default 56)\n"
        "  --seed S         32-bit seed for MWC PRNG (default 0x11111)\n"
        "  --sweep          run the 11-cell sweep matrix\n"
        "  --mode MODE      cpu|gpu|both (default both)\n"
        "  --repeats R      runs per cell (default 3)\n"
        "  --warmup         drop first run from median\n"
        "  --no-verify      skip slow qsort verifier\n"
        "  --bucket-hash    hash key into bucket id (Knuth mix) instead of low bits\n"
        "  --threads N      CPU thread count for parallel collision_ref.c (default 1)\n"
        "  --verify-values  also verify msieve-style packed value hit semantics\n"
        "  --inject-dups N  inject N duplicate-key pairs that pass msieve final checks\n"
        "  --pshift S       low-bit width of packed p values (default 16)\n"
        "  --csv FILE       write per-cell CSV results\n",
        argv0);
}

static double median3(double a, double b, double c) {
    double v[3] = {a, b, c};
    std::sort(v, v + 3);
    return v[1];
}

static double median_of(double *v, int n) {
    std::sort(v, v + n);
    return v[n / 2];
}

/* collision_ref.c's internal stripping caps key_bits at 32 + log2(num_buckets=8) = 40.
   Above that its assertion fires. Skip the CPU path; verify GPU against qsort. */
#define CPU_MAX_BITS 40u

static int run_cell(uint32 n, uint32 bits, uint32 seed, uint32 carry,
                    int do_cpu, int do_gpu, int do_qsort, int bucket_hash,
                    int repeats, int warmup, int cpu_threads,
                    int verify_values, uint32 inject_dups, uint32 pshift,
                    gpu_ctx_t *gpu_ctx,
                    result_t *out, FILE *csv) {
    if (do_cpu && bits > CPU_MAX_BITS) {
        fprintf(stderr, "[note] bits=%u > %u — skipping CPU (collision_ref.c limit)\n",
                bits, CPU_MAX_BITS);
        do_cpu = 0;
    }

    uint64_t *keys = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    uint32_t *values = NULL;
    generate_keys(keys, n, bits, seed, carry);
    if (verify_values) {
        values = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
        generate_msieve_values(values, n, pshift,
                               seed ^ 0x9e3779b9u, carry ^ 0x7f4a7c15u);
        inject_msieve_duplicates(keys, values, n, inject_dups, pshift);
    }

    out->n = n;
    out->bits = bits;
    out->cpu_hits = 0;
    out->gpu_hits = 0;
    out->qsort_hits = 0;
    out->gpu_msieve_hits = 0;
    out->qsort_msieve_hits = 0;
    out->cpu_wall = 0.0;
    out->gpu_wall = 0.0;
    memset(&out->gpu_t, 0, sizeof(out->gpu_t));

    int total_runs = repeats + (warmup ? 1 : 0);
    double *cpu_walls = (double *)calloc(total_runs, sizeof(double));
    double *gpu_walls = (double *)calloc(total_runs, sizeof(double));
    gpu_timings_t last_gpu_t;
    memset(&last_gpu_t, 0, sizeof(last_gpu_t));
    uint32 last_cpu_hits = 0, last_gpu_hits = 0;
    uint32 last_gpu_msieve_hits = 0;

    for (int r = 0; r < total_runs; r++) {
        if (do_cpu) {
            double t0 = wall_seconds();
            if (cpu_threads > 1) {
                last_cpu_hits = omp_collision_search(keys, n, bits, 0, cpu_threads);
            } else {
                last_cpu_hits = cpu_collision_search(keys, n, bits, 0);
            }
            cpu_walls[r] = wall_seconds() - t0;
        }
        if (do_gpu) {
            double t0 = wall_seconds();
            if (verify_values) {
                last_gpu_hits = gpu_collision_search_with_values(
                        gpu_ctx, keys, values, n, bits, bucket_hash,
                        pshift, &last_gpu_t, &last_gpu_msieve_hits);
            } else {
                last_gpu_hits = gpu_collision_search(gpu_ctx, keys, n, bits,
                                                      bucket_hash, &last_gpu_t);
            }
            gpu_walls[r] = wall_seconds() - t0;
        }
    }

    out->cpu_hits = last_cpu_hits;
    out->gpu_hits = last_gpu_hits;
    out->gpu_msieve_hits = last_gpu_msieve_hits;
    out->gpu_t = last_gpu_t;

    /* Drop warmup runs from the median. */
    int start = warmup ? 1 : 0;
    if (do_cpu) out->cpu_wall = median_of(cpu_walls + start, repeats);
    if (do_gpu) out->gpu_wall = median_of(gpu_walls + start, repeats);

    if (do_qsort) {
        out->qsort_hits = qsort_collision_count(keys, n);
        if (verify_values)
            out->qsort_msieve_hits =
                    qsort_msieve_hit_count(keys, values, n, pshift);
    }

    int ok = 1;
    if (do_cpu && do_qsort && out->cpu_hits != out->qsort_hits) ok = 0;
    if (do_gpu && do_qsort && out->gpu_hits != out->qsort_hits) ok = 0;
    if (do_cpu && do_gpu  && out->cpu_hits != out->gpu_hits)   ok = 0;
    if (verify_values && do_gpu && do_qsort &&
        out->gpu_msieve_hits != out->qsort_msieve_hits) ok = 0;
    out->ok = ok;

    /* Stdout report. */
    printf("N=%-9u bits=%-3u  ", n, bits);
    if (do_cpu)   printf("CPU_hits=%-8u t=%7.3fs  ", out->cpu_hits, out->cpu_wall);
    if (do_gpu)   printf("GPU_hits=%-8u t=%7.3fs  ", out->gpu_hits, out->gpu_wall);
    if (do_qsort) printf("qsort=%-8u  ", out->qsort_hits);
    if (verify_values) {
        if (do_gpu)   printf("GPU_msieve=%-6u  ", out->gpu_msieve_hits);
        if (do_qsort) printf("qsort_msieve=%-6u  ", out->qsort_msieve_hits);
    }
    if (do_cpu && do_gpu && out->gpu_wall > 0) {
        printf("speedup=%5.2fx  ", out->cpu_wall / out->gpu_wall);
    }
    printf("[%s]\n", ok ? "OK" : "MISMATCH");

    if (do_gpu) {
        printf("    GPU phases:  H2D=%.2fms  A=%.2fms  B=%.2fms  C=%.2fms  D2H=%.2fms  total=%.2fms  cand=%u  bucket_max=%u",
               out->gpu_t.ms_h2d, out->gpu_t.ms_phase_a, out->gpu_t.ms_phase_b,
               out->gpu_t.ms_phase_c, out->gpu_t.ms_d2h, out->gpu_t.ms_total,
               out->gpu_t.candidate_count, out->gpu_t.bucket_max);
        if (verify_values) {
            printf("  matched_values=%u  msieve_hits=%u",
                   out->gpu_t.value_match_count,
                   out->gpu_t.msieve_hit_count);
        }
        printf("\n");
    }

    if (csv) {
        fprintf(csv, "%u,%u,%u,%u,%u,%u,%u,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%u,%u,%u,%d\n",
                n, bits, out->cpu_hits, out->gpu_hits, out->qsort_hits,
                out->gpu_msieve_hits, out->qsort_msieve_hits,
                out->cpu_wall, out->gpu_wall,
                out->gpu_t.ms_phase_a, out->gpu_t.ms_phase_b,
                out->gpu_t.ms_phase_c, out->gpu_t.ms_total,
                out->gpu_t.candidate_count, out->gpu_t.bucket_max,
                out->gpu_t.value_match_count, ok);
        fflush(csv);
    }

    free(keys);
    free(values);
    free(cpu_walls);
    free(gpu_walls);
    return ok;
}

int main(int argc, char **argv) {
    uint32 n = 1000000;
    uint32 bits = 56;
    uint32 seed  = 0x11111;
    uint32 carry = 0x22222;
    int sweep = 0;
    int do_cpu = 1, do_gpu = 1;
    int do_qsort = 1;
    int bucket_hash = 0;
    int repeats = 3;
    int warmup = 0;
    int cpu_threads = 1;
    int verify_values = 0;
    uint32 inject_dups = 0;
    uint32 pshift = 16;
    const char *csv_path = NULL;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--n")       && i + 1 < argc) n = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--bits")    && i + 1 < argc) bits = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--seed")    && i + 1 < argc) seed = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--sweep"))                   sweep = 1;
        else if (!strcmp(a, "--mode")    && i + 1 < argc) {
            ++i;
            do_cpu = !strcmp(argv[i], "cpu") || !strcmp(argv[i], "both");
            do_gpu = !strcmp(argv[i], "gpu") || !strcmp(argv[i], "both");
        }
        else if (!strcmp(a, "--repeats") && i + 1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(a, "--warmup"))                  warmup = 1;
        else if (!strcmp(a, "--no-verify"))               do_qsort = 0;
        else if (!strcmp(a, "--bucket-hash"))             bucket_hash = 1;
        else if (!strcmp(a, "--threads") && i + 1 < argc) cpu_threads = atoi(argv[++i]);
        else if (!strcmp(a, "--verify-values"))           verify_values = 1;
        else if (!strcmp(a, "--inject-dups") && i + 1 < argc) {
            inject_dups = strtoul(argv[++i], NULL, 0);
            verify_values = 1;
        }
        else if (!strcmp(a, "--pshift") && i + 1 < argc) pshift = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(a, "--csv")     && i + 1 < argc) csv_path = argv[++i];
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a); usage(argv[0]); return 2; }
    }

    if (bits < LOG2_NUM_BUCKETS + 6u) {
        fprintf(stderr, "bits=%u too small (need >= %u for NUM_BUCKETS=%u)\n",
                bits, LOG2_NUM_BUCKETS + 6u, NUM_BUCKETS);
        return 2;
    }
    if (bits > 64) {
        fprintf(stderr, "bits=%u > 64 not supported\n", bits);
        return 2;
    }
    if (pshift == 0 || pshift >= 32) {
        fprintf(stderr, "pshift=%u not supported (need 1..31)\n", pshift);
        return 2;
    }

    cell_t sweep_cells[] = {
        {  1000000u, 40u },
        {  1000000u, 56u },
        { 10000000u, 48u },
        { 10000000u, 56u },
        { 10000000u, 64u },
        { 50000000u, 48u },
        { 50000000u, 56u },
        { 50000000u, 64u },
        {100000000u, 56u },
    };
    int num_cells = sweep ? (int)(sizeof(sweep_cells) / sizeof(*sweep_cells)) : 1;
    if (!sweep) {
        sweep_cells[0].n = n;
        sweep_cells[0].bits = bits;
    }

    uint32 max_n = 0, max_bits = 0;
    for (int i = 0; i < num_cells; i++) {
        if (sweep_cells[i].n    > max_n)    max_n    = sweep_cells[i].n;
        if (sweep_cells[i].bits > max_bits) max_bits = sweep_cells[i].bits;
    }

    gpu_ctx_t *gpu_ctx = do_gpu ? gpu_ctx_init(max_n, max_bits) : NULL;

    FILE *csv = NULL;
    if (csv_path) {
        csv = fopen(csv_path, "w");
        if (!csv) { perror(csv_path); return 1; }
        fprintf(csv,
            "n,bits,cpu_hits,gpu_hits,qsort_hits,gpu_msieve_hits,"
            "qsort_msieve_hits,cpu_wall_s,gpu_wall_s,"
            "phase_a_ms,phase_b_ms,phase_c_ms,gpu_total_ms,"
            "candidate_count,bucket_max,value_match_count,ok\n");
    }

    int all_ok = 1;
    for (int i = 0; i < num_cells; i++) {
        result_t r;
        int ok = run_cell(sweep_cells[i].n, sweep_cells[i].bits,
                          seed, carry,
                          do_cpu, do_gpu, do_qsort, bucket_hash,
                          repeats, warmup, cpu_threads,
                          verify_values, inject_dups, pshift,
                          gpu_ctx, &r, csv);
        if (!ok) all_ok = 0;
    }

    if (csv) fclose(csv);
    if (gpu_ctx) gpu_ctx_free(gpu_ctx);

    printf("\n%s\n", all_ok ? "ALL CELLS OK" : "MISMATCH(es) DETECTED");
    return all_ok ? 0 : 1;
}
