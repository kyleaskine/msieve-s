/* s23_sopt: CADO-compatible size optimization (degree 5) on the GPU, or on the CPU with
 * the same source.
 *
 *   s23_sopt [-cpu] [-dev N] [-sopteffort E] [-t THREADS] [-batch N] [-budget SECONDS]
 *            [-compare] [-rawstats] [-force-gpu-effort] -inputpolys FILE
 *
 * Reads CADO-format polynomials (e.g. msieve -nps output, or fixture_raw.py output) and
 * prints CADO sopt's format ("### Input raw polynomial (i) ###" commented, then
 * "### Size-optimized polynomial (i) ###" with CADO's own stats line), so
 * utils/sort_cado_by_expe.py, the pipeline and stage23_bench/tools/sopt_compare.py read
 * it unchanged. Input is processed and printed in batches of N polynomials, so memory
 * stays bounded and a later failure keeps everything already written. A polynomial the
 * port flags (overflow, loop caps, ...), runs out of time on, or cannot take (not degree
 * 5, too wide) is redone by CADO's own size_optimization on the CPU and reported on
 * stderr. A malformed input block stops the run after the polynomials before it, with an
 * error and exit status 1, as CADO's sopt does (blank lines between blocks are skipped,
 * where CADO's reader stops). -compare also runs CADO on every polynomial and reports how
 * many results are identical (development aid).
 *
 * -t THREADS (default 8) host threads run the CPU path, the CADO fallbacks and the
 * stats and formatting of each batch (CADO's code, which CADO's own polyselect also
 * runs from several threads). -dev N picks the GPU.
 *
 * Time budget (-budget, default 40 s on the GPU, none on the CPU): a polynomial still
 * running that long after its launch (or CPU batch) started gives up at its next
 * translation candidate and is redone by CADO, so no launch reaches the 60 s WSL2
 * watchdog. On the GPU, one thread runs one polynomial's whole search, which at effort > 0
 * takes far longer than the budget; such runs are refused unless -force-gpu-effort is
 * given (until the work is split per translation candidate).
 *
 * Build: make s23_sopt_cpu (g++, no GPU) or make s23_sopt CUDA=120 (nvcc).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <omp.h>
#include "cado_io.h"
#include "gmp_bridge.h"
#include "sopt.h"

using namespace s23;

static const int LP = 16;  /* polynomial coefficients: 512 bits */
static const int LL = 128; /* LLL and discriminant: 4096 bits */
static const int DEG = 5;
typedef SoptScratch<LP, LL, DEG> Scratch;

struct Job {
    Int<LP> f[DEG + 1], g[2], skew;
    int skip; /* could not be converted: CADO redoes it, nothing is run */
};
struct Out {
    int status;
    double exp_e;
    Int<LP> f[DEG + 1], g[2];
};

S23_HD void run_one(Out &o, const Job &j, int effort, uint64_t deadline, Scratch &S)
{
    if (j.skip) {
        o.status = SOPT_FAIL;
        return;
    }
    o.status = size_optimize<LP, LL, DEG>(o.f, o.g, o.exp_e, j.f, j.g, j.skew, effort, deadline, S);
}

#ifdef __CUDACC__
__device__ uint64_t s23_launch_start;

/* launched just before sopt_kernel on the same stream: the start of the time budget */
__global__ void stamp_kernel()
{
    s23_launch_start = now_ns();
}

__global__ void sopt_kernel(const Job *jobs, Out *outs, Scratch *scratch, int n, int effort, uint64_t budget_ns)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t deadline = budget_ns ? s23_launch_start + budget_ns : 0;
    if (i < n)
        run_one(outs[i], jobs[i], effort, deadline, scratch[i]);
}

#define CUDA_CHECK(x)                                                                     \
    do {                                                                                  \
        cudaError_t e = (x);                                                              \
        if (e != cudaSuccess) {                                                           \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
            exit(2);                                                                      \
        }                                                                                 \
    } while (0)

struct Gpu {
    Job *jobs;
    Out *outs;
    Scratch *scr;
};

static void gpu_init(Gpu &G, int dev, int batch)
{
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024));
    CUDA_CHECK(cudaMalloc(&G.jobs, batch * sizeof(Job)));
    CUDA_CHECK(cudaMalloc(&G.outs, batch * sizeof(Out)));
    CUDA_CHECK(cudaMalloc(&G.scr, batch * sizeof(Scratch)));
}

static void gpu_run(Gpu &G, Out *outs, const Job *jobs, int n, int effort, uint64_t budget_ns)
{
    CUDA_CHECK(cudaMemcpy(G.jobs, jobs, n * sizeof(Job), cudaMemcpyHostToDevice));
    stamp_kernel<<<1, 1>>>();
    sopt_kernel<<<(n + 31) / 32, 32>>>(G.jobs, G.outs, G.scr, n, effort, budget_ns);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(outs, G.outs, n * sizeof(Out), cudaMemcpyDeviceToHost));
}
#endif

static void cpu_run(std::vector<Scratch *> &scr, Out *outs, const Job *jobs, int n, int effort, uint64_t budget_ns)
{
    const uint64_t deadline = budget_ns ? now_ns() + budget_ns : 0;
#pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n; i++)
        run_one(outs[i], jobs[i], effort, deadline, *scr[omp_get_thread_num()]);
}

int main(int argc, char **argv)
{
    const char *input = NULL;
    int effort = 0, threads = 8, use_cpu = 0, compare = 0, raw_stats = 0, batch = 1024, force_effort = 0, dev = 0;
    double budget = -1; /* default: 40 s on the GPU, none on the CPU */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-inputpolys") && i + 1 < argc)
            input = argv[++i];
        else if (!strcmp(argv[i], "-sopteffort") && i + 1 < argc)
            effort = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-batch") && i + 1 < argc)
            batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-budget") && i + 1 < argc)
            budget = atof(argv[++i]);
        else if (!strcmp(argv[i], "-dev") && i + 1 < argc)
            dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-cpu"))
            use_cpu = 1;
        else if (!strcmp(argv[i], "-compare"))
            compare = 1;
        else if (!strcmp(argv[i], "-rawstats"))
            raw_stats = 1;
        else if (!strcmp(argv[i], "-force-gpu-effort"))
            force_effort = 1;
        else {
            fprintf(stderr, "usage: %s [-cpu] [-dev N] [-sopteffort E] [-t THREADS] [-batch N] "
                            "[-budget SECONDS] [-compare] [-rawstats] [-force-gpu-effort] -inputpolys FILE\n",
                    argv[0]);
            return 2;
        }
    }
#ifndef __CUDACC__
    use_cpu = 1; /* host-only build */
#endif
    if (batch < 1)
        batch = 1;
    if (threads < 1)
        threads = 1;
    if (effort < 0 || effort > SOPT_MAX_EFFORT) {
        fprintf(stderr, "s23_sopt: effort must be 0..%d (use CADO's sopt above that)\n", SOPT_MAX_EFFORT);
        return 2;
    }
    if (budget < 0)
        budget = use_cpu ? 0 : 40;
    const uint64_t budget_ns = (uint64_t)(budget * 1e9);
    if (!use_cpu && effort > 0 && !force_effort) {
        fprintf(stderr, "s23_sopt: effort %d on the GPU would run one polynomial per thread far past the "
                        "60 s watchdog; use -cpu (or -force-gpu-effort)\n", effort);
        return 2;
    }
    FILE *in = input ? fopen(input, "r") : stdin;
    if (!in) {
        perror(input);
        return 2;
    }
    omp_set_num_threads(threads);
    printf("# s23_sopt: effort %d, %s, batches of %d\n", effort, use_cpu ? "CPU" : "GPU", batch);

    std::vector<cio_poly> raws(batch);
    for (auto &r : raws)
        cio_init(&r);
    std::vector<Job> jobs(batch);
    std::vector<Out> outs(batch);
    std::vector<char *> text(batch);
    std::vector<size_t> text_len(batch);
    std::vector<int> same(batch);
    std::vector<Scratch *> scr;
#ifdef __CUDACC__
    Gpu G;
    if (!use_cpu)
        gpu_init(G, dev, batch);
#endif
    if (use_cpu)
        for (int t = 0; t < threads; t++)
            scr.push_back((Scratch *)malloc(sizeof(Scratch)));
    std::vector<cio_poly> opt(threads), ref(threads); /* per host thread */
    for (int t = 0; t < threads; t++) {
        cio_init(&opt[t]);
        cio_init(&ref[t]);
    }

    mpz_t skew;
    mpz_init(skew);
    long total = 0, nfallback = 0, ntimeout = 0, nsame = 0, ncompared = 0;
    double secs = 0;
    int rd = 1;
    while (rd > 0) {
        int n = 0;
        while (n < batch && (rd = cio_read_next(in, &raws[n])) > 0) {
            cio_poly &p = raws[n];
            Job &j = jobs[n];
            memset(&j, 0, sizeof j);
            bool ok = p.deg == DEG;
            for (int i = 0; i <= DEG && ok; i++)
                ok = from_mpz(j.f[i], p.f[i]);
            ok = ok && from_mpz(j.g[0], p.g[0]) && from_mpz(j.g[1], p.g[1]);
            if (ok) {
                cio_sopt_skew(skew, &p);
                ok = from_mpz(j.skew, skew);
            }
            j.skip = !ok;
            n++;
        }
        if (n == 0)
            break;

        auto t0 = std::chrono::steady_clock::now();
#ifdef __CUDACC__
        if (!use_cpu)
            gpu_run(G, outs.data(), jobs.data(), n, effort, budget_ns);
        else
#endif
            cpu_run(scr, outs.data(), jobs.data(), n, effort, budget_ns);
        secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        /* CADO fallbacks, -compare and the stats and text of each result, in parallel;
         * written in input order below */
#pragma omp parallel for schedule(dynamic, 8)
        for (int i = 0; i < n; i++) {
            const int t = omp_get_thread_num();
            cio_poly &o = opt[t];
            same[i] = -1;
            if (outs[i].status != SOPT_OK) {
                cio_cado_sopt(&o, &raws[i], effort);
            } else {
                mpz_set(o.n, raws[i].n);
                o.deg = DEG;
                for (int k = 0; k <= DEG; k++)
                    to_mpz(o.f[k], outs[i].f[k]);
                to_mpz(o.g[0], outs[i].g[0]);
                to_mpz(o.g[1], outs[i].g[1]);
                if (compare) {
                    cio_cado_sopt(&ref[t], &raws[i], effort);
                    bool s = mpz_cmp(ref[t].g[0], o.g[0]) == 0 && mpz_cmp(ref[t].g[1], o.g[1]) == 0;
                    for (int k = 0; k <= DEG && s; k++)
                        s = mpz_cmp(ref[t].f[k], o.f[k]) == 0;
                    same[i] = s;
                }
            }
            FILE *ms = open_memstream(&text[i], &text_len[i]);
            if (!ms) {
                perror("open_memstream");
                exit(2);
            }
            cio_print_pair(ms, (unsigned)(total + i), &raws[i], &o, raw_stats);
            fclose(ms);
        }

        for (int i = 0; i < n; i++, total++) {
            if (outs[i].status != SOPT_OK) {
                nfallback++;
                ntimeout += outs[i].status == SOPT_TIMEOUT;
                fprintf(stderr, "# s23_sopt: poly %ld redone by CADO (%s)\n", total,
                        jobs[i].skip                         ? "not degree 5 or too wide"
                        : outs[i].status == SOPT_TIMEOUT ? "out of time"
                                                         : "port flagged it");
            }
            if (same[i] >= 0) {
                ncompared++;
                nsame += same[i];
                if (!same[i])
                    fprintf(stderr, "# s23_sopt: poly %ld differs from CADO\n", total);
            }
            fwrite(text[i], 1, text_len[i], stdout);
            free(text[i]);
        }
        fflush(stdout);
    }
    const bool read_error = rd < 0;
    if (input)
        fclose(in);

    fprintf(stderr, "# s23_sopt: %ld polynomials, %.2f s optimizing (%.3f ms each); %ld redone by CADO",
            total, secs, 1000.0 * secs / (total ? total : 1), nfallback);
    if (ntimeout)
        fprintf(stderr, " (%ld out of time)", ntimeout);
    if (compare)
        fprintf(stderr, "; identical to CADO: %ld of %ld", nsame, ncompared);
    fprintf(stderr, "\n");
    if (read_error)
        fprintf(stderr, "s23_sopt: bad input after polynomial %ld; stopped there\n", total);

    for (auto &r : raws)
        cio_clear(&r);
    for (auto *s : scr)
        free(s);
    for (int t = 0; t < threads; t++) {
        cio_clear(&opt[t]);
        cio_clear(&ref[t]);
    }
    mpz_clear(skew);
    return read_error ? 1 : 0;
}
