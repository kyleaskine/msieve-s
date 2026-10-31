/* s23_sopt: CADO-compatible size optimization (degree 5) on the GPU, or on the CPU with
 * the same source.
 *
 *   s23_sopt [-cpu] [-dev N] [-sopteffort E] [-t THREADS] [-batch N] [-slice SECONDS]
 *            [-compare] [-rawstats] [-trace FILE [-trace-index N]] -inputpolys FILE
 *
 * Reads CADO-format polynomials (e.g. msieve -nps output, or fixture_raw.py output) and
 * prints CADO sopt's format ("### Input raw polynomial (i) ###" commented, then
 * "### Size-optimized polynomial (i) ###" with CADO's own stats line), so
 * utils/sort_cado_by_expe.py, the pipeline and stage23_bench/tools/sopt_compare.py read
 * it unchanged. Input is processed and printed in batches of N polynomials, so memory
 * stays bounded and a later failure keeps everything already written. A polynomial the
 * port flags (overflow, loop caps, ...) or cannot take (not degree 5, too wide) is redone
 * by CADO's own size_optimization on the CPU and reported on stderr. A malformed input
 * block stops the run after the polynomials before it, with an error and exit status 1,
 * as CADO's sopt does (blank lines between blocks are skipped, where CADO's reader
 * stops). -compare also runs CADO on every polynomial and reports how many results are
 * identical (development aid).
 *
 * Each batch runs the five phases of include/sopt.h (prepare, lll, dedupe, descent,
 * reduce), each over all its items: polynomials or translation candidates. On the CPU
 * every phase is an OpenMP loop over -t THREADS threads. On the GPU every phase is a work
 * queue that warps take 32 items at a time from; a launch stops taking items after
 * -slice seconds (default 10) and is relaunched until the queue is empty, so no launch
 * comes near the 60 s WSL2 watchdog however long a batch takes. -t THREADS (default 8)
 * also runs the CADO fallbacks and CADO's stats and formatting of each batch. -dev N
 * picks the GPU.
 *
 * Build: make s23_sopt_cpu (g++, no GPU) or make s23_sopt CUDA=120 (nvcc).
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <vector>
#include <chrono>
#include <omp.h>
#include "cado_io.h"
#include "gmp_bridge.h"
#define S23_SOPT_HOST_LOG
#include "sopt.h"
#include "sopt_trace.h"
#include "host_log_check.h"

using namespace s23;

static const int LP = 16;  /* polynomial coefficients: 512 bits */
static const int LL = 128; /* LLL and discriminant: 4096 bits */
static const int DEG = 5;
typedef SoptScratch<LP, LL, DEG> Scratch;
typedef SoptPoly<LP, DEG> Poly;
typedef SoptCand<LP, DEG> Cand;

/* What a phase works on. Per-polynomial phases take item i = polynomial i; the
 * candidate phases take items[i] = the candidate's slot, polynomial * maxk + j. */
struct Ctx {
    Poly *polys;
    Cand *cands;
    const unsigned *items;
    int maxk, effort;
};

template <int PH> S23_HD void run_item(const Ctx &c, unsigned i, Scratch &S)
{
    if (PH == PH_PREPARE) {
        sopt_prepare<LP, LL, DEG>(c.polys[i], c.cands + (size_t)i * c.maxk, c.effort, S);
    } else if (PH == PH_LLL) {
        const unsigned slot = c.items[i];
        sopt_lll<LP, LL, DEG>(c.polys[slot / c.maxk], c.cands[slot], S);
    } else if (PH == PH_DEDUPE) {
        sopt_dedupe<LP, DEG>(c.polys[i], c.cands + (size_t)i * c.maxk);
    } else if (PH == PH_DESCENT) {
        sopt_descent<LP, LL, DEG>(c.cands[c.items[i]], S);
    } else {
        sopt_reduce<LP, DEG>(c.polys[i], c.cands + (size_t)i * c.maxk);
    }
}

#ifdef __CUDACC__
__device__ uint64_t s23_launch_start;

/* launched just before a phase kernel on the same stream: the start of its time slice */
__global__ void stamp_kernel()
{
    s23_launch_start = now_ns();
}

/* Each warp takes 32 consecutive items at a time (so a warp works on neighbouring
 * candidates, mostly of one polynomial) until the queue is empty or the slice is over.
 * A warp's first chunk ignores the deadline only while the launch has taken nothing yet
 * (the queue head is still at `start`), so every launch makes progress whatever the
 * slice, while blocks that start late (a phase with lower occupancy than the worker
 * count assumes runs in waves) stop at once instead of each running one more chunk. */
template <int PH>
__global__ void phase_kernel(Ctx c, unsigned n, unsigned *next, unsigned start, Scratch *scr, uint64_t slice_ns)
{
    const unsigned lane = threadIdx.x & 31;
    Scratch &S = scr[blockIdx.x * blockDim.x + threadIdx.x];
    const uint64_t deadline = s23_launch_start + slice_ns;
    for (bool first = true;; first = false) {
        unsigned base = 0, stop = 0;
        if (lane == 0) {
            const bool exempt = first && *(volatile unsigned *)next == start;
            if (!exempt && now_ns() > deadline)
                stop = 1;
            else {
                base = atomicAdd(next, 32u);
                stop = base >= n;
            }
        }
        base = __shfl_sync(0xffffffffu, base, 0);
        stop = __shfl_sync(0xffffffffu, stop, 0);
        if (stop)
            break;
        if (base + lane < n)
            run_item<PH>(c, base + lane, S);
    }
}

#define CUDA_CHECK(x)                                                                     \
    do {                                                                                  \
        cudaError_t e = (x);                                                              \
        if (e != cudaSuccess) {                                                           \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
            exit(2);                                                                      \
        }                                                                                 \
    } while (0)

static const int BLOCK = 32;

struct Gpu {
    Poly *polys;
    Cand *cands;
    unsigned *items, *next;
    Scratch *scr;
    int workers;
    uint64_t slice_ns;
    int launches;
};

template <int PH> static int resident_blocks()
{
    int nb = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, phase_kernel<PH>, BLOCK, 0));
    return nb;
}

/* workers: enough warps to fill the GPU in the highest-occupancy phase, but their scratch
 * (about 220 KB each) capped at a fraction of the VRAM still free after the candidate
 * slots, so a big card or a GPU shared with another job does not run out (fewer workers
 * only means fewer warps in flight) */
static void gpu_init(Gpu &G, int dev, int batch, int maxk, double slice, double scratch_frac)
{
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int nb = resident_blocks<PH_PREPARE>();
    nb = std::max(nb, resident_blocks<PH_LLL>());
    nb = std::max(nb, resident_blocks<PH_DESCENT>());
    G.workers = std::max(1, nb) * prop.multiProcessorCount * BLOCK;
    G.slice_ns = (uint64_t)(slice * 1e9);
    G.launches = 0;
    const size_t ncand = (size_t)batch * maxk;
    CUDA_CHECK(cudaMalloc(&G.polys, batch * sizeof(Poly)));
    CUDA_CHECK(cudaMalloc(&G.cands, ncand * sizeof(Cand)));
    CUDA_CHECK(cudaMalloc(&G.items, ncand * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&G.next, sizeof(unsigned)));
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t cap = (size_t)(scratch_frac * (double)free_b) / sizeof(Scratch) / BLOCK * BLOCK;
    if ((size_t)G.workers > cap)
        G.workers = (int)std::max<size_t>(cap, BLOCK);
    CUDA_CHECK(cudaMalloc(&G.scr, (size_t)G.workers * sizeof(Scratch)));
    fprintf(stderr, "# s23_sopt: %s, %d workers (%.2f GB scratch), %.2f GB for %zu candidate slots\n", prop.name,
            G.workers, G.workers * sizeof(Scratch) / 1e9, ncand * sizeof(Cand) / 1e9, ncand);
}

/* one phase over n items: relaunched until the queue is empty */
template <int PH> static void gpu_phase(Gpu &G, const Ctx &c, unsigned n)
{
    if (n == 0)
        return;
    CUDA_CHECK(cudaMemset(G.next, 0, sizeof(unsigned)));
    for (unsigned taken = 0; taken < n;) {
        stamp_kernel<<<1, 1>>>();
        phase_kernel<PH><<<G.workers / BLOCK, BLOCK>>>(c, n, G.next, taken, G.scr, G.slice_ns);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&taken, G.next, sizeof(unsigned), cudaMemcpyDeviceToHost));
        G.launches++;
    }
}
#endif

#ifdef __CUDACC__
/* the candidate slots of polynomials still OK whose candidates are in state `want`
 * (all candidates if want < 0) */
static void build_items(std::vector<unsigned> &items, const std::vector<Poly> &P, int n, int maxk,
                        const std::vector<int> *state, int want)
{
    items.clear();
    for (int p = 0; p < n; p++) {
        if (P[p].status != SOPT_OK)
            continue;
        for (int j = 0; j < P[p].ncand; j++) {
            const unsigned slot = (unsigned)(p * maxk + j);
            if (want < 0 || (*state)[slot] == want)
                items.push_back(slot);
        }
    }
}
#endif

int main(int argc, char **argv)
{
    const char *input = NULL;
    const char *trace_path = NULL;
    long trace_index = 0;
    bool trace_index_given = false;
    int effort = 0, threads = 8, use_cpu = 0, compare = 0, raw_stats = 0, batch = 4096, dev = 0;
    bool host_log_fallback = false;
    double slice = 10, scratch_frac = 0.5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-inputpolys") && i + 1 < argc)
            input = argv[++i];
        else if (!strcmp(argv[i], "-sopteffort") && i + 1 < argc)
            effort = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-batch") && i + 1 < argc)
            batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-slice") && i + 1 < argc)
            slice = atof(argv[++i]);
        else if (!strcmp(argv[i], "-scratch") && i + 1 < argc)
            scratch_frac = atof(argv[++i]);
        else if (!strcmp(argv[i], "-dev") && i + 1 < argc)
            dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-cpu"))
            use_cpu = 1;
        else if (!strcmp(argv[i], "-compare"))
            compare = 1;
        else if (!strcmp(argv[i], "-rawstats"))
            raw_stats = 1;
        else if (!strcmp(argv[i], "-trace") && i + 1 < argc)
            trace_path = argv[++i];
        else if (!strcmp(argv[i], "-trace-index") && i + 1 < argc) {
            char *end;
            const char *arg = argv[++i];
            errno = 0;
            trace_index = strtol(arg, &end, 10);
            if (errno || end == arg || *end || trace_index < 0) {
                fprintf(stderr, "s23_sopt: -trace-index must be a nonnegative integer (zero-based)\n");
                return 2;
            }
            trace_index_given = true;
        }
        else {
            fprintf(stderr, "usage: %s [-cpu] [-dev N] [-sopteffort E] [-t THREADS] [-batch N] "
                            "[-slice SECONDS] [-scratch FRACTION] [-compare] [-rawstats] "
                            "[-trace FILE [-trace-index N]] -inputpolys FILE\n",
                    argv[0]);
            return 2;
        }
    }
#ifndef __CUDACC__
    use_cpu = 1; /* host-only build */
#endif
    if (trace_index_given && !trace_path) {
        fprintf(stderr, "s23_sopt: -trace-index requires -trace FILE\n");
        return 2;
    }
    if (threads < 1)
        threads = 1;
    if (effort < 0 || effort > SOPT_MAX_EFFORT) {
        fprintf(stderr, "s23_sopt: effort must be 0..%d (use CADO's sopt above that)\n", SOPT_MAX_EFFORT);
        return 2;
    }
    if (slice <= 0)
        slice = 10;
    if (scratch_frac <= 0 || scratch_frac > 0.9)
        scratch_frac = 0.5;
    /* Reject existing destinations and unusable parent directories before any
     * device setup or input processing. Create nothing until the target batch;
     * fopen(wx) remains authoritative if the filesystem changes meanwhile. */
    FILE *trace = NULL;
    if (trace_path) {
        if (!*trace_path) {
            fprintf(stderr, "s23_sopt: trace path must not be empty\n");
            return 2;
        }
        struct stat info;
        if (lstat(trace_path, &info) == 0) {
            fprintf(stderr, "s23_sopt: trace file already exists: %s\n", trace_path);
            return 2;
        }
        if (errno != ENOENT) {
            perror(trace_path);
            return 2;
        }
        const std::string path(trace_path);
        const auto slash = path.find_last_of('/');
        const std::string parent = slash == std::string::npos ? "." :
                                   slash == 0 ? "/" : path.substr(0, slash);
        if (stat(parent.c_str(), &info) != 0) {
            perror(parent.c_str());
            return 2;
        }
        if (!S_ISDIR(info.st_mode)) {
            fprintf(stderr, "s23_sopt: trace parent is not a directory: %s\n", parent.c_str());
            return 2;
        }
        if (faccessat(AT_FDCWD, parent.c_str(), W_OK | X_OK, AT_EACCESS) != 0) {
            perror(parent.c_str());
            return 2;
        }
    }
    /* GPU candidate slots: at most 1.5 GB and 40% of the VRAM free once the device's
     * stack reservation is made (the CPU path keeps one polynomial's slots per thread) */
    const int maxk = sopt_max_cand(effort);
#ifdef __CUDACC__
    if (!use_cpu) {
        auto inputs = matched_log::probe_inputs();
        auto check = matched_log::identify_host(inputs);
        fprintf(stderr, "# s23_sopt: host log %s; %zu samples, mismatches FMA=%zu scalar=%zu\n",
                matched_log::name(check.mode), check.samples, check.fused_mismatches, check.unfused_mismatches);
        if (check.mode == matched_log::unsupported) {
            fprintf(stderr, "# s23_sopt: unsupported host log/rounding environment; using CADO CPU fallback\n");
            use_cpu = 1;
            host_log_fallback = true;
        } else {
            CUDA_CHECK(cudaSetDevice(dev));
            if (!matched_log::check_device(inputs, check)) return 2;
            fprintf(stderr, "# s23_sopt: device log verified against host (%zu values)\n", check.samples);
        }
    }
    if (!use_cpu) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024));
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const double slot_bytes = std::min(1.5e9, 0.4 * (double)free_b);
        const long max_batch = (long)(slot_bytes / ((double)maxk * sizeof(Cand)));
        if (batch > max_batch)
            batch = (int)max_batch;
    }
#endif
    if (batch < 1)
        batch = 1;
    FILE *in = input ? fopen(input, "r") : stdin;
    if (!in) {
        perror(input);
        return 2;
    }
    omp_set_num_threads(threads);
    printf("# s23_sopt: effort %d, %s, batches of %d\n", effort, use_cpu ? "CPU" : "GPU", batch);

    const size_t nslots = (size_t)batch * maxk;
    std::vector<cio_poly> raws(batch);
    for (auto &r : raws)
        cio_init(&r);
    std::vector<Poly> P(batch);
    std::vector<int> skipped(batch), state(use_cpu ? 0 : nslots), same(batch);
    std::vector<unsigned> items;
    std::vector<char *> text(batch);
    std::vector<size_t> text_len(batch);
    std::vector<Scratch *> scr;
    std::vector<std::vector<Cand>> host_cands; /* CPU path: one polynomial's candidates per thread */
#ifdef __CUDACC__
    Gpu G;
    if (!use_cpu)
        gpu_init(G, dev, batch, maxk, slice, scratch_frac);
#endif
    if (use_cpu) {
        host_cands.assign(threads, std::vector<Cand>(maxk));
        for (int t = 0; t < threads; t++)
            scr.push_back((Scratch *)malloc(sizeof(Scratch)));
    }
    std::vector<cio_poly> opt(threads), ref(threads); /* per host thread */
    for (int t = 0; t < threads; t++) {
        cio_init(&opt[t]);
        cio_init(&ref[t]);
    }

    mpz_t skew;
    mpz_init(skew);
    long total = 0, nfallback = 0, nsame = 0, ncompared = 0, ncand = 0, nkept = 0;
    double secs = 0, phase_secs[PH_COUNT] = {0};
    int rd = 1;
    while (rd > 0) {
        int n = 0;
        while (n < batch && (rd = cio_read_next(in, &raws[n])) > 0) {
            cio_poly &r = raws[n];
            Poly &p = P[n];
            memset(&p, 0, sizeof p);
            bool ok = !host_log_fallback && r.deg == DEG;
            for (int i = 0; i <= DEG && ok; i++)
                ok = from_mpz(p.f[i], r.f[i]);
            ok = ok && from_mpz(p.g[0], r.g[0]) && from_mpz(p.g[1], r.g[1]);
            if (ok) {
                cio_sopt_skew(skew, &r);
                ok = from_mpz(p.skew, skew);
            }
            p.status = ok ? SOPT_OK : SOPT_FAIL;
            skipped[n] = !ok;
            n++;
        }
        if (n == 0)
            break;
        const int trace_slot = trace_path && trace_index >= total && trace_index - total < n
                                   ? (int)(trace_index - total) : -1;
        if (trace_slot >= 0) {
            trace = fopen(trace_path, "wx");
            if (!trace) {
                perror(trace_path);
                return 2;
            }
        }

        auto t0 = std::chrono::steady_clock::now();
        auto lap = [&](int ph) {
            auto t = std::chrono::steady_clock::now();
            phase_secs[ph] += std::chrono::duration<double>(t - t0).count();
            t0 = t;
        };
        const auto start = t0;
#ifdef __CUDACC__
        if (!use_cpu) {
            const Ctx c = {G.polys, G.cands, G.items, maxk, effort};
            auto snapshot = [&](SoptPhase ph, const Poly *host = NULL) {
                if (trace_slot < 0) return;
                Poly p;
                if (host) p = *host;
                else CUDA_CHECK(cudaMemcpy(&p, G.polys + trace_slot, sizeof p, cudaMemcpyDeviceToHost));
                if (p.ncand < 0 || p.ncand > maxk) {
                    fprintf(stderr, "s23_sopt: invalid candidate count in phase trace\n");
                    exit(2);
                }
                std::vector<Cand> candidates(p.ncand);
                if (p.ncand)
                    CUDA_CHECK(cudaMemcpy(candidates.data(), G.cands + (size_t)trace_slot * maxk,
                                          p.ncand * sizeof(Cand), cudaMemcpyDeviceToHost));
                sopt_trace(trace, trace_index, "GPU", effort, ph, p, candidates.data());
            };
            CUDA_CHECK(cudaMemcpy(G.polys, P.data(), n * sizeof(Poly), cudaMemcpyHostToDevice));
            gpu_phase<PH_PREPARE>(G, c, n);
            CUDA_CHECK(cudaMemcpy(P.data(), G.polys, n * sizeof(Poly), cudaMemcpyDeviceToHost));
            snapshot(PH_PREPARE, trace_slot < 0 ? NULL : &P[trace_slot]);
            lap(PH_PREPARE);
            build_items(items, P, n, maxk, NULL, -1);
            ncand += items.size();
            CUDA_CHECK(cudaMemcpy(G.items, items.data(), items.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
            gpu_phase<PH_LLL>(G, c, items.size());
            snapshot(PH_LLL);
            lap(PH_LLL);
            gpu_phase<PH_DEDUPE>(G, c, n);
            CUDA_CHECK(cudaMemcpy(P.data(), G.polys, n * sizeof(Poly), cudaMemcpyDeviceToHost));
            snapshot(PH_DEDUPE, trace_slot < 0 ? NULL : &P[trace_slot]);
            CUDA_CHECK(cudaMemcpy2D(state.data(), sizeof(int), (char *)G.cands + offsetof(Cand, state), sizeof(Cand),
                                    sizeof(int), (size_t)n * maxk, cudaMemcpyDeviceToHost));
            build_items(items, P, n, maxk, &state, CAND_LLL);
            nkept += items.size();
            lap(PH_DEDUPE);
            CUDA_CHECK(cudaMemcpy(G.items, items.data(), items.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
            gpu_phase<PH_DESCENT>(G, c, items.size());
            snapshot(PH_DESCENT);
            lap(PH_DESCENT);
            gpu_phase<PH_REDUCE>(G, c, n);
            CUDA_CHECK(cudaMemcpy(P.data(), G.polys, n * sizeof(Poly), cudaMemcpyDeviceToHost));
            snapshot(PH_REDUCE, trace_slot < 0 ? NULL : &P[trace_slot]);
            lap(PH_REDUCE);
        } else
#endif
        {
            /* CPU: the same five phases, run back to back for one polynomial at a time
             * (polynomials in parallel), so its candidates stay in cache */
            (void)lap;
            long nc = 0, nk = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : nc, nk)
            for (int p = 0; p < n; p++) {
                const int t = omp_get_thread_num();
                Scratch &S = *scr[t];
                Cand *C = host_cands[t].data();
                auto snapshot = [&](SoptPhase ph) {
                    if (p == trace_slot)
                        sopt_trace(trace, trace_index, "CPU", effort, ph, P[p], C);
                };
                sopt_prepare<LP, LL, DEG>(P[p], C, effort, S);
                snapshot(PH_PREPARE);
                if (P[p].status == SOPT_OK) {
                    nc += P[p].ncand;
                    for (int j = 0; j < P[p].ncand; j++)
                        sopt_lll<LP, LL, DEG>(P[p], C[j], S);
                }
                snapshot(PH_LLL);
                sopt_dedupe<LP, DEG>(P[p], C);
                snapshot(PH_DEDUPE);
                if (P[p].status == SOPT_OK)
                    for (int j = 0; j < P[p].ncand; j++)
                        if (C[j].state == CAND_LLL) {
                            nk++;
                            sopt_descent<LP, LL, DEG>(C[j], S);
                        }
                snapshot(PH_DESCENT);
                sopt_reduce<LP, DEG>(P[p], C);
                snapshot(PH_REDUCE);
            }
            ncand += nc;
            nkept += nk;
        }
        secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (trace && ferror(trace)) {
            fprintf(stderr, "s23_sopt: failed writing phase trace\n");
            return 2;
        }

        /* CADO fallbacks, -compare and the stats and text of each result, in parallel;
         * written in input order below */
#pragma omp parallel for schedule(dynamic, 8)
        for (int i = 0; i < n; i++) {
            const int t = omp_get_thread_num();
            cio_poly &o = opt[t];
            same[i] = -1;
            if (P[i].status != SOPT_OK) {
                cio_cado_sopt(&o, &raws[i], effort);
            } else {
                mpz_set(o.n, raws[i].n);
                o.deg = DEG;
                for (int k = 0; k <= DEG; k++)
                    to_mpz(o.f[k], P[i].f_opt[k]);
                to_mpz(o.g[0], P[i].g_opt[0]);
                to_mpz(o.g[1], P[i].g_opt[1]);
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
            if (P[i].status != SOPT_OK) {
                nfallback++;
                if (!host_log_fallback)
                    fprintf(stderr, "# s23_sopt: poly %ld redone by CADO (%s)\n", total,
                            skipped[i] ? "not degree 5 or too wide" : "port flagged it");
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
    if (host_log_fallback)
        fprintf(stderr, " (unsupported host log)");
    if (compare)
        fprintf(stderr, "; identical to CADO: %ld of %ld", nsame, ncompared);
    fprintf(stderr, "\n# s23_sopt: %ld candidates (%.1f per poly), %ld kept after dedupe", ncand,
            (double)ncand / (total ? total : 1), nkept);
#ifdef __CUDACC__
    if (!use_cpu) {
        fprintf(stderr, "; phases:");
        for (int ph = 0; ph < PH_COUNT; ph++)
            fprintf(stderr, " %s %.2f s", phase_name[ph], phase_secs[ph]);
        fprintf(stderr, "; %d launches", G.launches);
    }
#endif
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
    if (trace) {
        if (fclose(trace)) {
            perror(trace_path);
            return 2;
        }
    }
    if (trace_path && trace_index >= total) {
        fprintf(stderr, "s23_sopt: trace index %ld not found in %ld inputs\n", trace_index, total);
        return 2;
    }
    return read_error ? 1 : 0;
}
