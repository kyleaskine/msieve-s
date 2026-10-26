/* Optional CUDA alpha/size work: setup/allocation failures leave the input cells
 * untouched so the caller can use CPU screening or final refinement. */
#pragma once
#include "rscreen.h"

struct RsScreenDeviceCell {
    int64_t u, v, t;
    double L, A;
    unsigned failed;
    s23::Int<RS_LP> f[6];
};

__global__ void rs_screen_prepare(const RsizePoly<RS_LP> *r, RsScreenDeviceCell *cells, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    auto &c = cells[i];
    bool ok = rs_rotate(c.f, *r, c.u, c.v);
    double log_content = 0;
    if (ok) c.L = rs_screen_size(c.f, c.t, ok);
    if (ok) ok = rs_screen_primitive(c.f, log_content);
    c.L -= log_content;
    c.failed = !ok || !isfinite(c.L);
}

__global__ void rs_screen_alpha(RsScreenDeviceCell *cells, const uint32_t *primes,
                               const double *logs, int np)
{
    auto &c = cells[blockIdx.x];
    if (c.failed) return;
    __shared__ double terms[RS_SCREEN_MAX_PRIMES];
    __shared__ unsigned failed;
    if (!threadIdx.x) failed = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < np; j += blockDim.x) {
        bool ok = true;
        double v = rs_screen_valuation(c.f, primes[j], ok);
        terms[j] = (1.0 / (primes[j] - 1) - v) * logs[j];
        if (!ok || !isfinite(terms[j])) atomicOr(&failed, 1u);
    }
    __syncthreads();
    if (!threadIdx.x) {
        double a = terms[0];
        for (int j = 1; j < np; ++j) a += terms[j];
        c.A = a;
        c.failed = failed;
    }
}

static bool rs_screen_cuda_ok(cudaError_t error)
{
    if (error == cudaSuccess) return true;
    fprintf(stderr, "s23_ropt: GPU alpha/size evaluation unavailable (%s); using CPU\n",
            cudaGetErrorString(error));
    cudaGetLastError(); // clear a recoverable allocation/configuration error
    return false;
}

// Reserve before the sieve, including replay. Failure only disables this screen.
static bool rs_screen_gpu_start(int dev)
{
    if (!rs_screen_cuda_ok(cudaSetDevice(dev)) || !rs_screen_cuda_ok(cudaFree(0))) return false;
    size_t current = 0;
    return rs_screen_cuda_ok(cudaDeviceGetLimit(&current, cudaLimitStackSize)) &&
           (current >= 32 * 1024 || rs_screen_cuda_ok(cudaDeviceSetLimit(cudaLimitStackSize, 32 * 1024)));
}

// Owned by a process/session, not a seed. Scratch and prime tables survive
// successive jobs. Only one job uses the workspace at a time.
struct RsScreenWorkspace {
    RsizePoly<RS_LP> *r = nullptr;
    RsScreenDeviceCell *cells = nullptr;
    uint32_t *primes = nullptr;
    double *logs = nullptr;
    std::vector<RsScreenDeviceCell> host;
    int device = -1, sms = 0, bound = 0;
    size_t capacity = 0, setups = 0, allocations = 0, prime_uploads = 0;
    RsScreenWorkspace() = default;
    RsScreenWorkspace(const RsScreenWorkspace &) = delete;
    RsScreenWorkspace &operator=(const RsScreenWorkspace &) = delete;

    void clear_buffers() {
        if (device >= 0) cudaSetDevice(device);
        if (logs) cudaFree(logs);
        if (primes) cudaFree(primes);
        if (cells) cudaFree(cells);
        if (r) cudaFree(r);
        r = nullptr; cells = nullptr; primes = nullptr; logs = nullptr;
        capacity = 0; bound = 0;
        host.clear();
    }
    ~RsScreenWorkspace() { clear_buffers(); }

    bool start(int dev) {
        if (device == dev && device >= 0) return true;
        clear_buffers();
        device = -1;
        if (!rs_screen_gpu_start(dev) ||
            !rs_screen_cuda_ok(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev))) return false;
        device = dev;
        ++setups;
        return true;
    }

    bool reserve(size_t count, int new_bound, const std::vector<uint32_t> &ps) {
        if (device < 0 || !rs_screen_cuda_ok(cudaSetDevice(device))) return false;
        const size_t needed = std::min(count, std::max<size_t>(2048, size_t(sms) * 4 * 64));
        if (!needed) return true;
        if (!r || capacity < needed) {
            clear_buffers();
            if (!rs_screen_cuda_ok(cudaMalloc(&r, sizeof(*r))) ||
                !rs_screen_cuda_ok(cudaMalloc(&cells, needed * sizeof(*cells))) ||
                !rs_screen_cuda_ok(cudaMalloc(&primes, RS_SCREEN_MAX_PRIMES * sizeof(*primes))) ||
                !rs_screen_cuda_ok(cudaMalloc(&logs, RS_SCREEN_MAX_PRIMES * sizeof(*logs)))) {
                clear_buffers();
                return false;
            }
            capacity = needed;
            host.resize(capacity);
            ++allocations;
        }
        if (bound != new_bound) {
            std::vector<double> ls;
            for (auto p : ps) ls.push_back(log(double(p)));
            if (!rs_screen_cuda_ok(cudaMemcpy(primes, ps.data(), ps.size() * sizeof(*primes), cudaMemcpyHostToDevice)) ||
                !rs_screen_cuda_ok(cudaMemcpy(logs, ls.data(), ls.size() * sizeof(*logs), cudaMemcpyHostToDevice))) {
                clear_buffers();
                return false;
            }
            bound = new_bound;
            ++prime_uploads;
        }
        return true;
    }
};

static bool rs_screen_gpu(std::vector<RsCandidate> &cells, const RsizePoly<RS_LP> &r,
                          const cio_poly &p, int bound, double aw, int dev, bool check,
                          size_t &fallbacks, RsScreenWorkspace &workspace, bool final_refine = false)
{
    std::vector<uint32_t> primes;
    if (!rs_screen_primes(bound, primes)) {
        fprintf(stderr, "s23_ropt: GPU alpha bound/capacity unsupported; using CPU\n");
        return false;
    }
    if (workspace.device != dev || !workspace.reserve(cells.size(), bound, primes)) return false;
    if (cells.empty()) return true;
    const size_t batch = workspace.capacity;
    auto dr = workspace.r;
    auto dc = workspace.cells;
    auto dp = workspace.primes;
    auto dl = workspace.logs;
    if (!rs_screen_cuda_ok(cudaMemcpy(dr, &r, sizeof r, cudaMemcpyHostToDevice))) return false;
    // Commit only on success: a later batch failure must not change the CPU starts.
    auto screened = cells;
    auto &h = workspace.host;
    fallbacks = 0;
    double max_L = 0, max_A = 0, max_key = 0;
    size_t different_t = 0;
    for (size_t start = 0; start < screened.size(); start += batch) {
        int n = int(std::min(batch, screened.size() - start));
        for (int j = 0; j < n; ++j) {
            auto &c = screened[start+j];
            h[j] = {};
            h[j].u = c.h.u; h[j].v = c.h.v; h[j].t = c.t;
        }
        if (!rs_screen_cuda_ok(cudaMemcpy(dc, h.data(), n * sizeof(*dc), cudaMemcpyHostToDevice))) return false;
        rs_screen_prepare<<<(n + 63) / 64, 64>>>(dr, dc, n);
        if (!rs_screen_cuda_ok(cudaGetLastError())) return false;
        rs_screen_alpha<<<n, 128>>>(dc, dp, dl, int(primes.size()));
        if (!rs_screen_cuda_ok(cudaGetLastError()) ||
            !rs_screen_cuda_ok(cudaMemcpy(h.data(), dc, n * sizeof(*dc), cudaMemcpyDeviceToHost))) return false;
        unsigned nf = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+:nf,different_t) reduction(max:max_L,max_A,max_key)
        for (int j = 0; j < n; ++j) {
            auto &c = screened[start+j];
            // Save the CPU start before accepting device output. Every final
            // candidate gets the CPU size search, not only those near a cut:
            // small libm differences can send that search to another basin.
            int64_t canonical_t = c.t;
            double canonical_L = c.L;
            bool canonical_ready = c.screen_host;
            c.L = h[j].L; c.A = h[j].A;
            c.key = rs_screen_key(c.L, c.A, aw);
            if (h[j].failed || check) {
                int64_t reference_t = c.t;
                const RsExact x = rs_exact_cell(p, r, c.h.u, c.h.v, reference_t, bound);
                const double reference_key = rs_screen_key(x.L, x.A, aw);
                if (check && !h[j].failed) {
                    if (std::isfinite(reference_key) != std::isfinite(c.key)) max_key = INFINITY;
                    else if (std::isfinite(reference_key)) {
                        max_L = std::max(max_L, fabs(c.L - x.L));
                        max_A = std::max(max_A, fabs(c.A - x.A));
                        max_key = std::max(max_key, fabs(c.key - reference_key));
                    }
                    different_t += reference_t != h[j].t;
                }
                if (!canonical_ready) {
                    canonical_t = reference_t;
                    canonical_L = x.L;
                    canonical_ready = true;
                }
                // Failed arithmetic also needs CADO alpha. Healthy final
                // refinement retains device alpha but uses canonical size/t.
                if (h[j].failed || !final_refine) {
                    c.L = x.L; c.A = x.A; c.key = reference_key;
                    c.t = reference_t;
                    c.screen_host = true;
                }
                nf += h[j].failed != 0;
            }
            if (final_refine && !h[j].failed) {
                if (!canonical_ready) {
                    Int<RS_LP> f[6];
                    double content = 0;
                    const bool ok = rs_rotate(f, r, c.h.u, c.h.v) && rs_screen_primitive(f, content);
                    canonical_L = ok ? rs_size(r, c.h.u, c.h.v, canonical_t, 1 << 10) - content : INFINITY;
                }
                c.t = canonical_t;
                c.L = canonical_L;
                c.key = rs_screen_key(c.L, c.A, aw);
                c.screen_host = true;
            }
            // Screening retains the original start for its later final pass.
        }
        fallbacks += nf;
    }
    if (check) {
        printf("# %s GPU check: max |dL| %.12g, |dA| %.12g, |dkey| %.12g; "
               "%zu translations differ%s\n", final_refine ? "full-refine" : "alpha-screen",
               max_L, max_A, max_key, different_t, final_refine ? "" : " before host canonicalization");
        if (max_key > 1e-8) {
            fprintf(stderr, "s23_ropt: GPU %s disagrees with CPU reference; no polynomials written\n",
                    final_refine ? "final refinement" : "alpha screen");
            exit(2);
        }
    }
    cells.swap(screened);
    return true;
}
