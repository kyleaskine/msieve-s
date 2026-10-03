/* The same sopt_best_norm on the GPU: one thread per case, results compared with the
 * host build of the identical source (which test_lll checks against CADO). Run this
 * after test_lll passes; it only needs a free GPU.
 *
 *   test_device CASES [max_cases]
 */
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <cuda_runtime.h>
#include "gmp_bridge.h"
#include "sopt_lll.h"
#include "sopt_skew.h"

using namespace s23;

static const int LIMBS = 128;
static const int DEG = 5;
typedef Int<LIMBS> Z;
typedef LLLState<LIMBS, DEG, DEG + 1> State;
typedef SoptLLLResult<LIMBS, DEG> Result;

struct Case {
    Z f[DEG + 1], g[2], k, skew;
};

typedef BestNormWork<LIMBS, DEG> Work;

__global__ void kernel(const Case *cases, State *scratch, Work *work, Result *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    sopt_best_norm<LIMBS, DEG>(out[i], scratch[i], work[i], cases[i].f, cases[i].g, cases[i].k, cases[i].skew);
}

#define CUDA_CHECK(x)                                                                     \
    do {                                                                                  \
        cudaError_t e = (x);                                                              \
        if (e != cudaSuccess) {                                                           \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
            exit(2);                                                                      \
        }                                                                                 \
    } while (0)

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: test_device CASES [max_cases]\n");
        return 2;
    }
    long max_cases = argc > 2 ? atol(argv[2]) : -1; /* -1: all */
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("no CUDA device available; nothing to do\n");
        return 0;
    }
    FILE *fh = fopen(argv[1], "r");
    if (!fh) {
        perror(argv[1]);
        return 2;
    }
    std::vector<Case> cases;
    mpz_t z, f0, fd;
    mpz_inits(z, f0, fd, NULL);
    char buf[8192], tok[2048];
    while (fgets(buf, sizeof buf, fh) && (max_cases < 0 || (long)cases.size() < max_cases)) {
        char *p = buf;
        int off, d = (int)strtol(p, &p, 10);
        if (d != DEG)
            continue;
        Case c;
        for (int i = 0; i <= DEG; i++) {
            sscanf(p, "%2047s%n", tok, &off); p += off; mpz_set_str(z, tok, 10);
            from_mpz(c.f[i], z);
            if (i == DEG) mpz_set(fd, z);
        }
        sscanf(p, "%2047s%n", tok, &off); p += off; mpz_set_str(z, tok, 10); from_mpz(c.g[0], z); mpz_set(f0, z);
        sscanf(p, "%2047s%n", tok, &off); p += off; mpz_set_str(z, tok, 10); from_mpz(c.g[1], z);
        sscanf(p, "%2047s%n", tok, &off); p += off; mpz_set_str(z, tok, 10); from_mpz(c.k, z);
        /* skew on the host, exactly as CADO's sopt_get_skewness */
        sopt_skew_mpz(z, f0, fd, DEG);
        from_mpz(c.skew, z);
        cases.push_back(c);
    }
    fclose(fh);
    int n = (int)cases.size();

    /* the multi-limb routines are real calls with large frames: raise the per-thread
       stack limit (ptxas reports ~18 KB cumulative for L = 128) */
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 24 * 1024));

    Case *d_cases;
    State *d_scratch;
    Work *d_work;
    Result *d_out;
    CUDA_CHECK(cudaMalloc(&d_cases, n * sizeof(Case)));
    CUDA_CHECK(cudaMalloc(&d_scratch, n * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&d_work, n * sizeof(Work)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(Result)));
    CUDA_CHECK(cudaMemcpy(d_cases, cases.data(), n * sizeof(Case), cudaMemcpyHostToDevice));
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    kernel<<<(n + 63) / 64, 64>>>(d_cases, d_scratch, d_work, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    std::vector<Result> gpu(n);
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_out, n * sizeof(Result), cudaMemcpyDeviceToHost));

    static State S;
    static Work W;
    static Result R;
    long same = 0, diff = 0, fail = 0;
    for (int i = 0; i < n; i++) {
        sopt_best_norm<LIMBS, DEG>(R, S, W, cases[i].f, cases[i].g, cases[i].k, cases[i].skew);
        if (gpu[i].status != LLL_OK || R.status != LLL_OK) {
            fail++;
            continue;
        }
        bool eq = gpu[i].row == R.row && gpu[i].norm == R.norm;
        for (int j = 0; j <= DEG && eq; j++)
            eq = cmp(gpu[i].f[j], R.f[j]) == 0;
        eq ? same++ : diff++;
    }
    printf("%d cases on the GPU in %.1f ms: %ld identical to the host, %ld different, %ld failed\n",
           n, ms, same, diff, fail);
    cudaFree(d_cases);
    cudaFree(d_scratch);
    cudaFree(d_work);
    cudaFree(d_out);
    mpz_clears(z, f0, fd, NULL);
    return (diff || fail) ? 1 : 0;
}
