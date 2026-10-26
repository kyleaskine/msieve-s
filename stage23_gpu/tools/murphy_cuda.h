/* Accurate MurphyE quadrature on CUDA. CPU code retains bounded search control
 * and GMP translations; every angle's norm/rho evaluation runs on the device.
 * Workspaces are private to an OpenMP worker and survive seeds in a batch.
 * Explicit FMA matches CADO's homogeneous evaluator; implicit fusion is off.
 */
#pragma once
#include <atomic>
#include "murphy_rank.h"
#include "murphy_rho.h"
#include "../../stage23_bench/tools/murphy_quadrature.h"

__global__ void murphy_integrals(const MurphyIntegralInput *inputs, const double2 *angles,
                                double *partial)
{
    const auto &in = inputs[blockIdx.y];
    const double x = sqrt(in.area * in.skew), y = sqrt(in.area / in.skew);
    const double bf = 1.0 / log(in.Bf), bg = 1.0 / log(in.Bg);
    double sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < in.K;
         i += gridDim.x * blockDim.x) {
        const double xi = x * angles[i].x, yi = y * angles[i].y;
        const double f = (log(fabs(s23_murphy::homogeneous(in.f, in.degree, xi, yi))) + in.af) * bf;
        const double g = (log(fabs(s23_murphy::homogeneous(in.g, 1, xi, yi))) + in.ag) * bg;
        sum += s23::murphy_rho(f) * s23::murphy_rho(g);
    }
    __shared__ double sums[256];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned step = 128; step; step /= 2) {
        if (threadIdx.x < step) sums[threadIdx.x] += sums[threadIdx.x + step];
        __syncthreads();
    }
    if (!threadIdx.x) partial[blockIdx.y * gridDim.x + blockIdx.x] = sums[0];
}

struct MurphyCudaEvaluator : MurphyEvaluator {
    int device;
    bool check, unavailable = false, audit_failed = false;
    size_t calls = 0, integrals = 0, uploads = 0;
    double max_relative_error = 0;
    double max_alpha_error = 0;
    std::atomic<bool> *abort = nullptr;
    cudaStream_t stream = nullptr;
    MurphyIntegralInput *inputs = nullptr;
    double *partial = nullptr;
    double2 *angles[2]{};
    int orders[2]{};
    unsigned recent = 0;
    size_t capacity = 0;
    std::vector<double> host_partial;

    explicit MurphyCudaEvaluator(int dev, bool audit) : device(dev), check(audit) {}
    MurphyCudaEvaluator(const MurphyCudaEvaluator &) = delete;
    MurphyCudaEvaluator &operator=(const MurphyCudaEvaluator &) = delete;
    void clear_buffers() {
        if (stream || inputs || partial || angles[0] || angles[1]) {
            cudaSetDevice(device);
            if (stream) cudaStreamSynchronize(stream);
            if (inputs) cudaFree(inputs);
            if (partial) cudaFree(partial);
            for (auto p : angles) if (p) cudaFree(p);
            if (stream) cudaStreamDestroy(stream);
        }
        stream = nullptr; inputs = nullptr; partial = nullptr;
        angles[0] = angles[1] = nullptr;
        orders[0] = orders[1] = 0;
        recent = 0; capacity = 0;
    }
    ~MurphyCudaEvaluator() override { clear_buffers(); }

    void begin_job(std::atomic<bool> &stop) {
        // Retry transient setup/allocation failures next seed, retaining healthy
        // allocations and angle grids. Statistics and audit state are per job.
        if (unavailable) clear_buffers();
        unavailable = audit_failed = false;
        calls = integrals = uploads = 0;
        max_relative_error = max_alpha_error = 0;
        abort = &stop;
    }
    bool cancelled() const override {
        return audit_failed || (abort && abort->load(std::memory_order_relaxed));
    }
    void fail_audit() {
        audit_failed = true;
        if (abort) abort->store(true, std::memory_order_relaxed);
    }
    bool check_alpha(double supplied, double reference) override {
        if (!check) return true;
        const double error = fabs(supplied - reference);
        max_alpha_error = std::max(max_alpha_error, std::isfinite(error) ? error : INFINITY);
        if (!std::isfinite(supplied) || !std::isfinite(reference) || error > 1e-10) {
            fail_audit();
            fprintf(stderr, "s23_ropt: GPU MurphyE alpha audit failed: CADO %.17g supplied %.17g\n",
                    reference, supplied);
            return false;
        }
        return true;
    }
    bool accept_integral(const MurphyIntegralInput &in, double value) {
        const bool valid = std::isfinite(value) && value >= 0;
        if (!check) return valid;
        const double reference = valid ? murphy_integral_reference(in) : NAN;
        const double relative = reference == value ? 0 :
            fabs(reference - value) / std::max(fabs(reference), 1e-300);
        max_relative_error = std::max(max_relative_error, std::isfinite(relative) ? relative : INFINITY);
        if (!valid || !std::isfinite(reference) || !std::isfinite(relative) || relative > 2e-10) {
            fail_audit();
            fprintf(stderr, "s23_ropt: GPU MurphyE audit failed: CPU %.17g GPU %.17g K %d\n",
                    reference, value, in.K);
            return false;
        }
        return true;
    }

    bool good(cudaError_t error) {
        if (error == cudaSuccess) return true;
        if (!unavailable)
            fprintf(stderr, "s23_ropt: GPU MurphyE unavailable (%s); retrying refinement on CPU\n",
                    cudaGetErrorString(error));
        unavailable = true;
        cudaGetLastError();
        return false;
    }

    bool evaluate(const MurphyIntegralInput *in, size_t n, double *out) override {
        if (unavailable || cancelled() || !n || n > 256) return false;
        const int K = in[0].K;
        // Bound cache allocation even for library callers. Larger orders retain
        // the streaming CPU implementation, without changing the requested K.
        if (K < 1 || K > 1048576) return false;
        for (size_t i = 0; i < n; ++i)
            if (in[i].K != K || in[i].degree < 1 || in[i].degree > CIO_MAXDEG ||
                !(in[i].skew > 0) || !std::isfinite(in[i].skew)) return false;
        if (!good(cudaSetDevice(device))) return false;
        if (!stream && !good(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) return false;
        if (capacity < n) {
            if (!good(cudaStreamSynchronize(stream))) return false;
            if (inputs) cudaFree(inputs);
            if (partial) cudaFree(partial);
            inputs = nullptr; partial = nullptr; capacity = 0;
            if (!good(cudaMalloc(&inputs, 256 * sizeof(*inputs))) ||
                !good(cudaMalloc(&partial, 256 * 256 * sizeof(*partial)))) return false;
            capacity = 256;
            host_partial.resize(256 * 256);
        }
        int slot = -1;
        for (int j = 0; j < 2; ++j) if (orders[j] == K) slot = j;
        if (slot < 0) {
            slot = recent ^ 1;
            if (angles[slot]) cudaFree(angles[slot]);
            angles[slot] = nullptr; orders[slot] = 0;
            std::vector<double2> xy(K);
            for (int i = 0; i < K; ++i) {
                const double t = s23_murphy::angle(i, K);
                xy[i] = make_double2(cos(t), sin(t));
            }
            if (!good(cudaMalloc(&angles[slot], K * sizeof(double2))) ||
                !good(cudaMemcpyAsync(angles[slot], xy.data(), K * sizeof(double2),
                                      cudaMemcpyHostToDevice, stream)) ||
                !good(cudaStreamSynchronize(stream))) return false;
            orders[slot] = K; ++uploads;
        }
        recent = slot;
        const int blocks = std::min(256, (K + 1023) / 1024);
        if (!good(cudaMemcpyAsync(inputs, in, n * sizeof(*in), cudaMemcpyHostToDevice, stream))) return false;
        murphy_integrals<<<dim3(blocks, n), 256, 0, stream>>>(inputs, angles[slot], partial);
        if (!good(cudaGetLastError()) ||
            !good(cudaMemcpyAsync(host_partial.data(), partial, n * blocks * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream)) ||
            !good(cudaStreamSynchronize(stream))) return false;
        ++calls; integrals += n;
        for (size_t i = 0; i < n; ++i) {
            double sum = 0;
            for (int b = 0; b < blocks; ++b) sum += host_partial[i * blocks + b];
            out[i] = sum / K;
            if (cancelled() || !accept_integral(in[i], out[i])) return false;
        }
        return true;
    }
};
