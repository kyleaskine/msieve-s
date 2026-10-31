/* Runtime compatibility checks for the sopt device log port.
 * A finite test corpus is a compatibility check, not a proof over all doubles.
 * Unknown host behavior uses CADO on the CPU; device disagreement is an error.
 */
#pragma once
#include "host_log.h"
#include <cfenv>
#include <cstdio>
#include <random>
#include <vector>

namespace s23 { namespace matched_log {

inline std::vector<double> probe_inputs(size_t random_pairs = 1000000)
{
    std::vector<double> inputs;
    inputs.reserve(2 * random_pairs + 7000);
    std::mt19937_64 rng(0x534f5054);
    for (size_t i = 0; i < random_pairs;) {
        uint64_t u = rng() & 0x7fffffffffffffffULL;
        if (!u || u >= 0x7ff0000000000000ULL) continue;
        inputs.push_back(number(u));
        inputs.push_back(1.0 + (double)(int64_t)rng() * 0x1p-66);
        ++i;
    }
    for (int k = -1074; k <= 1023; ++k) {
        double x = std::ldexp(1., k);
        inputs.push_back(x);
        if (k > -1074) inputs.push_back(std::nextafter(x, 0.));
        inputs.push_back(std::nextafter(x, INFINITY));
    }
    for (int i = 0; i <= 128; ++i) {
        double x = number(0x3fe6000000000000ULL + ((uint64_t)i << 45));
        inputs.push_back(x);
        inputs.push_back(std::nextafter(x, 0.));
        inputs.push_back(std::nextafter(x, INFINITY));
    }
    return inputs;
}

struct Check {
    Mode mode = unsupported;
    size_t samples = 0, fused_mismatches = 0, unfused_mismatches = 0;
    std::vector<double> reference;
};

// Injectable reference keeps variant selection testable on either host variant.
template<class Reference> inline Check identify(const std::vector<double> &inputs, Reference reference)
{
    Check check;
    if (std::fegetround() != FE_TONEAREST) return check;
    check.reference.reserve(inputs.size());
    for (double x : inputs) {
        double ref = reference(x);
        check.reference.push_back(ref);
        check.fused_mismatches += bits(ref) != bits(eval<true>(x));
        check.unfused_mismatches += bits(ref) != bits(eval<false>(x));
    }
    check.samples = inputs.size();
    if (check.samples) {
        if (!check.fused_mismatches) check.mode = fused;
        else if (!check.unfused_mismatches) check.mode = unfused;
    }
    return check;
}

inline Check identify_host(const std::vector<double> &inputs)
{
#if defined(__GLIBC__) && defined(__x86_64__) && !defined(__FAST_MATH__)
    // Prevent a compiler builtin from replacing the library function under test.
    double (*volatile reference)(double) = ::log;
    return identify(inputs, [&](double x) { return reference(x); });
#else
    (void)inputs;
    return {};
#endif
}

#ifdef __CUDACC__
__global__ void probe_kernel(const double *inputs, double *outputs, size_t count)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) outputs[i] = matched_log::log(inputs[i]);
}

// CUDA failures and mismatches are explicit failures, never a cached success.
inline bool check_device(const std::vector<double> &inputs, const Check &host)
{
    if (host.mode == unsupported || host.reference.size() != inputs.size() || inputs.empty()) return false;
    double *in = nullptr, *out = nullptr;
    auto checked = [](cudaError_t error) {
        if (error == cudaSuccess) return true;
        fprintf(stderr, "s23_sopt: device log check: %s\n", cudaGetErrorString(error));
        return false;
    };
    size_t bytes = inputs.size() * sizeof(double);
    int mode = host.mode;
    bool ok = checked(cudaMemcpyToSymbol(device_mode, &mode, sizeof(mode))) &&
              checked(cudaMalloc(&in, bytes)) && checked(cudaMalloc(&out, bytes));
    std::vector<double> values(inputs.size());
    if (ok) ok = checked(cudaMemcpy(in, inputs.data(), bytes, cudaMemcpyHostToDevice));
    if (ok) {
        probe_kernel<<<(inputs.size() + 255) / 256, 256>>>(in, out, inputs.size());
        ok = checked(cudaGetLastError()) && checked(cudaMemcpy(values.data(), out, bytes, cudaMemcpyDeviceToHost));
    }
    if (in) ok = checked(cudaFree(in)) && ok;
    if (out) ok = checked(cudaFree(out)) && ok;
    if (!ok) return false;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (bits(values[i]) != bits(host.reference[i])) {
            fprintf(stderr, "s23_sopt: device log mismatch x=%a host=%a device=%a\n",
                    inputs[i], host.reference[i], values[i]);
            return false;
        }
    }
    return true;
}
#endif
} } // namespace s23::matched_log
