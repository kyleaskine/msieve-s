/* Device audit failure semantics and same-device recovery across jobs. */
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <vector>
#include "murphy_cuda.h"

int main()
{
    MurphyIntegralInput in;
    in.degree = 5; in.f[0] = 1; in.f[1] = -3; in.f[5] = 1;
    in.g[0] = -1; in.g[1] = 1;
    in.skew = 3.3e6; in.area = 1e16; in.Bf = 1e7; in.Bg = 5e6;
    in.af = -5; in.ag = .5; in.K = 16000;
    std::atomic<bool> stop{false};
    MurphyCudaEvaluator worker(0, true), peer(0, true);
    worker.begin_job(stop); peer.begin_job(stop);
    double value;
    int bad = !worker.evaluate(&in, 1, &value);
    bad += worker.calls != 1 || worker.integrals != 1 || worker.uploads != 1;
    auto angles = worker.angles[worker.recent];
    worker.max_relative_error = 1e-12;
    worker.begin_job(stop);
    bad += worker.max_relative_error != 0 || worker.calls || worker.integrals || worker.uploads;
    bad += !worker.evaluate(&in, 1, &value) || worker.uploads != 0 || worker.angles[worker.recent] != angles;
    for (double invalid : {double(NAN), double(INFINITY), -1.0}) {
        stop.store(false); worker.begin_job(stop); peer.begin_job(stop);
        bad += worker.accept_integral(in, invalid) || !worker.audit_failed || !peer.cancelled();
        bad += !std::isinf(worker.max_relative_error);
    }
    stop.store(false); worker.begin_job(stop);
    bad += worker.check_alpha(-4, -5) || !worker.audit_failed || !stop.load();
    stop.store(false); worker.begin_job(stop);
    bad += worker.max_alpha_error != 0 || !worker.check_alpha(-5, -5);
    // Simulate one recoverable allocation failure without exhausting device RAM.
    bad += worker.good(cudaErrorMemoryAllocation) || !worker.unavailable;
    bad += worker.evaluate(&in, 1, &value) || stop.load();
    worker.begin_job(stop);
    bad += worker.unavailable || !worker.evaluate(&in, 1, &value) || worker.calls != 1;
    MurphyCudaEvaluator unchecked(0, false);
    unchecked.begin_job(stop);
    bad += unchecked.accept_integral(in, NAN) || unchecked.audit_failed || stop.load();
    printf("CUDA MurphyE audit, cancellation and per-job recovery: %d failures\n", bad);
    return bad ? 1 : 0;
}
