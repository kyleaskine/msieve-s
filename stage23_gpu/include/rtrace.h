/* Read-only diagnostics for one original-coordinate rotation. Never add the
 * watched cell to a proposal pool, a size-model start, or a survivor list. */
#pragma once
#include <cstdio>
#include <cstdint>

namespace s23 {
struct RsCellTrace {
    bool enabled = false, block_hit = false;
    int64_t u = 0, v = 0;
    const char *first_loss = nullptr;

    bool matches(int64_t a, int64_t b) const { return enabled && u == a && v == b; }
    bool active() const { return enabled && !first_loss; }

    void stage(const char *name, bool kept)
    {
        if (!active()) return;
        printf("# trace-cell u %lld v %lld: stage %s %s\n",
               (long long)u, (long long)v, name, kept ? "pass" : "lost");
        if (!kept) first_loss = name;
    }

    // rank is one-based, or zero when absent from the available sorted pool.
    void rank(const char *name, size_t rank, size_t keep, size_t available)
    {
        if (!active()) return;
        printf("# trace-cell u %lld v %lld: %s rank %zu keep %zu available %zu\n",
               (long long)u, (long long)v, name, rank, keep, available);
        stage(name, rank && rank <= keep);
    }

    void finish(const char *success) const
    {
        if (enabled)
            printf("# trace-cell u %lld v %lld: result %s\n",
                   (long long)u, (long long)v, first_loss ? first_loss : success);
    }
};
} // namespace s23
