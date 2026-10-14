/* Geometry of the original-coordinate class (u,v) = (ru,rv) mod M.
 * Scores still use the ORIGINAL seed and prime-power tables: restricting the
 * indices keeps all constant valuations and conditional lifts, including p | M.
 * No modular inverse of M and no transformed polynomial are needed.
 */
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace s23 {

inline uint32_t rs_residue(int64_t x, uint32_t m)
{
    const int64_t r = x % (int64_t)m;
    return (uint32_t)(r < 0 ? r + m : r);
}

struct RsClass {
    uint32_t modulus = 1, u = 0, v = 0;
};

struct RsProgression {
    int64_t first = 1, last = 0;
    uint64_t count = 0;
};

/* Intersection with [lo,hi]. Endpoints and offsets may be negative. The caller
 * excludes a full 2^64-cell interval (not representable by count). */
inline RsProgression rs_progression(int64_t lo, int64_t hi, uint32_t m, uint32_t residue)
{
    if (hi < lo)
        return {};
    const uint64_t span = (uint64_t)hi - (uint64_t)lo;
    const uint32_t delta = (residue + m - rs_residue(lo, m)) % m;
    if (delta > span)
        return {};
    const int64_t first = lo + delta; // delta <= hi-lo, so addition cannot overflow
    const uint64_t n = (span - delta) / m + 1;
    const int64_t last = hi - (int64_t)((span - delta) % m);
    return {first, last, n};
}

/* Keep at most count consecutive members, centered as near minimum as possible.
 * Arithmetic before clipping is wider than the coordinates to handle endpoints
 * near INT64_MIN/MAX; only host-side work planning uses it. */
inline RsProgression rs_trim_progression(RsProgression p, uint32_t m, int64_t minimum, uint64_t count)
{
    if (!count || !p.count)
        return {};
    if (p.count <= count)
        return p;
    const __int128 delta = (__int128)minimum - p.first;
    const __int128 nearest = (delta + m / 2) / m;
    const __int128 start = std::max<__int128>(0, std::min<__int128>(p.count - count, nearest - (count - 1) / 2));
    const int64_t first = (int64_t)((__int128)p.first + start * m);
    const int64_t last = (int64_t)((__int128)first + (__int128)(count - 1) * m);
    return {first, last, count};
}

/* Disjoint v residues on a matching u row. Union counts and ordinal selection
 * let the final cap trim a mixed-class line without enumerating its cells. */
struct RsClassRows {
    uint32_t modulus = 1;
    std::map<uint32_t, std::vector<uint32_t>> rows;
    const std::vector<uint32_t> &at(int64_t u) const
    {
        static const std::vector<uint32_t> empty;
        auto it = rows.find(rs_residue(u, modulus));
        return it == rows.end() ? empty : it->second;
    }
    uint64_t count(int64_t u, int64_t lo, int64_t hi) const
    {
        uint64_t n = 0;
        for (uint32_t v : at(u))
            n += rs_progression(lo, hi, modulus, v).count;
        return n;
    }
    int64_t ordinal(int64_t u, int64_t lo, int64_t hi, uint64_t index) const
    {
        const int64_t origin = lo;
        while (lo < hi) {
            const int64_t mid = (int64_t)((__int128)lo + ((__int128)hi - lo) / 2);
            if (count(u, origin, mid) > index)
                hi = mid;
            else
                lo = mid + 1;
        }
        return lo;
    }
    RsProgression clip(int64_t u, int64_t lo, int64_t hi) const
    {
        RsProgression out;
        for (uint32_t v : at(u)) {
            const auto p = rs_progression(lo, hi, modulus, v);
            if (!p.count)
                continue;
            if (!out.count)
                out = p;
            else {
                out.first = std::min(out.first, p.first);
                out.last = std::max(out.last, p.last);
                out.count += p.count;
            }
        }
        return out;
    }
    RsProgression trim(int64_t u, int64_t lo, int64_t hi, int64_t minimum, uint64_t n) const
    {
        const auto p = clip(u, lo, hi);
        if (at(u).size() == 1)
            return rs_trim_progression(p, modulus, minimum, n);
        if (!n || !p.count)
            return {};
        if (n >= p.count)
            return p;
        const uint64_t before = minimum <= p.first ? 0 : count(u, p.first, minimum - 1);
        const uint64_t start = std::min(p.count - n, before > (n - 1) / 2 ? before - (n - 1) / 2 : 0);
        return {ordinal(u, p.first, p.last, start), ordinal(u, p.first, p.last, start + n - 1), n};
    }
};

} // namespace s23
