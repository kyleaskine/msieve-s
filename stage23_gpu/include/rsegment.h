/* Streaming work geometry shared by the dense and residue-class root sieves. */
#pragma once
#include <algorithm>
#include <vector>
#include "rblock.h"
#include "rclass.h"

namespace s23 {

/* Original-coordinate endpoints; step counts the spacing of selected cells.
 * L0 + dL*(v-a) is always evaluated in original v coordinates. */
struct Seg {
    int64_t u, a, b;
    double L0, dL;
    uint32_t step = 1;
};

inline uint64_t seg_cells(const Seg &s)
{
    return s.b < s.a ? 0 : ((uint64_t)s.b - (uint64_t)s.a) / s.step + 1;
}

inline size_t seg_blocks(const Seg &s, int seg_log)
{
    return s.b < s.a ? 0 : (size_t)((((uint64_t)s.b - (uint64_t)s.a) / s.step >> seg_log) + 1);
}

/* Clip a piece between size-model knots to one v class; empty pieces are omitted. */
inline void class_segment(std::vector<Seg> &segs, Seg s, const RsClass &c)
{
    if (c.modulus == 1) {
        if (s.a <= s.b)
            segs.push_back(s);
        return;
    }
    const auto p = rs_progression(s.a, s.b, c.modulus, c.v);
    if (!p.count)
        return;
    s.L0 += s.dL * (double)(p.first - s.a);
    s.a = p.first;
    s.b = p.last;
    s.step = c.modulus;
    segs.push_back(s);
}

/* At most 2^seg_log SELECTED cells per block. Original coordinates are restored
 * before lookup, interpolation or export. Memory is bounded by the launch size. */
struct BlockGen {
    const std::vector<Seg> &segs;
    const int seg_log;
    size_t si = 0, made = 0;
    int64_t v = 0;
    bool in_seg = false;
    BlockGen(const std::vector<Seg> &s, int sl) : segs(s), seg_log(sl) {}
    bool next(Blk &B)
    {
        for (; si < segs.size(); si++, in_seg = false) {
            const Seg &S = segs[si];
            if (!in_seg) {
                v = S.a;
                in_seg = true;
            }
            if (S.b < S.a || v > S.b)
                continue;
            const uint64_t rest = ((uint64_t)S.b - (uint64_t)v) / S.step + 1;
            const uint32_t len = (uint32_t)std::min<uint64_t>(rest, 1ull << seg_log);
            B = {S.u, v, len, (float)(S.L0 + S.dL * (double)((uint64_t)v - (uint64_t)S.a)),
                 (float)(S.dL * S.step), S.step};
            made++;
            if (rest <= len) {
                si++;
                in_seg = false;
            } else
                v += (int64_t)len * S.step;
            return true;
        }
        return false;
    }
};

} // namespace s23
