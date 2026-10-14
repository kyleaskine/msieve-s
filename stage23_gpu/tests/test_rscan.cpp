/* Basin discovery against known landscapes, independent of polynomial descent. */
#include <cassert>
#include <cstdio>
#include "rscan.h"

using namespace s23;

static RsLine line(int64_t u, double value, int64_t v = 0, int64_t t = 0)
{
    RsLine out;
    out.u = u;
    out.L_min = value;
    out.v_min = v;
    out.t_min = t;
    return out;
}

int main()
{
    RsScanConfig c{10, 4, 2, 1};
    assert((rs_scan_grid(c) == std::vector<int64_t>{-10, -8, -4, 0, 4, 8, 10}));
    c.umax = 8;
    assert((rs_scan_grid(c) == std::vector<int64_t>{-8, -4, 0, 4, 8}));
    c.umax = 0;
    assert((rs_scan_grid(c) == std::vector<int64_t>{0}));
    c = {3, 8, 1, 0};
    assert((rs_scan_grid(c) == std::vector<int64_t>{-3, 0, 3}));
    for (auto bad : {RsScanConfig{1000000001, 1000000, 1, 0}, {4096, 1, 1, 0},
                     {10, -1, 1, 0}, {10, 1, 0, 0}, {10, 1, 17, 0},
                     {10, 1, 1, -1}, {10, 1, 1, 4097}, {10, 1, 16, 1024}})
        assert(!rs_scan_valid(bad));

    std::map<int64_t, RsLine> lines;
    auto landscape = [](int64_t u, int64_t, int64_t) {
        return line(u, std::min(10. + u * u / 100., 1. + (u + 300) * (u + 300) / 100.));
    };
    lines[0] = landscape(0, 0, 0);
    c = {512, 64, 2, 4};
    auto stats = rs_scan_basins(lines, c, landscape);
    assert((stats.centers == std::vector<int64_t>{-300, 0}));
    assert(lines.at(0).L_min == 10. && lines.at(-300).L_min == 1.);
    for (int64_t u = -304; u <= -296; u++)
        assert(lines.count(u));
    assert(stats.added + 1 == lines.size());
    assert(stats.evaluations == stats.added);
    assert(stats.samples == 17);
    auto repeat = rs_scan_basins(lines, c, landscape);
    assert(repeat.centers == stats.centers && repeat.added == 0 && repeat.evaluations == 0);
    c.step = 0;
    auto disabled = rs_scan_basins(lines, c, landscape);
    assert(disabled.samples == 0 && disabled.evaluations == 0);

    // Overlapping expansions and clipped endpoints are cached; ties are stable.
    auto flat = [](int64_t u, int64_t, int64_t) { return line(u, 1.); };
    c = {10, 4, 4, 3};
    lines.clear();
    stats = rs_scan_basins(lines, c, flat);
    assert((stats.centers == std::vector<int64_t>{-10, 0, 8}));
    assert(lines.begin()->first == -10 && lines.rbegin()->first == 10);
    assert(stats.evaluations == lines.size());

    // Independent start rescues failed or inferior warm starts.
    for (double warm : {100., double(NAN)}) {
        lines.clear();
        lines[0] = line(0, 10., 7, 9);
        c = {2, 2, 1, 0};
        stats = rs_scan_basins(lines, c, [=](int64_t u, int64_t v, int64_t t) {
            return line(u, v || t ? warm : 1., 7, 9);
        });
        assert(lines.at(-2).L_min == 1. && lines.at(2).L_min == 1.);
        assert(stats.evaluations == 2 * stats.added);
    }
    // Failed probes cannot become winning basins, including cached NaN lines.
    lines.clear();
    lines[0] = line(0, NAN);
    c = {10, 4, 2, 1};
    stats = rs_scan_basins(lines, c, [](int64_t u, int64_t, int64_t) { return line(u, NAN); });
    assert(stats.centers.empty() && stats.nonfinite == 6 && stats.added == 6);
    lines.clear();
    lines[0] = line(0, NAN);
    c = {1, 1, 1, 0};
    stats = rs_scan_basins(lines, c, flat);
    assert((stats.centers == std::vector<int64_t>{-1}));
    puts("root basin scan: grids, bounds, distant basins, caching, starts and failures passed");
}
