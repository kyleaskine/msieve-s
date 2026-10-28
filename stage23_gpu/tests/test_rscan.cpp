/* Basin discovery against known landscapes, independent of polynomial descent. */
#include <cassert>
#include <cstdio>
#include <omp.h>
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
    omp_set_num_threads(2);
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

    // The same guarded probe powers batch hints and basin scans. A NaN warm
    // result must not displace a finite cold result; INF cannot be a warm start.
    for (double bad : {double(NAN), double(INFINITY)}) {
        lines.clear();
        lines[0] = line(0, 10., 7, 9);
        lines[10] = line(10, INFINITY, 99, 99);
        RsProbeStats probes;
        auto evaluate = [=](int64_t u, int64_t v, int64_t t) {
            assert(v!=99 && t!=99);
            return line(u,v||t ? bad : 1.,7,9);
        };
        rs_probe_batch(lines, {0,1,1,11}, probes, evaluate);
        assert(lines.at(1).L_min==1 && lines.at(11).L_min==1);
        assert(probes.evaluations==3 && probes.cache_hits==1);
        rs_probe_batch(lines, {0,1,11}, probes, evaluate);
        assert(probes.evaluations==3 && probes.cache_hits==4);
    }
    // Far-away selected rows expand on both sides of their hinted basin.
    // Cached coarse/hinted points are reused even at a tight work quota.
    lines.clear();
    auto distant = [](int64_t u, int64_t, int64_t) {
        return line(u,1.+(u-400000)*(u-400000)/1e6);
    };
    RsProbeStats probes;
    rs_probe_batch(lines,{0,400000},probes,distant);
    auto walk = rs_scan_progression(lines,400000,30,1000000,3.,20,9,probes,distant);
    assert(walk.visited==9 && walk.limited);
    for (int i=-4;i<=4;++i) assert(lines.count(400000+30*i));
    assert(lines.count(30)==0 && lines.count(-30)==0);
    assert(probes.evaluations==lines.size() && probes.cache_hits==1);
    const auto evaluations=probes.evaluations;
    rs_scan_progression(lines,400000,30,1000000,3.,20,9,probes,distant);
    assert(probes.evaluations==evaluations);
    lines.clear(); probes={};
    walk=rs_scan_progression(lines,-100,30,100,3.,20,100,probes,flat);
    assert(walk.visited==7 && !walk.limited && lines.rbegin()->first==80);
    lines.clear(); probes={};
    walk=rs_scan_progression(lines,0,1,100,0.,2,100,probes,
        [](int64_t u,int64_t,int64_t){return line(u,u ? INFINITY : 1.);});
    assert(walk.visited==5 && !walk.limited && probes.nonfinite==4);
    puts("root basin scan: grids, bounds, distant basins, caching, starts and failures passed");
}
