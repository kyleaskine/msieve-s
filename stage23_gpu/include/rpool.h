/* Versioned pre-refinement pool: replay the same cells without modeling/sieving.
 * Bind it to the exact seed, alpha weight and projective setting. */
#pragma once
#include <fstream>
#include <iomanip>
#include <set>
#include <string>
#include "rblock.h"
#include "cado_io.h"

namespace s23 {
struct RsCandidate {
    double key;
    Hit h;
    double L;
    int64_t t;
    double A;
};

inline bool rs_pool_write(const char *path, const cio_poly &p, double aw, bool proj,
                          const std::vector<RsCandidate> &cells, size_t count)
{
    std::ofstream out(path);
    out << "S23_ROPT_POOL 1\n" << std::setprecision(17) << aw << ' ' << proj << '\n';
    auto integer = [&](mpz_srcptr z) {
        char *s = mpz_get_str(nullptr, 10, z);
        out << s << '\n';
        free(s);
    };
    integer(p.n);
    integer(p.g[0]);
    integer(p.g[1]);
    for (int i = 0; i < 6; i++) integer(p.f[i]);
    out << count << '\n';
    for (size_t i = 0; i < count; i++) {
        const auto &c = cells[i];
        out << c.h.u << ' ' << c.h.v << ' ' << c.h.score << ' ' << c.key << ' ' << c.L << ' ' << c.t << '\n';
    }
    out << "END\n";
    out.close();
    return !out.fail();
}

inline bool rs_pool_read(const char *path, const cio_poly &p, double aw, bool proj,
                         std::vector<RsCandidate> &cells)
{
    std::ifstream in(path);
    std::string word;
    int version, projective;
    double weight;
    if (!(in >> word >> version >> weight >> projective) || word != "S23_ROPT_POOL" ||
        version != 1 || weight != aw || projective != (int)proj)
        return false;
    auto integer = [&](mpz_srcptr z) {
        if (!(in >> word)) return false;
        mpz_t value;
        mpz_init(value);
        const bool ok = mpz_set_str(value, word.c_str(), 10) == 0 && mpz_cmp(value, z) == 0;
        mpz_clear(value);
        return ok;
    };
    if (!integer(p.n) || !integer(p.g[0]) || !integer(p.g[1])) return false;
    for (int i = 0; i < 6; i++) if (!integer(p.f[i])) return false;
    size_t n;
    if (!(in >> n) || !n || n > (1u << 24)) return false;
    std::vector<RsCandidate> loaded;
    std::set<std::pair<int64_t, int64_t>> seen;
    for (size_t i = 0; i < n; i++) {
        RsCandidate c{};
        if (!(in >> c.h.u >> c.h.v >> c.h.score >> c.key >> c.L >> c.t) ||
            !std::isfinite(c.h.score) || !std::isfinite(c.key) || !std::isfinite(c.L) ||
            !seen.emplace(c.h.u, c.h.v).second)
            return false;
        c.A = NAN;
        loaded.push_back(c);
    }
    if (!(in >> word) || word != "END" || (in >> word)) return false;
    cells = std::move(loaded);
    return true;
}
} // namespace s23
