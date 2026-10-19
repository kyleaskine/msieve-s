/* A batch is a list of argv arrays, not shell commands. All options and path
 * collisions are checked before the first job writes anything. Jobs execute
 * serially with one shared CUDA session and independent numerical state. */
#pragma once
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>
#include <utility>
#include <sys/stat.h>

struct RoptJobPaths {
    std::vector<std::string> inputs, outputs;
};

template<class Validate, class Execute>
int ropt_batch(const char *manifest, const char *program, Validate validate, Execute execute)
{
    std::ifstream in(manifest);
    std::string line;
    auto read_line = [&]() {
        if (!std::getline(in, line)) return false;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
    };
    if (!read_line() || line != "S23_ROPT_BATCH 1") {
        fprintf(stderr, "s23_ropt: batch needs a readable manifest with header S23_ROPT_BATCH 1\n");
        return 2;
    }
    std::vector<std::vector<std::string>> jobs;
    std::vector<RoptJobPaths> paths;
    size_t lineno = 1;
    while (read_line()) {
        ++lineno;
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> args{program};
        size_t begin = 0;
        for (;;) {
            size_t end = line.find('\t', begin);
            std::string arg = line.substr(begin, end == std::string::npos ? end : end - begin);
            if (arg.empty() || arg.find('\0') != std::string::npos) {
                fprintf(stderr, "s23_ropt: batch line %zu: empty argument or NUL byte\n", lineno);
                return 2;
            }
            args.push_back(std::move(arg));
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        jobs.push_back(std::move(args));
    }
    if (in.bad() || jobs.empty()) {
        fprintf(stderr, "s23_ropt: batch is empty or unreadable\n");
        return 2;
    }
    for (auto &args : jobs) {
        std::vector<char *> argv;
        for (auto &s : args) argv.push_back(s.data());
        argv.push_back(nullptr);
        RoptJobPaths p;
        if (validate(int(args.size()), argv.data(), p)) {
            fprintf(stderr, "s23_ropt: invalid batch job %zu; no jobs started\n", paths.size() + 1);
            return 2;
        }
        paths.push_back(std::move(p));
    }
    // Resolve symlinks/relative paths and also protect existing hard-link aliases.
    using FileId = std::pair<dev_t, ino_t>;
    std::set<std::string> reads, writes;
    std::set<FileId> read_ids, write_ids;
    auto name = [](const std::string &p) { return std::filesystem::weakly_canonical(p).string(); };
    auto add_read = [&](const std::string &p) {
        reads.insert(name(p));
        struct stat s;
        if (!stat(p.c_str(), &s)) read_ids.emplace(s.st_dev, s.st_ino);
    };
    try {
        add_read(manifest);
        for (const auto &p : paths) for (const auto &f : p.inputs) add_read(f);
        for (const auto &p : paths) for (const auto &f : p.outputs) {
            const auto path = name(f);
            struct stat s;
            const bool exists = !stat(f.c_str(), &s);
            const FileId id = exists ? FileId{s.st_dev, s.st_ino} : FileId{};
            if (reads.count(path) || !writes.insert(path).second ||
                (exists && (S_ISDIR(s.st_mode) || read_ids.count(id) || !write_ids.insert(id).second))) {
                fprintf(stderr, "s23_ropt: batch output aliases an input or another output: %s; no jobs started\n", f.c_str());
                return 2;
            }
        }
    } catch (const std::filesystem::filesystem_error &e) {
        fprintf(stderr, "s23_ropt: invalid batch path: %s\n", e.what());
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    size_t partial = 0;
    for (size_t j = 0; j < jobs.size(); ++j) {
        auto &args = jobs[j];
        std::vector<char *> argv;
        for (auto &s : args) argv.push_back(s.data());
        argv.push_back(nullptr);
        printf("# batch job %zu/%zu begin\n", j + 1, jobs.size());
        fflush(stdout);
        const auto before = std::chrono::steady_clock::now();
        const int status = execute(int(args.size()), argv.data());
        printf("# batch job %zu status %d, %.6f s\n", j + 1, status,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - before).count());
        fflush(stdout);
        if (status && status != 3) {
            fprintf(stderr, "s23_ropt: batch stopped at job %zu; earlier completed exports retained\n", j + 1);
            return status;
        }
        partial += status == 3;
    }
    printf("# batch: %zu jobs complete, %zu partial, %.6f s\n", jobs.size(), partial,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return partial ? 3 : 0;
}
