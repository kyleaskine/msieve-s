/* Optional GPU stage-1 audit controls. No CUDA dependency, for CPU tests. */
#ifndef STAGE1_AUDIT_H
#define STAGE1_AUDIT_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <time.h>
#if defined(WIN32) || defined(_WIN64)
#include <windows.h>
#endif

typedef struct {
    char prefix[1024];
    uint32_t qmin, qmax, aprog;
} stage1_audit_options;

/* Values are whitespace-delimited, as for the other nfs_args options.
   Reject malformed or duplicate audit controls instead of silently widening work. */
static inline int stage1_audit_parse(const char *args, stage1_audit_options *o) {
    const char *s = args ? args : "";
    unsigned seen = 0;
    memset(o, 0, sizeof(*o)); o->qmin = 1; o->qmax = UINT32_MAX;
    while (*s) {
        const char *end, *v = NULL; unsigned flag = 0;
        while (isspace((unsigned char)*s)) s++;
        if (!*s) break;
        end = s; while (*end && !isspace((unsigned char)*end)) end++;
        if (!strncmp(s,"stage1_provenance=",18)) { v=s+18; flag=1; }
        else if (!strncmp(s,"stage1_qmin=",12)) { v=s+12; flag=2; }
        else if (!strncmp(s,"stage1_qmax=",12)) { v=s+12; flag=4; }
        else if (!strncmp(s,"stage1_aprog=",13)) { v=s+13; flag=8; }
        else if (!strncmp(s,"stage1_",7) && strncmp(s,"stage1_norm=",12)) return 0;
        if (flag) {
            if ((seen & flag) || v == end) return 0;
            seen |= flag;
            if (flag == 1) {
                size_t n = end-v;
                /* nfs_args still has legacy comma and substring parsers. */
                if (n >= sizeof(o->prefix) || memchr(v, ',', n) ||
                    memchr(v, '=', n)) return 0;
                memcpy(o->prefix,v,n); o->prefix[n]=0;
            } else {
                const char *p; char *last; unsigned long long n;
                for(p=v;p<end;p++) if(!isdigit((unsigned char)*p)) return 0;
                errno=0; n=strtoull(v,&last,10);
                if(errno || last!=end || !n || n>UINT32_MAX) return 0;
                if(flag==2) o->qmin=(uint32_t)n;
                if(flag==4) o->qmax=(uint32_t)n;
                if(flag==8) { if(n>5) return 0; o->aprog=(uint32_t)n; }
            }
        }
        s=end;
    }
    return o->qmin <= o->qmax;
}

static inline double stage1_audit_clock(void) {
#if defined(WIN32) || defined(_WIN64)
    LARGE_INTEGER t, f; QueryPerformanceCounter(&t); QueryPerformanceFrequency(&f);
    return (double)t.QuadPart/(double)f.QuadPart;
#else
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(1); }
    return (double)t.tv_sec + 1e-9*t.tv_nsec;
#endif
}
static inline double stage1_audit_wall_clock(void) {
#if defined(WIN32) || defined(_WIN64)
    FILETIME ft; ULARGE_INTEGER t;
    GetSystemTimeAsFileTime(&ft);
    t.LowPart=ft.dwLowDateTime; t.HighPart=ft.dwHighDateTime;
    return (double)t.QuadPart * 1e-7 - 11644473600.0;
#else
    struct timespec t;
    if (clock_gettime(CLOCK_REALTIME, &t)) { perror("clock_gettime"); exit(1); }
    return (double)t.tv_sec + 1e-9*t.tv_nsec;
#endif
}
/* Fixed progressions may require 64-bit keys even with 32-bit p^2. */
static inline uint32_t stage1_audit_batch_limit(uint32_t pmax, uint32_t aprog,
        uint32_t limit32, uint32_t limit64) {
    uint64_t maxroot;
    uint32_t capacity;
    if (pmax>=65536) return limit64/aprog;
    maxroot=(uint64_t)pmax*pmax*((aprog+1)/2)*(aprog>1 ? 2 : 1);
    capacity=maxroot>(UINT64_C(1)<<32) ? limit64 : limit32;
    return capacity/aprog;
}
static inline void stage1_audit_string(FILE *f, const char *s) {
    const unsigned char *p=(const unsigned char *)s;
    fputc('"',f);
    for(;*p;p++) {
        if(*p=='"' || *p=='\\') { fputc('\\',f); fputc(*p,f); }
        else if(*p<32) fprintf(f,"\\u%04x",*p);
        else fputc(*p,f);
    }
    fputc('"',f);
}
static inline int stage1_audit_flush(FILE *f) {
    return !f || (fflush(f)==0 && !ferror(f));
}
#endif
