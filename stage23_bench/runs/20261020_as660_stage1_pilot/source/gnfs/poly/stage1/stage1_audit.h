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
static int stage1_audit_parse(const char *args, stage1_audit_options *o) {
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
                if (n >= sizeof(o->prefix)) return 0;
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

static double stage1_audit_clock(void) {
#if defined(WIN32) || defined(_WIN64)
    LARGE_INTEGER t, f; QueryPerformanceCounter(&t); QueryPerformanceFrequency(&f);
    return (double)t.QuadPart/(double)f.QuadPart;
#else
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(1); }
    return (double)t.tv_sec + 1e-9*t.tv_nsec;
#endif
}
static void stage1_audit_string(FILE *f, const char *s) {
    const unsigned char *p=(const unsigned char *)s;
    fputc('"',f);
    for(;*p;p++) {
        if(*p=='"' || *p=='\\') { fputc('\\',f); fputc(*p,f); }
        else if(*p<32) fprintf(f,"\\u%04x",*p);
        else fputc(*p,f);
    }
    fputc('"',f);
}
static void stage1_audit_flush(FILE *f) {
    if(f && (fflush(f) || ferror(f))) { perror("stage1 provenance write"); exit(1); }
}
#endif
