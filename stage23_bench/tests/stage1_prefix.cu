// Tiny arithmetic regression: nvcc -std=c++17 -arch=sm_120 -I. this.cu -o test
// Uses the production kernel and an independent host modular inverse.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include "gnfs/poly/stage1/stage1_core_gpu/stage1_core.cu"
#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { \
 fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); exit(2); } } while(0)
static uint64 inverse(uint64 a, uint64 n) {
    __int128 t=0, u=1;
    uint64 r=n, s=a;
    while(s) { uint64 q=r/s, v=r%s; r=s; s=v;
        __int128 w=t-(__int128)q*u; t=u; u=w; }
    if(r!=1) abort();
    t%=n; if(t<0) t+=n; return (uint64)t;
}
int main() {
    uint32 *p, *meta; uint64 *start; int64 *out; specialq_t *qs;
    CUDA_OK(cudaMallocManaged(&p,2*sizeof(*p)));
    CUDA_OK(cudaMallocManaged(&start,4*sizeof(*start)));
    CUDA_OK(cudaMallocManaged(&qs,8*sizeof(*qs)));
    CUDA_OK(cudaMallocManaged(&meta,128*sizeof(*meta)));
    CUDA_OK(cudaMallocManaged(&out,128*sizeof(*out)));
    p[0]=65537; p[1]=65539;
    for(int i=0;i<4;i++) start[i]=1234+i*73;
    int failures=0, cases=0;
    // Repeated q=2^16 makes the initial prefix exactly 2^32. The old
    // narrowing test skips the last valid root, including before invalid q.
    for(int trailing=0;trailing<2;trailing++)
    for(int tile: {2, 3, 8})
    for(int aprog: {1, 3}) {
        int nq=2+trailing, entries=4;
        qs[0]={65536,0,1ULL<<32,17}; qs[1]={65536,0,1ULL<<32,89};
        qs[2]={65537,0,65537ULL*65537,31};
        for(int i=0;i<128;i++) {out[i]=0;meta[i]=0;}
        sieve_kernel_trans_pp64_r64<<<dim3(1,(nq+tile-1)/tile),32>>>(
            p,2,start,2,meta,out,qs,nq,tile,entries,17,aprog);
        CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
        for(int q=0;q<nq;q++) for(int r=0;r<2;r++) for(int pi=0;pi<2;pi++) {
            uint64 pp=(uint64)p[pi]*p[pi];
            bool valid=std::gcd(p[pi],qs[q].p)==1;
            uint64 root=valid ? (uint64)((unsigned __int128)
                ((start[r*2+pi]+pp-qs[q].root%pp)%pp)*inverse(qs[q].pp%pp,pp)%pp):0;
            for(int a=0;a<aprog;a++) {
                int idx=a*entries*nq+q*entries+r*2+pi;
                int64 expected=0;
                if(valid) expected=aprog==1 ? (root>pp/2 ? (int64)root-(int64)pp : (int64)root)
                    : (int64)root+(a-aprog/2)*(int64)pp;
                if(out[idx]!=expected || (valid && meta[idx]!=((q<<17)|p[pi]))) {
                    if(failures++<8) fprintf(stderr,"case %d slot %d got %lld expected %lld\n",cases,idx,out[idx],expected);
                }
            }
        }
        cases++;
    }
    cudaFree(out);cudaFree(meta);cudaFree(qs);cudaFree(start);cudaFree(p);
    printf("%d cases, %d mismatches\n",cases,failures);
    return failures?1:0;
}
