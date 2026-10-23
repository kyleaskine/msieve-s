#include "gnfs/poly/stage1/stage1_audit.h"
#include "gnfs/poly/stage1/stage1_audit_sha256.h"
#include <assert.h>
int main(void) {
    stage1_audit_options o;
    assert(stage1_audit_parse(NULL,&o) && o.qmin==1 && o.qmax==UINT32_MAX && !o.aprog);
    assert(stage1_audit_parse("stage1_norm=1e24 collengine=gerbicz stage1_qmin=1 stage1_qmax=4294967295 stage1_aprog=3 stage1_provenance=/tmp/a",&o));
    assert(o.qmin==1 && o.qmax==UINT32_MAX && o.aprog==3 && !strcmp(o.prefix,"/tmp/a"));
    const char *bad[]={"stage1_qmin=0","stage1_qmax=4294967296","stage1_qmin=-1",
      "stage1_qmin=1x","stage1_qmax=2 stage1_qmin=3","stage1_aprog=6","stage1_aprog=0",
      "stage1_qmin=1 stage1_qmin=2","stage1_provenance=","stage1_qmax=", "stage1_qmin=+1",
      "stage1_qmn=1", "stage1_qmin=184467440737095516160",
      "stage1_provenance=runs/c196,ad138600",
      "stage1_provenance=runs/min_coeff=99"};
    for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++) assert(!stage1_audit_parse(bad[i],&o));
    FILE *f=tmpfile(); assert(f); stage1_audit_string(f,"a\"b\\c\n\t"); assert(stage1_audit_flush(f));
    rewind(f); char text[100]={0}; fread(text,1,99,f);
    assert(!strcmp(text,"\"a\\\"b\\\\c\\u000a\\u0009\"")); fclose(f);
    assert(stage1_audit_clock()>0);
    assert(stage1_audit_wall_clock()>0);
    assert(stage1_audit_batch_limit(60000,1,300,200)==300);
    assert(stage1_audit_batch_limit(70000,1,300,200)==200);
    assert(stage1_audit_batch_limit(30000,3,300,200)==100);
    assert(stage1_audit_batch_limit(60000,3,300,200)==66);
    assert(stage1_audit_batch_limit(30000,5,300,200)==40);
    char hash[65];
    stage1_audit_sha256("",0,hash);
    assert(!strcmp(hash,"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    stage1_audit_sha256("abc",3,hash);
    assert(!strcmp(hash,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    const char *longtext="abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    stage1_audit_sha256(longtext,strlen(longtext),hash);
    assert(!strcmp(hash,"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
    char *million=malloc(1000000); assert(million); memset(million,'a',1000000);
    stage1_audit_sha256(million,1000000,hash); free(million);
    assert(!strcmp(hash,"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
#ifndef _WIN32
    f=fopen("/dev/full","w"); assert(f); fputs("test",f);
    assert(!stage1_audit_flush(f)); fclose(f);
#endif
    puts("audit option and JSON escaping tests passed");
}
