#ifndef _COMMOM_H_
#define _COMMOM_H_
//#include <slave.h>

//enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
//enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
//enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
//enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
//enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};


#define GET_RTC(rpcc) \
  asm volatile ("rcsr %0, 4":"=&r"(rpcc)::"memory");

#define get_slv_id(tid) asm volatile ("rcsr %0, 0" : "=r"(tid))
#define get_row_id(rid) asm volatile ("rcsr %0, 1" : "=r"(rid))
#define get_col_id(cid) asm volatile ("rcsr %0, 2" : "=r"(cid))
#define REG_PUTR(var, dst) asm volatile ("putr %0,%1"::"r"(var),"r"(dst))
#define REG_PUTC(var, dst) asm volatile ("putc %0,%1"::"r"(var),"r"(dst))
#define REG_GETR(var) asm volatile ("getr %0":"=r"(var))
#define REG_GETC(var) asm volatile ("getc %0":"=r"(var))

extern void* ldm_malloc(size_t size);
extern void ldm_free(void* ptr, size_t size);
#define ROWSYN  athread_syn(ROW_SCOPE,0xff)
#define COLSYN  athread_syn(COL_SCOPE,0xff)
#define ALLSYN  athread_syn(ARRAY_SCOPE,0xffff)

#define THREADS 64

typedef struct ConvData_st{
  void* input; //0
  void* weight; //8
  void* output; //16
  void* inputp;
  void* weightp;
  void* outputp;
  int N, blkN;
  int K, blkK;
  int M, blkM;
  //old var
  int Ni, No, B, T;
  // add leading dimension
  int ldi, ldw, ldo;
} ConvData;

typedef struct CopyData{
    void* src;
    void* dst;
    int M, N, Ms, Ns, Me, Ne;
    int trans;
    // add leading dimension
    int ldx;
}CopyData;

typedef struct ZeropadStruct_st {
  float* A;
  float* A_zeropad;
  int ld;
  int ld_pad;
  int hd;
  int hd_pad;
} ZeropadStruct;

#define MIN(x,y) (x>y?y:x)
#define MAX(x,y) (x>y?x:y)
#define TIME(a,b) (1.0*((b).tv_sec-(a).tv_sec)+0.000001*((b).tv_usec-(a).tv_usec))
#define ALIGNED(addr) ((((unsigned long)(addr)>>5)+1)<<5)
#define TEST_STEPS 1

#endif
