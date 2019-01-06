#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include "./include/common_slave.h"
#include <cblas.h>

extern SLAVE_FUN(FJR_blas_sgemm_trans_implicit)();


/******
 * N should not be blocked
**** */

void sw_sgemm_trans(float* input, float* weight, float* output, int M, int N, int K, int blkM, int blkN, int blkK) {
  ConvData* params = (ConvData*)malloc(sizeof(ConvData));
  params->input = input;
  params->weight = weight;
  params->output = output;
  params->K = K;
  params->blkK = blkK;
  params->N = N;
  params->blkN = blkN;
  params->M = M;
  params->blkM = blkM;
  //params->numK = (K + blkK - 1)/blkK;

  //int ldm_use = sizeof(double)*(blkK*blkN*2 + blkK*blkM*2 + blkN*blkM)/64;
  //printf("M %d K %d N %d, blkM %d blkK %d, blkN %d LDM %d KM\n", M, K, N, blkM, blkK, blkN, ldm_use/1024);
  //if(ldm_use > 56*1024)
  //  return;

  if(params->blkM%128 == 0 && params->blkN%32 == 0 && params->blkK%8 == 0){
    athread_spawn(FJR_blas_sgemm_trans_implicit, params);
    athread_join();
  } else {
    float alpha = 1.;
    float beta = 0.;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, M, K, alpha, weight, N, input, M, beta, output, M);
  }
  free(params);
  return;
}

