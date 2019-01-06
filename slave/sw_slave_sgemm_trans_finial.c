#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "swblas.h"
#include <dma.h>
#include "include/common_slave.h"

/***************
 * @deprecated!
 * slave code for A Trans B NoTrans version GEMM
 * input  is of dim(K, M)
 * weight is of dim(K, N)
 * ouput  is of dim(N, M)
 * blkM is 128x, blkK is 8x (>16 initial instruction pipeline), blkN is 32x
 * Attention please. Issues left over from history.
 * weight is A
 * input  is B
 * output is C
 * so, the block size of parameter M in cblas_sgemm, should be blkN here!!!
 * By Jiarui Fang 2019.1.4
 * ************/
//#undef USE_RTC
#define LDM_ALINGN_SIZE 256
void FJR_blas_sgemm_trans_dma_full_pipeline(ConvData* param)
{
  long long rtc_start, rtc_end;
  long long rtc_part_start, rtc_part_end;
  int i;
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int blkK = param->blkK;
  int N = param->N;
  int blkN = param->blkN;
  int numN = N/blkN;
  int M = param->M;
  int blkM = param->blkM;
  int T = M/blkM;
  int K = param->K;
  int numK = K/blkK;
//#define _LDM_ALIGN_

//M, K, Ci, Ri
#ifdef _LDM_ALIGN_
  char* local_input_start  = (char*)ldm_malloc(sizeof(double)*blkK*blkM/8/8*2 + LDM_ALINGN_SIZE);
  double* local_input = (double*)(local_input_start + LDM_ALINGN_SIZE - (long)local_input_start%LDM_ALINGN_SIZE);
#else
  double* local_input = (double*)((doublev4*)ldm_malloc(sizeof(double)*blkK*blkM/8/8*2));
#endif
  int local_input_size = blkK*blkM/8/8;
//N, K, K, K
#ifdef _LDM_ALIGN_
  char* local_weight_start = (char*)ldm_malloc(sizeof(double)*blkK*blkN/8/8*2 + LDM_ALINGN_SIZE);
  double* local_weight = (double*)(local_weight_start + LDM_ALINGN_SIZE - (long)local_weight_start%LDM_ALINGN_SIZE);
#else
  double* local_weight = (double*)((doublev4*)ldm_malloc(sizeof(double)*blkK*blkN/8/8*2));
#endif
  int local_weight_size = blkK*blkN/64;
//M, N, Co, Ro
#ifdef _LDM_ALIGN_
  char* local_output_start = (char*)ldm_malloc(sizeof(double)*blkN*blkM/8/8 + LDM_ALINGN_SIZE);
  double* local_output = (double*)(local_output_start + LDM_ALINGN_SIZE - (long)local_output_start%LDM_ALINGN_SIZE);
#else
  double* local_output = (double*)((doublev4*)ldm_malloc(sizeof(double)*blkN*blkM/8/8));
#endif
  int local_output_size = blkN*blkM/8/8;

  volatile int  input_replyget = 0, weight_replyget = 0,  replyput = 0;
  dma_desc dma_get_input, dma_get_weight, dma_get_output, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_weight, DMA_GET);
  dma_set_mode(&dma_get_weight, PE_MODE);
  dma_set_reply(&dma_get_weight, &weight_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_iutput(M/8, K/8)
  dma_set_size(&dma_get_input, blkM*blkK/8/8*sizeof(float));
  dma_set_bsize(&dma_get_input, blkM/8*sizeof(float));
  dma_set_stepsize(&dma_get_input, (M-blkM/8)*sizeof(float));

  //DMA for local_weight(K/8, N/8)
  dma_set_size(&dma_get_weight, blkN*blkK/8/8*sizeof(float));
  dma_set_bsize(&dma_get_weight, blkN/8*sizeof(float));
  dma_set_stepsize(&dma_get_weight, (N - blkN/8)*sizeof(float));

  //DMA for local_output(M/8, N/8)
  dma_set_size(&dma_put_output, blkM*blkN/8/8*sizeof(float));
  dma_set_bsize(&dma_put_output, blkM/8*sizeof(float));
  dma_set_stepsize(&dma_put_output, (M-blkM/8)*sizeof(float));

#ifdef USE_RTC
  GET_RTC(rtc_start);
#endif
  int cur_input_offset = local_input_size;
  int cur_weight_offset = local_weight_size;

  float* input_start = (float*)param->input + rid*blkM/8 + cid*blkK*M/8;
  float* output_start = (float*)param->output + rid*blkM/8 + cid*M*blkN/8;
  float* weight_start = (float*)param->weight + rid*blkK/8*N + cid*blkN/8;

  int cT, cK, cN;
  floatv4 vflt;
  doublev4 vdbl;
  double* dptr;
  float* fptr;

  double gemm_time = 0.;

  int cur_weight_idx = 0;
  dma(dma_get_weight, (long)(weight_start),
      (long)(local_weight + cur_weight_idx*cur_weight_offset));
  dma_wait(&weight_replyget, 1); weight_replyget = 0;
  cur_weight_idx = 1 - cur_weight_idx;

  int cur_input_idx = 0;
  dma(dma_get_input, (long)((float*)input_start),
      (long)(local_input + cur_input_idx*cur_input_offset));
  dma_wait(&input_replyget, 1); input_replyget = 0;
  cur_input_idx = 1 - cur_input_idx;

  for(cN = 0; cN < numN; ++cN) {

    for(cT = 0; cT < T; ++cT) {
      for(i = 0; i < local_output_size; ++i)
        (local_output)[i] = 0;

      for(cK = 0; cK < numK; ++cK) {
        if(cN*numK*T + cT*numK + cK + 1 < numN*T*numK) {
          if(cK == numK-1) {
            if(cT == T-1) {
              dma(dma_get_weight, (long)(weight_start + (cN+1)*blkN),
                (long)(local_weight + cur_weight_idx*cur_weight_offset));
              dma(dma_get_input, (long)((float*)input_start),
                (long)(local_input + cur_input_idx*cur_input_offset));
            }
            else {
              dma(dma_get_weight, (long)(weight_start + cN*blkN),
                (long)(local_weight + cur_weight_idx*cur_weight_offset));
              dma(dma_get_input, (long)((float*)input_start + (cT+1)*blkM),
                (long)(local_input + cur_input_idx*cur_input_offset));

            }
          }
          else {
            dma(dma_get_weight, (long)(weight_start + cN*blkN + (cK+1)*blkK*N),
              (long)(local_weight + cur_weight_idx*cur_weight_offset));
            dma(dma_get_input, (long)((float*)input_start + (cK+1)*M*blkK + blkM*cT),
              (long)(local_input + cur_input_idx*cur_input_offset));
          }
        }
        fptr  = (float*)(local_input + (1 - cur_input_idx)*cur_input_offset);
        dptr = (double*)(local_input + (1 - cur_input_idx)*cur_input_offset);
        for(i = (local_input_size - 4); i>= 0; i -= 4) {
          simd_load(vflt, &fptr[i]);
          vdbl = (doublev4)vflt;
          simd_store(vdbl, &dptr[i]);
        }

        fptr  = (float*)(local_weight + (1 - cur_weight_idx)*cur_weight_offset);
        dptr = (double*)(local_weight + (1 - cur_weight_idx)*cur_weight_offset);
        for(i = (local_weight_size - 4); i >= 0; i -= 4) {
          simd_load(vflt, &fptr[i]);
          vdbl = (doublev4)vflt;
          simd_store(vdbl, &dptr[i]);
        }
        //gemmdouble(
        //ldm_dgemm_trans(
        //ldm_dgemm_trans(
        GET_RTC(rtc_part_start);
        dgemmtransasm(
            (double*)(local_input + (1 - cur_input_idx)*cur_input_offset),
    				(double*)(local_weight + (1 - cur_weight_idx)*cur_weight_offset),
    				(double*)(local_output),
    				blkM/8/4,
    				blkM/8/4,
    				blkN/8,
    				blkK/8,
    				rid,
    				cid);
        GET_RTC(rtc_part_end);
        double kernel_time = (double)(rtc_part_end - rtc_part_start) / (1.45*1024*1024*1024);
        gemm_time += kernel_time;
        //if(cK == 0 && cT == 0 && cN ==0 && id == 0)
        //  printf("gemmkernel time %lf sec %lf Gflops\n", gemm_time, 2*blkN*blkM*blkK/1e9/kernel_time);
        if(cN*numK*T + cT*numK + cK + 1 < numN*T*numK) {
          dma_wait(&input_replyget, 1); input_replyget = 0;
          cur_input_idx = 1 - cur_input_idx;
          dma_wait(&weight_replyget, 1); weight_replyget = 0;
          cur_weight_idx = 1 - cur_weight_idx;
        }
      }//for cK
      float* fptr2  = (float*)(local_output);
      double* dptr2 = (double*)(local_output);
      //TODO: BUG,not compatible with -O2 compile flag
      for(i = 0; i < local_output_size; i += 4) {
        simd_load(vdbl, &dptr2[i]);
        vflt = (floatv4)vdbl;
        simd_store(vflt, &fptr2[i]);
      }
      dma(dma_put_output, (long)((float*)output_start + cT*blkM + cN*blkN*M),
        (long)(local_output));
      dma_wait(&replyput, 1); replyput = 0;
    }//cT
  }
#ifdef USE_RTC
  GET_RTC(rtc_end);
  //rtc_end = slave_get_rtc();
  double t = (double)(rtc_end - rtc_start) / (1.45*1024*1024*1024);
  if ( id == 63 ) {
    printf("rpc_sgemm_trans gflops %lf time %lf sec\n", (double)M*N*blkK*numK*2/1e9/t, t);
  }
#endif

#ifdef _LDM_ALIGN_
  ldm_free(local_input_start, sizeof(double)*local_input_size*2 + LDM_ALINGN_SIZE);
  ldm_free(local_weight_start, sizeof(double)*local_weight_size*2 + LDM_ALINGN_SIZE);
  ldm_free(local_output_start, sizeof(double)*local_output_size + LDM_ALINGN_SIZE);
#else
  ldm_free(local_input, sizeof(double)*local_input_size*2);
  ldm_free(local_weight, sizeof(double)*local_weight_size*2);
  ldm_free(local_output, sizeof(double)*local_output_size);
#endif


#ifdef USE_RTC
  if ( id == 63 ) {
    printf("gemm time %lf sec %lf Gflops\n", gemm_time, 2.*N*M*K/1e9/gemm_time);
  }
#endif
}//main func

