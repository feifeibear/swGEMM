#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include "include/common_slave.h"

//inline uint64_t slave_get_rtc()
//{
//  unsigned long rpcc;
//  asm volatile ("rcsr %0, 4":"=&r"(rpcc)::"memory");
//  return rpcc;
//}



/***************
 * GEMM PLAN 
 * Jerry Fang 
 * 2016.Sep.28th
 * On good if Water
 *
 * input  is of dim(K, M)
 * weight is of dim(K, N)
 * ouput  is of dim(N, M)
 *
 * fully overlap input DMA and weight DMA
 * ************/
#define SIMDSIZE 4
void FJR_blas_sgemm_trans(ConvData* param)
{
  long long rtc_start, rtc_end;
  int i;
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int K = param->Ni;
  int N = param->No;
  int M = param->B;
  int T = param->T;

//M, K, Ci, Ri
  char* local_input_start  = (char*)ldm_malloc(sizeof(double)*K*M/8/8*2 + 16);
  double* local_input = (double*)(local_input_start + 16- (long)local_input_start%16);
  int local_input_size = K*M/8/8;
//N, K, K, K
  char* local_weight_start = (char*)ldm_malloc(sizeof(double)*K*N/8/8 + 16);
  double* local_weight = (double*)(local_weight_start + 16 - (long)local_weight_start%16);
  int local_weight_size = K*N/64;
//M, N, Co, Ro
  char* local_output_start = (char*)ldm_malloc(sizeof(double)*N*M/8/8*2 + 16);
  double* local_output = (double*)(local_output_start + 16 - (long)local_output_start%16);
  int local_output_size = N*M/8/8;

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
  dma_set_size(&dma_get_input, M*K/8/8*sizeof(float));
  dma_set_bsize(&dma_get_input, M/8*sizeof(float));
  dma_set_stepsize(&dma_get_input, (M*T-M/8)*sizeof(float));

  //DMA for local_weight(K/8, N/8)
  dma_set_size(&dma_get_weight, N*K/8/8*sizeof(float));
  dma_set_bsize(&dma_get_weight, N/8*sizeof(float));
  dma_set_stepsize(&dma_get_weight, N/8*7*sizeof(float));

  //DMA for local_output(M/8, N/8)
  dma_set_size(&dma_put_output, M*N/8/8*sizeof(float));
  dma_set_bsize(&dma_put_output, M/8*sizeof(float));
  dma_set_stepsize(&dma_put_output, (M*T-M/8)*sizeof(float));

//  rtc_start = slave_get_rtc();
  GET_RTC(rtc_start);
  //fetch weight into LDM and use it all the time
  dma(dma_get_weight, (long)((float*)param->weight+(rid*K/8*N +cid*N/8)), (long)(local_weight));
  dma_wait(&weight_replyget, 1); weight_replyget = 0;

  floatv4 vflt;
  doublev4 vdbl;
  float* fptr  = (float*)local_weight;
  double* dptr = (double*)local_weight;
  for(i = (local_weight_size-4); i >= 0; i -= 4) {
    simd_load(vflt, &fptr[i]);
    vdbl = (doublev4)vflt;
    simd_store(vdbl, &dptr[i]);
  }

  int cur_input_idx = 0;
  int cur_output_idx = 0;
  int cur_input_offset = local_input_size;
  int cur_output_offset = local_output_size;

  float* input_start = (float*)param->input + cid*K*M*T/8 + rid*M/8;
  float* output_start = (float*)param->output + rid*M/8 + cid*M*T*N/8;

  dma(dma_get_input, (long)(input_start), 
      (long)(local_input + cur_input_idx*cur_input_offset));
  dma_wait(&input_replyget, 1); input_replyget = 0;
  cur_input_idx = 1 - cur_input_idx;

  int cT;
  for(cT = 0; cT < T; ++cT) {
    if(cT+1 < T)
      dma(dma_get_input, (long)((float*)input_start + (cT+1)*M), (long)(local_input + cur_input_idx*cur_input_offset));
    for(i = 0; i < local_output_size; ++i)
      (local_output + cur_output_idx*cur_output_offset)[i] = 0;


    float* fptr  = (float*)(local_input + (1 - cur_input_idx)*cur_input_offset);
    double* dptr = (double*)(local_input + (1 - cur_input_idx)*cur_input_offset);
    for(i = (local_input_size - 4); i>= 0; i -= 4) {
      simd_load(vflt, &fptr[i]);
      vdbl = (doublev4)vflt;
      simd_store(vdbl, &dptr[i]);
    }

    float* fptr2  = (float*)(local_output + cur_output_idx*cur_output_offset);
    double* dptr2 = (double*)(local_output + cur_output_idx*cur_output_offset);

#ifdef USE_COMP
    //gemmdouble(
    //ldm_dgemm_trans(
    //ldm_dgemm_trans(
    dgemmtransasm(
        (double*)(local_input + (1-cur_input_idx)*cur_input_offset),
				(double*)(local_weight),
				(double*)(local_output + cur_output_idx*cur_output_offset),
				M/8/4,
				M/8/4,
				N/8,
				K/8,
				rid,
				cid);
#endif

//TODO: BUG,not compatible with -O2 compile flag
    for(i = 0; i < local_output_size; i += 4) {
      simd_load(vdbl, &dptr2[i]);
      vflt = (floatv4)vdbl;
      simd_store(vflt, &fptr2[i]);
    }

    if(cT+1 < T) {
      dma_wait(&input_replyget, 1); input_replyget = 0;
      cur_input_idx = 1-cur_input_idx;
    }
    if(cT > 0)
      dma_wait(&replyput, 1); replyput = 0;
    dma(dma_put_output, (long)((float*)output_start + cT*M), 
        (long)(local_output + cur_output_idx*cur_output_offset));
    cur_output_idx = 1 - cur_output_idx;
  }
  dma_wait(&replyput, 1); replyput = 0;

  GET_RTC(rtc_end);
  //rtc_end = slave_get_rtc();
  double t = (double)(rtc_end - rtc_start) / (1.45*1024*1024*1024);
  if ( id == 63 ) {
    printf("rpc_sgemm_trans gflops %lf \n", (double)M*N*K*T*2/1024/1024/1024/t);
  }

  ldm_free(local_input, sizeof(double)*local_input_size*2 + 16);
  ldm_free(local_weight, sizeof(double)*local_weight_size + 16);
  ldm_free(local_output, sizeof(double)*local_output_size*2 + 16);

}//main func

