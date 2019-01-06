#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include "include/common_slave.h"

/***************
 * GEMM PLAN 
 * Jiarui Fang 
 * 2018.Dec.9th
 * input  is of dim(B, Ni)
 * weight is of dim(Ni, No) cached
 * ouput  is of dim(B, No)
 *
 * ************/

#define SIMDSIZE 4
void FJR_blas_sgemm_small(ConvData* param)
{
  long long rtc_start, rtc_end;
  double t = 0.;

#ifdef USE_RTC
  GET_RTC(rtc_start);
#endif
  int i;
  int id = athread_get_id(-1);
  int Ni = param->blkK;
  int No = param->blkN;
  int B = param->blkM;
  int T = param->M/B;

  float* local_input = (float*)(floatv4*)ldm_malloc(sizeof(float)*Ni*B/64*2);
  int local_input_size = Ni*B/64;
//N, K, K, K
  float* local_weight = (float*) (doublev4*)ldm_malloc(sizeof(float)*Ni*No);
  int local_weight_size = Ni*No;
//M, N, Co, Ro
  float* local_output = (float*) (doublev4*)ldm_malloc(sizeof(float)*B*No/64*2);
  int local_output_size = No*B/64;


//  double local_weight[K*K*Ni/64*No];
//initilize DMA variables
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

  dma_set_size(&dma_get_input, B*Ni/64*sizeof(float));
  dma_set_size(&dma_get_weight, No*Ni*sizeof(float));
  dma_set_size(&dma_put_output, B*No/64*sizeof(float));

#ifdef USE_RTC
  GET_RTC(rtc_end);
  t = (double)(rtc_end - rtc_start) / (1.45*1024*1024*1024);
  //double bw = total_len * 8  / t / 1024/ 1024/ 1024;
  if ( id == 63 ) {
    printf("define %lf sec\n", t);
  }
  //rtc_start = slave_get_rtc();
  ALLSYN;
  GET_RTC(rtc_start);
#endif
  //fetch weight into LDM and use it all the time
  dma(dma_get_weight, (long)((float*)param->weight), (long)(local_weight));
  dma_wait(&weight_replyget, 1); weight_replyget = 0;

  int cur_input_idx = 0;
  int cur_output_idx = 0;
  int cur_input_offset = local_input_size;
  int cur_output_offset = local_output_size;

  float* input_start = (float*)param->input + id*Ni*B/64;
  float* output_start = (float*)param->output + id*No*B/64;

  dma(dma_get_input, (long)(input_start), (long)(local_input + cur_input_idx*cur_input_offset));
  dma_wait(&input_replyget, 1); input_replyget = 0;
  cur_input_idx = 1 - cur_input_idx;

  int cT;
  int m, n, k;
  for(cT = 0; cT < T; ++cT) {
    if(cT+1 < T)
      dma(dma_get_input, (long)((float*)input_start + (cT+1)*Ni*B), (long)(local_input + cur_input_idx*cur_input_offset));
    for(i = 0; i < local_output_size; ++i)
      (local_output + cur_output_idx*cur_output_offset)[i] = 0;

#ifdef USE_COMP
    //ldm_sgemm_small(
    ldm_sgempasm(
        (local_weight),
				(local_input + (1 - cur_input_idx)*cur_input_offset),
				(local_output + cur_output_idx*cur_output_offset),
				No/4,
				No/4,
				B/64,
				Ni);

    /*
    for(m = 0; m < B/64; ++m)
      for(n = 0; n < No; ++n)
        for(k = 0; k < Ni; ++k) {
          (local_output + cur_output_idx*cur_output_offset)[m*No + n] += (local_input + (1 - cur_input_idx)*cur_input_offset)[m*Ni + k] * local_weight[n*Ni + k];
        }
        */
#endif

    if(cT+1 < T) {
      dma_wait(&input_replyget, 1); input_replyget = 0;
      cur_input_idx = 1-cur_input_idx;
    }
    if(cT > 0) {
      dma_wait(&replyput, 1); replyput = 0;
    }
    dma(dma_put_output, (long)((float*)output_start + cT*No*B), (long)(local_output + cur_output_idx*cur_output_offset));
    cur_output_idx = 1 - cur_output_idx;
  }
  dma_wait(&replyput, 1); replyput = 0;

#ifdef USE_RTC
  //rtc_end = slave_get_rtc();
  GET_RTC(rtc_end);
  t = (double)(rtc_end - rtc_start) / (1.45*1024*1024*1024);
  //double bw = total_len * 8  / t / 1024/ 1024/ 1024;
  if ( id == 63 ) {
    printf("rpc_sgemm gflops %lf \n", (double)Ni*No*B*T*2/1024/1024/1024/t);
    printf("comp %lf sec\n", t);
  }
#endif

  ldm_free(local_input, sizeof(float)*local_input_size*2);
  ldm_free(local_weight, sizeof(float)*local_weight_size);
  ldm_free(local_output, sizeof(float)*local_output_size*2);

}//main func

