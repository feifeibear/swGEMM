#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include "include/common_slave.h"

/***************
 * GEMM PLAN 
 * Jerry Fang 
 * 2016.Sep.28th
 * On good if Water
 *
 * input  is of dim(B, Ni)
 * weight is of dim(Ni, No)
 * ouput  is of dim(B, No)
 *
 * fully overlap input DMA and weight DMA
 *
 * Oct 16th
 * change weight to 1 pixel read leading to batch-size-version
 *
 * ************/
#define SIMDSIZE 4
void FJR_blas_sgemm_float(ConvData* param)
{
  long long rtc_start, rtc_end;
  int i, cT;
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int Ni = param->Ni;
  int No = param->No;
  int B = param->B;
  int T = param->T;

  double* local_input_start  = (double*) ldm_malloc(sizeof(double)*Ni*B/8/8*2 + 64);
  double* local_input = local_input_start + (long)local_input_start % 256/8;
  int local_input_size = Ni*B/8/8;
//N, K, K, K
  double* local_weight_start = (double*) ldm_malloc(sizeof(double)*Ni*No/8/8 + 32);
  double* local_weight = local_weight_start + (long)local_weight_start % 256/8;
  int local_weight_size = Ni*No/64;
//M, N, Co, Ro
  double* local_output_start = (double*) ldm_malloc(sizeof(double)*No*B/8/8*2 + 64);
  double* local_output = local_output_start + (long)local_output_start % 256/8;
  int local_output_size = No*B/8/8;


//  float local_weight[K*K*Ni/64*No];
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

  //DMA for local_iutput(B/8, Ni/8)
  dma_set_size(&dma_get_input, B*Ni/8/8/SIMDSIZE*sizeof(floatv4));
  dma_set_bsize(&dma_get_input, Ni/SIMDSIZE/8*sizeof(floatv4));
  dma_set_stepsize(&dma_get_input, Ni/SIMDSIZE/8*7*sizeof(floatv4));

  //DMA for local_weight(Ni/8, No/8)
  dma_set_size(&dma_get_weight, No*Ni/8/8*sizeof(float));
  dma_set_bsize(&dma_get_weight, No/8*sizeof(float));
  dma_set_stepsize(&dma_get_weight, No/8*7*sizeof(float));

  //DMA for local_output(B/8, No/8)
  dma_set_size(&dma_put_output, B*No/8/8/SIMDSIZE*sizeof(floatv4));
  dma_set_bsize(&dma_put_output, No/SIMDSIZE/8*sizeof(floatv4));
  dma_set_stepsize(&dma_put_output, No/SIMDSIZE/8*7*sizeof(floatv4));

  //fetch weight into LDM and use it all the time
  GET_RTC(rtc_start);
  dma(dma_get_weight, (long)((float*)param->weight+(cid*No/8*Ni + rid*No/8)), (long)(local_weight));
  dma_wait(&weight_replyget, 1); weight_replyget = 0;

  int cur_input_idx = 0;
  int cur_output_idx = 0;
  int cur_offset = local_input_size;
  int cur_output_offset = local_output_size;

  float* input_start = (float*)param->input + cid*Ni*B/8 + rid*Ni/8;
  float* output_start = (float*)param->output + cid*No*B/8 + rid*No/8;

  dma(dma_get_input, (long)(input_start), (long)(local_input + cur_input_idx*cur_offset));
  dma_wait(&input_replyget, 1); input_replyget = 0;
  cur_input_idx = 1 - cur_input_idx;

  for(cT = 0; cT < T; ++cT) {
    if(cT+1 < T)
      dma(dma_get_input, (long)(input_start + (cT+1)*Ni*B), (long)(local_input + cur_input_idx*cur_offset));
    for(i = 0; i < local_output_size; ++i)
      (local_output + cur_output_idx*cur_output_offset)[i] = 0.0;

#ifdef USE_COMP
    gemmfloat((float*)(local_weight),
				(float*)(local_input + (1-cur_input_idx)*cur_offset),
				(float*)(local_output + cur_output_idx*cur_output_offset),
				No/8/4,
				No/8/4,
				B/8,
				Ni/8,
				rid,
				cid);
#endif

    if(cT+1 < T) {
      dma_wait(&input_replyget, 1); input_replyget = 0;
      cur_input_idx = 1-cur_input_idx;
    }
    if(cT > 0) {
      dma_wait(&replyput, 1); replyput = 0;
    }
    dma(dma_put_output, (long)(output_start + cT*No*B), (long)(local_output + cur_output_idx*cur_output_offset));
    cur_output_idx = 1 - cur_output_idx;
  }
  dma_wait(&replyput, 1); replyput = 0;

  //rtc_end = slave_get_rtc();
  GET_RTC(rtc_end);
  double t = (double)(rtc_end - rtc_start) / (1.45*1024*1024*1024);
  //double bw = total_len * 8  / t / 1024/ 1024/ 1024;
  if ( id == 63 ) {
    printf("rpc_sgemm_float gflops %lf \n", (double)Ni*No*B*T*2/1024/1024/1024/t);
  }


  ldm_free(local_input, sizeof(double)*local_input_size*2 + 64);
  ldm_free(local_weight, sizeof(double)*local_weight_size + 32);
  ldm_free(local_output, sizeof(double)*local_output_size*2 + 64);

}//main func

