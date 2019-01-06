#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include "include/common_slave.h"

void sync_array(){
    int256 sync_tmp;
    asm volatile(\
        "ldi    %0, 0xff\n"   \
        "sync   %0\n"   \
        "synr   %0\n"   \
        :   \
        :"r"(sync_tmp):"memory");
}

void sync_row(){
    int256 sync_tmp;
    asm volatile(\
        "ldi    %0, 0xff\n"   \
        "synr   %0\n"   \
        :   \
        :"r"(sync_tmp):"memory");
}



/***************
 * Jerry Fang 
 * ************/
#define SIMDSIZE 4
#define USE_RANKMODE
//#define TEST_USE_RANKMODE
void FJR_blas_sgemm_trans_rank(ConvData* param)
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
  double* local_input_start  = (double*) ldm_malloc(sizeof(double)*K*M/8/8*2 + 64);
  double* local_input = local_input_start + (long)local_input_start % 256/8;
  int local_input_size = K*M/8/8;
//N, K, K, K
  double* local_weight_start = (double*) ldm_malloc(sizeof(double)*K*N/8/8 + 32);
  double* local_weight = local_weight_start + (long)local_weight_start % 256/8;
  int local_weight_size = K*N/64;
//M, N, Co, Ro
  double* local_output_start = (double*) ldm_malloc(sizeof(double)*N*M/8/8*2 + 64);
  double* local_output = local_output_start + (long)local_output_start % 256/8;
  int local_output_size = N*M/8/8;

  volatile int  input_replyget = 0, weight_replyget = 0,  replyput = 0;
  dma_desc dma_get_input, dma_get_weight, dma_get_output, dma_put_output;
#ifdef TEST_USE_RANKMODE
  dma_desc dma_test_put;
  volatile int test_replyput = 0;

  dma_set_op(&dma_test_put, DMA_PUT);
  dma_set_mode(&dma_test_put, PE_MODE);
  dma_set_reply(&dma_test_put, &test_replyput);

  dma_set_size(&dma_test_put, M*K/8/8*sizeof(float));
  dma_set_bsize(&dma_test_put, M/8*sizeof(float));
  dma_set_stepsize(&dma_test_put, (M-M/8)*sizeof(float));
#endif

  dma_set_op(&dma_get_input, DMA_GET);
#ifdef USE_RANKMODE
  dma_set_mode(&dma_get_input, RANK_MODE);
#else
  dma_set_mode(&dma_get_input, PE_MODE);
#endif
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_weight, DMA_GET);
  dma_set_mode(&dma_get_weight, PE_MODE);
  dma_set_reply(&dma_get_weight, &weight_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
#ifdef USE_RANKMODE
  dma_set_mode(&dma_put_output, RANK_MODE);
#else
  dma_set_mode(&dma_put_output, PE_MODE);
#endif
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_iutput(M/8, K/8)
  dma_set_size(&dma_get_input, M*K/8/8*sizeof(float));
#ifdef USE_RANKMODE
  dma_set_bsize(&dma_get_input, M/8*sizeof(float));
  dma_set_stepsize(&dma_get_input, sizeof(float)*0);
#else
  dma_set_bsize(&dma_get_input, M/8*sizeof(float));
  dma_set_stepsize(&dma_get_input, (M-M/8)*sizeof(float));
#endif

  //DMA for local_weight(K/8, N/8)
  dma_set_size(&dma_get_weight, N*K/8/8*sizeof(float));
  dma_set_bsize(&dma_get_weight, N/8*sizeof(float));
  dma_set_stepsize(&dma_get_weight, N/8*7*sizeof(float));

  //DMA for local_output(M/8, N/8)
  dma_set_size(&dma_put_output, M*N/8/8*sizeof(float));
#ifdef USE_RANKMODE
  dma_set_bsize(&dma_put_output, M/8*sizeof(float));
  dma_set_stepsize(&dma_put_output, 0);
#else
  dma_set_bsize(&dma_put_output, M/8*sizeof(float));
  dma_set_stepsize(&dma_put_output, (M-M/8)*sizeof(float));
#endif

  GET_RTC(rtc_start);
  //fetch weight into LDM and use it all the time
  dma(dma_get_weight, (long)((float*)param->weight+(cid*K/8*N + rid*N/8)), (long)(local_weight));
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
#ifdef USE_RANKMODE
  float* input_start = (float*)param->input + id*K*M/64;
  float* output_start = (float*)param->output + (id*N*M/64);
#else
  float* input_start = (float*)param->input + (rid*M*K/8 + cid*M/8);
  float* output_start = (float*)param->output + (rid*N*M/8 + cid*M/8);
#endif

#ifdef USE_RANKMODE
  dma(dma_get_input, (long)(input_start),
      (long)(local_input + cid*M*K/64/8/2 + cur_input_idx*cur_input_offset));
#else
  dma(dma_get_input, (long)(input_start),
      (long)(local_input + cur_input_idx*cur_input_offset));
#endif
  dma_wait(&input_replyget, 1); input_replyget = 0;
  sync_row();

  cur_input_idx = 1 - cur_input_idx;
  int cT;
  for(cT = 0; cT < T; ++cT) {
    if(cT+1 < T) {
#ifdef USE_RANKMODE
      dma(dma_get_input, (long)((float*)input_start + (cT+1)*M*K),
          (long)(local_input + cur_input_idx*cur_input_offset + cid*M*K/8/8/8/2));
#else
      dma(dma_get_input, (long)((float*)input_start + (cT+1)*M*K),
          (long)(local_input + cur_input_idx*cur_input_offset));
#endif
    }
    for(i = 0; i < local_output_size; ++i)
      (local_output + cur_output_idx*cur_output_offset)[i] = 0;


    float* fptr  = (float*)(local_input + (1 - cur_input_idx)*cur_input_offset);
    double* dptr = (double*)(local_input + (1 - cur_input_idx)*cur_input_offset);
    for(i = (local_input_size - 4); i>= 0; i -= 4) {
      simd_load(vflt, &fptr[i]);
      vdbl = (doublev4)vflt;
      simd_store(vdbl, &dptr[i]);
    }

    //for(i =(1-cur_input_idx)*cur_input_offset; i < (1-cur_input_idx)*cur_input_offset + local_input_size; ++i)
    //  if(local_input[i] != 1.0)
    //    printf("%lf\n", local_input[i]);
#ifdef USE_COMP
    /*
    ldm_dgemm_mnn(
				(double*)(local_weight),
        (double*)(local_input + (1-cur_input_idx)*cur_input_offset),
				(double*)(local_output + cur_output_idx*cur_output_offset),
				N/8,
				N/8,
				M/8/4,
				K/8,
				rid,
				cid);
        */

    dgemmasm(
				(double*)(local_weight),
        (double*)(local_input + (1-cur_input_idx)*cur_input_offset),
				(double*)(local_output + cur_output_idx*cur_output_offset),
				N/8/4,
				N/8/4,
				M/8,
				K/8,
				rid,
				cid);
#endif


    float* fptr2  = (float*)(local_output + cur_output_idx*cur_output_offset);
    double* dptr2 = (double*)(local_output + cur_output_idx*cur_output_offset);


//TODO: BUG,not compatible with -O2 compile flag
    for(i = 0; i < local_output_size; i += 4) {
      simd_load(vdbl, &dptr2[i]);
      vflt = (floatv4)vdbl;
      simd_store(vflt, &fptr2[i]);
    }

    if(cT+1 < T) {
      dma_wait(&input_replyget, 1); input_replyget = 0;
#ifdef USE_RANKMODE
      sync_row();
#endif
      cur_input_idx = 1-cur_input_idx;
    }
    if(cT > 0) {
      dma_wait(&replyput, 1); replyput = 0;
#ifdef USE_RANKMODE
      sync_row();
#endif
    }
#ifdef USE_RANKMODE
    dma(dma_put_output, (long)((float*)output_start + cT*M*N),
        (long)(local_output + cid*M*N/64/8/2 + cur_output_idx*cur_output_offset));
#else
    dma(dma_put_output, (long)((float*)output_start + cT*M*N),
        (long)(local_output + cur_output_idx*cur_output_offset));
#endif
    cur_output_idx = 1 - cur_output_idx;
  }
  dma_wait(&replyput, 1); replyput = 0;
#ifdef USE_RANKMODE
  sync_row();
#endif
  GET_RTC(rtc_end);
  double t = (double)(rtc_end - rtc_start) / (1.45*1024*1024*1024);
  if ( id == 63 ) {
    printf("rpc_sgemm_trans_row gflops %lf \n", (double)M*N*K*T*2/1024/1024/1024/t);
  }
#ifdef TEST_USE_RANKMODE
  dma(dma_test_put, (long)((float*)param->output + (rid*M*K/8 + cid*M/8)),
    (long)(local_input));
  dma_wait(&test_replyput, 1); test_replyput = 0;
#endif

  ldm_free(local_input, sizeof(double)*local_input_size*2 + 64);
  ldm_free(local_weight, sizeof(double)*local_weight_size + 32);
  ldm_free(local_output, sizeof(double)*local_output_size*2 + 64);

}//main func

