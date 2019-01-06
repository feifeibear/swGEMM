#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include "./include/common_slave.h"
#include <cblas.h>

//#define PART_TIME
extern void SLAVE_FUN(FJR_blas_sgemm)();
extern void SLAVE_FUN(FJR_blas_sgemm_float)();
extern void SLAVE_FUN(FJR_blas_sgemm_trans_implicit)();
//output information in this function, use for debug, may get wrong results
extern void SLAVE_FUN(FJR_blas_sgemm_trans_test_perfmdl)();
extern void SLAVE_FUN(FJR_blas_sgemm_trans_compute_only)();
extern void SLAVE_FUN(FJR_blas_sgemm_trans_dma_only)();
//final version
extern void SLAVE_FUN(FJR_blas_sgemm_trans_dma_full_pipeline)();

/*
 * get DMA time
 * BM
 */
double get_DMA_time(int BL, int BH) {
    double a = 1.00596957e-09;
    double b = 3.14618755e-07;
    double c = 9.89289916e-07;
    return (a*BL*BH + b*BL + c)/10;
}

void sw_cblas_sgemm_unit(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc) {
  struct timeval t1, t2;
  //3362.3000000000002, 6342.6000000000004
  double MBW_map[]={9362.3000000000002, 9342.6000000000004, 9091.3999999999996, 11966.799999999999, 14464.4, 10109.4, 10826.799999999999, 13355.9, 14225.6, 16268.0, 17285.200000000001, 19322.400000000001, 20039.099999999999, 8748.6000000000004, 16397.0, 17568.099999999999, 18846.599999999999, 19078.799999999999, 17884.799999999999, 21040.299999999999, 21277.799999999999, 18181.299999999999, 18960.400000000001, 19724.799999999999, 20330.599999999999, 21263.700000000001, 21535.799999999999, 11486.1, 22908.099999999999, 19666.900000000001, 20302.700000000001, 21102.5, 21682.700000000001, 21875.700000000001, 22555.200000000001, 23501.799999999999, 21774.299999999999, 20105.700000000001, 21358.700000000001, 21932.099999999999, 21482.5, 19173.299999999999, 22579.200000000001, 23836.799999999999, 23775.400000000001, 21602.5, 21919.700000000001, 22429.5, 22826.400000000001, 23273.799999999999, 23630.599999999999, 24175.700000000001, 24429.099999999999, 22369.0, 21537.200000000001, 20850.400000000001, 21515.099999999999, 23762.900000000001, 23600.599999999999, 24484.400000000001, 24604.900000000001, 22800.0, 22921.599999999999, 23484.599999999999, 3390.3000000000002, 6033.3000000000002, 9180.6000000000004, 11876.4, 13766.799999999999, 10086.6, 10522.6, 13365.1, 14133.6, 16451.5, 17313.799999999999, 19445.799999999999, 19996.900000000001, 10185.9, 16442.299999999999, 17509.299999999999, 18274.200000000001, 19345.099999999999, 19300.099999999999, 20965.700000000001, 20429.400000000001, 18137.299999999999, 18710.0, 19884.799999999999, 20117.299999999999, 21051.299999999999, 21009.099999999999, 12401.799999999999, 22579.200000000001, 19770.599999999999, 20368.599999999999, 21030.200000000001, 21637.900000000001, 22160.900000000001, 22898.099999999999, 23052.599999999999, 22213.900000000001, 18106.900000000001, 21088.0, 21992.099999999999, 22231.400000000001, 18906.799999999999, 22901.299999999999, 23413.0, 23620.400000000001, 21629.400000000001, 21916.799999999999, 22201.5, 22933.200000000001, 23258.599999999999, 23628.900000000001, 24179.299999999999, 24568.700000000001, 22159.0, 22024.0, 21703.700000000001, 22000.900000000001, 23542.5, 23898.5, 24653.200000000001, 24752.200000000001, 22704.700000000001, 22908.099999999999, 23676.799999999999, 23607.799999999999, 24061.099999999999, 24265.099999999999, 24463.0, 24885.200000000001, 22663.700000000001, 23200.099999999999, 23771.5, 23822.200000000001, 23878.200000000001, 24192.599999999999, 24364.599999999999, 24847.700000000001, 23300.099999999999, 23664.5, 23730.5, 24296.799999999999, 24190.299999999999, 24188.599999999999, 24158.900000000001, 24806.599999999999, 23510.599999999999, 23840.200000000001, 24218.299999999999, 24235.200000000001, 24448.099999999999, 24828.099999999999, 25147.5, 25309.599999999999, 23574.0, 23646.700000000001, 23983.700000000001, 24372.200000000001, 24844.0, 24924.700000000001, 25109.700000000001, 25378.700000000001, 23925.400000000001, 24214.799999999999, 24551.299999999999, 24587.900000000001, 24818.299999999999, 25051.5, 25338.299999999999, 25317.400000000001, 24099.799999999999, 24158.900000000001, 24436.799999999999, 23581.200000000001, 24539.900000000001, 24685.900000000001, 24942.099999999999, 25381.200000000001, 24162.900000000001, 24270.900000000001, 24740.599999999999, 24684.099999999999, 24816.400000000001, 25079.599999999999, 25461.5, 25446.700000000001, 23707.400000000001, 23754.0, 24641.700000000001, 24930.400000000001, 25199.900000000001, 25365.099999999999, 25548.299999999999, 25656.099999999999, 24629.700000000001, 24703.0, 24873.5, 24806.599999999999, 25210.400000000001, 24867.400000000001, 24863.200000000001, 25700.5, 24606.799999999999, 24904.5, 24893.200000000001, 25083.400000000001, 25196.0, 25277.099999999999, 25601.799999999999, 24222.299999999999, 24871.599999999999, 24890.799999999999, 24978.0, 24897.0, 24061.099999999999, 25350.400000000001, 25539.799999999999, 25816.200000000001, 24777.299999999999, 25125.5, 24796.799999999999, 25261.099999999999, 25498.400000000001, 25510.700000000001, 25738.099999999999, 25643.599999999999, 24911.099999999999, 24659.200000000001, 25003.599999999999, 25063.900000000001, 25409.5, 25566.599999999999, 26173.099999999999, 25681.5, 24767.5, 25033.900000000001, 25204.200000000001, 25141.799999999999, 25419.799999999999, 25579.5, 25675.5, 25702.5, 25407.5, 24873.5, 25331.900000000001, 25430.599999999999, 24968.099999999999, 24645.400000000001, 25573.0, 25950.200000000001, 25004.099999999999, 25081.5, 25333.900000000001, 25364.599999999999, 25558.599999999999, 25914.599999999999, 25814.200000000001, 25803.599999999999, 25248.599999999999, 25129.799999999999, 25356.799999999999, 25370.900000000001, 25616.700000000001, 25525.0, 25742.700000000001, 25208.0, 25129.299999999999, 25232.599999999999, 25451.200000000001, 25446.700000000001, 25710.5, 25742.700000000001, 25948.700000000001, 25412.0, 25397.299999999999, 25393.400000000001, 25510.700000000001, 25635.599999999999, 25446.700000000001, 25925.299999999999, 25914.099999999999, 25230.700000000001, 25107.299999999999, 23748.400000000001, 25196.0, 24577.900000000001, 25139.400000000001, 25759.200000000001, 25725.599999999999, 25258.700000000001, 25432.0, 25490.0, 25510.700000000001, 25539.400000000001, 25746.700000000001, 25806.099999999999, 25706.5, 25155.700000000001, 25206.099999999999, 25381.200000000001, 25307.200000000001, 25506.299999999999, 25769.799999999999, 25683.5, 25912.0, 24691.5, 25299.400000000001, 25432.5, 25417.799999999999, 25702.5, 25772.299999999999, 24928.5, 26051.700000000001, 25049.599999999999, 25459.5, 25444.799999999999};
  ConvData* params = (ConvData*)malloc(sizeof(ConvData));

  assert(Order == CblasRowMajor);
  assert(alpha == 1. && beta == 0);

  if(TransA == CblasTrans && TransB == CblasNoTrans) {
    assert(lda == M && ldb == N && ldc == N);
    //TODO 
    //serach for best align_size and blk_size 
    //align_size is equal to blk size
    //bsizeN range from 64B ~ 2KB 
    //bsizeM range from 16B ~ 2KB
    //MBW_map should be indexed as 16Bx

    printf("M %d K %d N %d\n", M, K, N);
    int N_align_size = 128; //128x
    int K_align_size = 32; //8x large than 16
    int M_align_size = 64; //32x

    int blkN, blkM, blkK;
    double est_best_perf = 0.;
    double real_best_perf = 0.;
    double find_best_perf = 0.;
    int real_blkN, real_blkM, real_blkK;
    for (blkN = 128; blkN <= N; blkN += 128)
      for (blkM = 32; blkM <= M; blkM += 32)
        for (blkK = 32; blkK <= K; blkK += 8)
    {
          int ldm_use = sizeof(double)*(blkN*blkK*2 + 
              blkK*blkM*2 + blkN*blkM)/64;
          if (ldm_use < 60*1024 && N%blkN == 0 && M%blkM == 0 && K%blkK == 0) {
            double gflop = 2.0*N*K*M;
            // run real code
            params->input = (void*)B;
            params->weight = (void*)A;
            params->output = (void*)C;
            params->K = K;
            params->blkK = blkK;
            params->N = M;
            params->blkN = blkM;
            params->M = N;
            params->blkM = blkN;

            gettimeofday(&t1, NULL);
            if(params->blkM%128 == 0 && params->blkN%32 == 0 && params->blkK%8 == 0){
              athread_spawn(FJR_blas_sgemm_trans_dma_full_pipeline, params);
              athread_join();
            }
            gettimeofday(&t2, NULL);
            double real_time = TIME(t1,t2);
            double real_perf = gflop/1e9/real_time;

#ifdef PART_TIME
            if(params->blkM%128 == 0 && params->blkN%32 == 0 && params->blkK%8 == 0){
              athread_spawn(FJR_blas_sgemm_trans_test_perfmdl, params);
              athread_join();
            }

            if(params->blkM%128 == 0 && params->blkN%32 == 0 && params->blkK%8 == 0){
              athread_spawn(FJR_blas_sgemm_trans_compute_only, params);
              athread_join();
            }

            if(params->blkM%128 == 0 && params->blkN%32 == 0 && params->blkK%8 == 0){
              athread_spawn(FJR_blas_sgemm_trans_dma_only, params);
              athread_join();
            }
#endif
            //performance model achieves an estimited performance
            int bsizeN = blkN/8*sizeof(float);
            int bsizeM = blkM/8*sizeof(float);
            double T_dma = N/blkN*M/blkM*K/blkK*(1.0*blkN*blkK*sizeof(float)/1e6/MBW_map[bsizeN/16 - 1] + 
              1.0*blkM*blkK*sizeof(float)/1e6/MBW_map[bsizeN/16 - 1]) +
              1.0*N/blkN*M/blkM*blkM*blkN*sizeof(float)/1e6/MBW_map[bsizeN/16 - 1];

            //double T_dma = N/blkN*M/blkM*K/blkK*(get_DMA_time(blkM, blkK) + get_DMA_time(blkN, blkK))
            //  + 1.0*N/blkN*M/blkM*get_DMA_time(blkM, blkN);

            //double EE = blkK/(18. + 17./16.*blkK);
            //double T_compute = gflop/1e9/742.4/EE;
            //double T_compute = (8*(17.0*blkK/8 + a)*blkM/128.*blkN/32. + b)/(1.45*1e9)
            //double a = 9.89241468e-09;
            //double b = 4.72407311e-10;
            //double c = 3.85722600e-11;
            //double d = 1.28410695e-05;
            double a = 9.55371467e-09;
            double b = 4.80294349e-10;
            double c = 3.85210279e-11;
            double d = 1.36105221e-05;
            double T_compute = (a*blkN + b*blkM*blkN + c*blkM*blkK*blkN + d)/10
              *M/blkM*K/blkK*N/blkN;
            double T_init_dma = (1.0*blkN*blkK*sizeof(float)/1e6/MBW_map[bsizeN/16-1] + 
              1.0*blkM*blkK*sizeof(float)/1e6/MBW_map[bsizeM/16-1]);
            //TODO
            double T_f2d = (blkM*blkK/64 + blkN*blkK/64)*(sizeof(float) + sizeof(double))/(46.4*1e9)
              *M/blkM*K/blkK*N/blkN;
            double est_time = MAX(T_compute, T_dma) + T_init_dma;
            double est_perf = gflop/1e9/est_time;

            if(est_perf > est_best_perf) {
              est_best_perf = est_perf;
              find_best_perf = real_perf;
              N_align_size = blkN;
              M_align_size = blkM;
              K_align_size = blkK;
            }

            //record best perf
            if(real_perf > real_best_perf) {
              real_best_perf = real_perf;
              real_blkN = blkN;
              real_blkM = blkM;
              real_blkK = blkK;
            }
#ifdef PART_TIME
            printf("ldm use %d B, T_compute_f2d %lf, T_dma %lf sec, T_real %lf\n", ldm_use, T_compute, T_dma, real_time);
            printf("estimited perf %lf GFlops, real perf %lf Gflops, blkN %d blkM %d blkK %d\n", 
                est_perf, real_perf, blkN, blkM, blkK);
            printf("====================\n");
#endif
          }
    }
    printf("real_best_perf %lf  Gflops, real_blkN %d real_blkM %d real_blkK %d\n", real_best_perf, 
        real_blkN, real_blkM, real_blkK);
    printf("find_best_perf %lf  Gflops, real_blkN %d real_blkM %d real_blkK %d\n", find_best_perf, 
        N_align_size, M_align_size, K_align_size);
    assert(N_align_size != 0 && K_align_size != 0 && M_align_size != 0);

    int N_pad = (N+N_align_size-1)/N_align_size*N_align_size;
    int K_pad = (K+K_align_size-1)/K_align_size*K_align_size;
    int M_pad = (M+M_align_size-1)/M_align_size*M_align_size;

    gettimeofday(&t1, NULL);
    float* A_zeropad;
    if(M != M_pad || K != K_pad) {
      A_zeropad = (float*)_aligned_malloc(sizeof(float)*M_pad * K_pad, 128);
      sw_zeropad_matrix(A, M, M_pad, K, K_pad, A_zeropad);
    }
    else
      A_zeropad = A;

    gettimeofday(&t2, NULL);
    double pad_time = TIME(t1,t2);
    //printf("sw_zeropad_matrix A time %lf sec\n", pad_time);
    //
     gettimeofday(&t1, NULL);
    float* A_zeropad_mpe;
    if(M != M_pad || K != K_pad) {
      A_zeropad_mpe = (float*)_aligned_malloc(sizeof(float)*M_pad * K_pad, 128);
      lw_zeropad_matrix(A, M, M_pad, M_align_size, K, K_pad, K_align_size, A_zeropad_mpe);
    }
    else
      A_zeropad = A;

    gettimeofday(&t2, NULL);
    pad_time = TIME(t1,t2);
    //printf("lw_zeropad_matrix time %lf sec\n", pad_time);

    gettimeofday(&t1, NULL);
    float* B_zeropad;
    if(N != N_pad || K != K_pad) {
      B_zeropad = (float*)_aligned_malloc(sizeof(float)*N_pad * K_pad, 128);
      sw_zeropad_matrix(B, N, N_pad, K, K_pad, B_zeropad);
    }
    else
      B_zeropad = B;
    gettimeofday(&t2, NULL);
    pad_time = TIME(t1,t2);
    //printf("sw_zeropad_matrix B time %lf sec\n", pad_time);

    gettimeofday(&t1, NULL);
    float* C_zeropad;
    if(N != N_pad || M != M_pad) {
      C_zeropad = (float*)_aligned_malloc(sizeof(float)*N_pad * M_pad, 128);
      //memset(C_zeropad, 0., sizeof(float)*N_pad*M_pad);
    }
    else
      C_zeropad = C;
    gettimeofday(&t2, NULL);
    pad_time = TIME(t1,t2);
    //printf("sw_zeropad_matrix C time %lf sec\n", pad_time);

    printf("M %d K %d N %d ; M_pad %d K_pad %d N_pad %d; blkN %d blkK %d blkM %d\n", 
        M, K, N,
        M_pad, K_pad, N_pad,
        N_align_size, K_align_size, M_align_size);

    params->input = B_zeropad;
    params->weight = A_zeropad;
    params->output = C_zeropad;
    params->K = K_pad;
    params->blkK = K_align_size;
    params->N = M_pad;
    params->blkN = M_align_size;
    params->M = N_pad;
    params->blkM = N_align_size;

    if(params->blkM%128 == 0 && params->blkN%32 == 0 && params->blkK%8 == 0){
      athread_spawn(FJR_blas_sgemm_trans_dma_full_pipeline, params);
      //athread_spawn(FJR_blas_sgemm_trans_test_perfmdl, params);
      athread_join();
    } else {
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, N);
    }

    if(M != M_pad || K != K_pad ) {
      _aligned_free(A_zeropad);
      _aligned_free(A_zeropad_mpe);
    }

    if(N != N_pad || K != K_pad)
      _aligned_free(B_zeropad);

    gettimeofday(&t1, NULL);
    if(M != M_pad || N != N_pad) {
      sw_depad_matrix(C, N, N_pad, M, M_pad, C_zeropad);
      _aligned_free(C_zeropad);
    }
    gettimeofday(&t2, NULL);
    pad_time = TIME(t1,t2);
    printf("sw_zeropad_matrix C time %lf sec\n", pad_time);

  }
  free(params);
  //lwpf_report_summary(stdout, &conf);
  //lwpf_report_summary_wide(stdout, &conf);
  //lwpf_report_detail(stdout, &conf);
  return;
}


