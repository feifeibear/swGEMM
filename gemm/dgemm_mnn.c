/*************************************************************************
	> File Name: gemm.c
	> Author: Jiarui Fang 
	> Mail: fang_jiarui@163.com 
	> Created Time: Mon 29 Oct 2018 01:04:42 PM CST
  data layout
               weight(K, N)
  input(K, M)  output(M, N)
 ************************************************************************/
#include "simd.h"
#include <stdio.h>

void ldm_dgemm_mnn(double* input, double* weight, double* output, int M, int Mld, int N, int K, int rid, int cid){
	int ccCore, cN, cM, cK;
	int i, j;

	doublev4 tmp_weight[4];
  doublev4 tmp_input[4];
	doublev4 tmp_output[16];

  for(ccCore=0; ccCore<8; ++ccCore){
    for(cM = 0; cM < M; cM += 4){
      //FJRmn: 
		  //double* output_ptr = output + 4*cM;
		  double* output_ptr = output + N*4*cM;
      for(cN = 0; cN < N; cN += 4){
			  //FJR trans: cN*K
			  double* weight_ptr = weight + 4*cN;
			  double* input_ptr  = input  + cM;
			  for(i = 0 ; i < 4; ++i)
			    for(j = 0; j < 4; ++j)
            //FJRmn:
			      simd_load( tmp_output[j*4+i],(output_ptr+ 4*(N*j+i)));
			      //simd_load( tmp_output[i*4+j],(output_ptr+4*(j+i*M)) );

        for(cK = 0; cK < K; ++cK){
          if(ccCore == cid){
            simd_loade(tmp_input[0], input_ptr);
            simd_putr(tmp_input[0],8);
            simd_loade(tmp_input[1], (input_ptr + 1));
            simd_putr(tmp_input[1],8);
            simd_loade(tmp_input[2], (input_ptr + 2));
            simd_putr(tmp_input[2],8);
            simd_loade(tmp_input[3], (input_ptr + 3));
            simd_putr(tmp_input[3],8);
          }
          else{
            tmp_input[0] = simd_getr(tmp_input[0]);
            tmp_input[1] = simd_getr(tmp_input[1]);
            tmp_input[2] = simd_getr(tmp_input[2]);
            tmp_input[3] = simd_getr(tmp_input[3]);
          }

					if(ccCore == rid){
            simd_load(tmp_weight[0], weight_ptr);
	       		simd_putc(tmp_weight[0], 8);
	       		simd_load(tmp_weight[1], weight_ptr + 4);
	       		simd_putc(tmp_weight[1], 8);
	       		simd_load(tmp_weight[2], weight_ptr + 8);
	       		simd_putc(tmp_weight[2], 8);
	       		simd_load(tmp_weight[3], weight_ptr + 12);
           	simd_putc(tmp_weight[3], 8);
          } else {
           	tmp_weight[0] = simd_getc(tmp_weight[0]);
           	tmp_weight[1] = simd_getc(tmp_weight[1]);
           	tmp_weight[2] = simd_getc(tmp_weight[2]);
           	tmp_weight[3] = simd_getc(tmp_weight[3]);
          }

          for(j = 0; j<4; ++j){
					  for(i = 0; i<4; ++i){
           	  tmp_output[j*4+i] += tmp_input[j]*tmp_weight[i];
           	}
          }

					//FJR trans
					weight_ptr += 4*N;
					input_ptr  += M;

			  }//cK
		    for(i = 0 ; i < 4; ++i)
			    for(j = 0; j < 4; ++j)
			      simd_store( tmp_output[j*4+i],(output_ptr+ 4*(N*j+i) ));
			      //simd_store(  tmp_output[i*4+j],(output_ptr+4*(j+i*M)) ); 
        //FJRmn
			  //output_ptr += 16*M;
			  output_ptr += 4*4;
		  }//cN
		}//cM
  }//ccCore
}
