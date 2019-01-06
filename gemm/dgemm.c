/*************************************************************************
	> File Name: gemm.c
	> Author: Jiarui Fang 
	> Mail: fang_jiarui@163.com 
	> Created Time: Thu 22 Sep 2016 01:04:42 PM CST
Parameters:
	This program implments a DGEMM routine, i.e. C+=A*B
	The A is of dimension 4Mld*K (However, only M*K is valid in GEMM)
	Mld is leading dim

	The B is of dim N*K
	K is leading Dim

	The C is of dim 4M*N
	M is leading dim
	rid and cid are the IDs of CPE in the 8*8 CPE mesh. 

Design:
	In this version, we use 4*4 register blocking plan.
 ************************************************************************/
#include "simd.h" 
#include <stdio.h>
#define Type double
#define SIMDType doublev4
//cM - cCo
//cN - cNo
//cK - cNi
//
//M*K K*N
void dgemm(Type* input, Type* weight, Type* output, int M, int Mld, int N, int K, int rid, int cid){
	int ccCore, cN, cM, cK;
	int i, j;

	SIMDType tmp_weight[4];
  SIMDType tmp_input[4];
	SIMDType tmp_output[16];

  for(ccCore=0; ccCore<8; ++ccCore){
      for(cM = 0; cM < M; cM += 4){
		      Type* output_ptr = output + 4*cM;
          for(cN = 0; cN < N; cN += 4){
			      for(i = 0 ; i < 4; ++i)
			        for(j = 0; j < 4; ++j)
			          simd_load( tmp_output[i*4+j],(output_ptr+4*(j+i*M)) );
			        //FJR trans
			        Type* weight_ptr = weight + cN*K;
			        Type* input_ptr  = input  + 4*cM;
              for(cK=0; cK < K; ++cK){
                   if(ccCore == cid){
                      simd_load(tmp_input[0], input_ptr);
                      simd_putr(tmp_input[0],8);
                      simd_load(tmp_input[1], (input_ptr + 4));
                      simd_putr(tmp_input[1],8);
                      simd_load(tmp_input[2],(input_ptr + 8));
                      simd_putr(tmp_input[2],8);
                      simd_load(tmp_input[3],(input_ptr + 12));
                      simd_putr(tmp_input[3],8);
                    }
                    else{
                      tmp_input[0] = simd_getr(tmp_input[0]);
                      tmp_input[1] = simd_getr(tmp_input[1]);
                      tmp_input[2] = simd_getr(tmp_input[2]);
                      tmp_input[3] = simd_getr(tmp_input[3]);
                    }

					    //FJR trans
					    if(ccCore == rid){
           			  simd_loade(tmp_weight[0], weight_ptr);
	       			    simd_putc(tmp_weight[0], 8);
	       			    simd_loade(tmp_weight[1], weight_ptr + K);
	       			    simd_putc(tmp_weight[1], 8);
	       			    simd_loade(tmp_weight[2], weight_ptr + 2*K);
	       			    simd_putc(tmp_weight[2], 8);
	       			    simd_loade(tmp_weight[3], weight_ptr + 3*K);
           			    simd_putc(tmp_weight[3], 8);
           			 }else{
           			    tmp_weight[0] = simd_getc(tmp_weight[0]);
           			    tmp_weight[1] = simd_getc(tmp_weight[1]);
           			    tmp_weight[2] = simd_getc(tmp_weight[2]);
           			    tmp_weight[3] = simd_getc(tmp_weight[3]);
           			 }

					    for(i = 0; i<4; ++i){
           	   		   for(j = 0; j<4; ++j){
           	   		     tmp_output[i*4+j] += tmp_input[j]*tmp_weight[i];
           	   		   }
           	   		 }

					 //FJR trans
					 weight_ptr += 1;
					 input_ptr  += 4*Mld;

			  }//cK
		    for(i = 0 ; i < 4; ++i)
			   for(j = 0; j < 4; ++j)
			    simd_store(  tmp_output[i*4+j],(output_ptr+4*(j+i*M)) ); 

			 output_ptr += 16*M;
		  }//cN
    }//cM
  }//ccCore
  return;
}
