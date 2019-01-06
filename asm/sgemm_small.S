	.set noreorder
	.set noat
	.arch sw3
	#  /usr/sw-mpp/swcc/lib/gcc-lib/sw_64-swcc-linux/5.421-sw-500/be::5.421-sw-500

	#-----------------------------------------------------------
	# Compiling gemm/sgemm_small.c (/tmp/ccI#.YUgMXX)
	#-----------------------------------------------------------

	#-----------------------------------------------------------
	# Options:
	#-----------------------------------------------------------
	#  Target:SW3, ISA:ISA_1, Endian:little, Pointer Size:64
	#  -O2	(Optimization level)
	#  -g0	(Debug level)
	#  -m2	(Report advisories)
	#-----------------------------------------------------------

	.file	1	"gemm/sgemm_small.c"
	.file	2	"/usr/sw-mpp/swcc/lib/gcc-lib/sw_64-swcc-linux/5.421-sw-500/include/simd.h"


	.section .text1, "ax", "progbits"
	.align	4

	.section .bss, "wa", "nobits"
	.align	5

	.section .bss
	.org 0x0
	.align	0
	.type	anon25_107, @object
	.size	anon25_107, 16
anon25_107:	# 0x0
	.skip 16
	.org 0x10
	.align	0
	.type	anon26_108, @object
	.size	anon26_108, 16
anon26_108:	# 0x10
	.skip 16
	.org 0x20
	.align	0
	.type	anon27_109, @object
	.size	anon27_109, 16
anon27_109:	# 0x20
	.skip 16
	.org 0x30
	.align	0
	.type	anon28_110, @object
	.size	anon28_110, 16
anon28_110:	# 0x30
	.skip 16
	.section .text1

	# Program Unit: ldm_sgempasm
	.align 4
	.ent	ldm_sgempasm#
	.globl	ldm_sgempasm
ldm_sgempasm:	# 0x0
	# anon14 = 296
	# anon15 = 300
	# anon16 = 292
	# anon17 = 288
	# anon21 = 32
	# return_address = 0
	# _temp_gra_spill0 = 304
	# _temp_gra_spill1 = 312
	# _temp_gra_spill2 = 320
	# _temp_gra_spill3 = 352
	# _temp_gra_spill4 = 384
	# _temp_gra_spill5 = 392
	# _temp_gra_spill6 = 400
	# _temp_gra_spill7 = 408
	# _temp_gra_spill8 = 416
	# _temp_gra_spill9 = 424
	# _temp_gra_spill10 = 432
	# _temp_gra_spill11 = 440
	# _temp_gra_spill12 = 448
	# _temp_gra_spill13 = 456
	# _temp_gra_spill14 = 464
	# _temp_gra_spill15 = 472
	# _temp_gra_spill16 = 480
	# _temp_gra_spill17 = 488
	# _temp_gra_spill18 = 496
	# _temp_gra_spill19 = 504
	# _temp_gra_spill20 = 512
	# _temp_gra_spill21 = 520
	# _temp_gra_spill22 = 528
	# _temp_gra_spill23 = 536
	# _temp_gra_spill24 = 544
	# _temp_gra_spill25 = 552
	# _temp_gra_spill26 = 560
	# _temp_gra_spill27 = 568
	# _temp_gra_spill28 = 576
	# _temp_gra_spill29 = 584
	# _temp_gra_spill30 = 592
	# _temp_gra_spill31 = 600
	# _temp_gra_spill32 = 608
	# _temp_gra_spill33 = 616
	.loc	1	31	0
#  27  //cM - cCo
#  28  //cN - cNo
#  29  //cK - cNi
#  30  //M*K K*N
#  31  void ldm_sgempasm(float* input, float* weight, float* output, int M, int Mld, int N, int K) {
.BB1_ldm_sgempasm:
	.prologue
	ldih	$gp,0($27)               	!gpdisp!1	# [0] 0
	ldi	$gp,0($gp)                	!gpdisp!1	# [1] 0
	
$ng..ldm_sgempasm:
	ldi	$sp,-704($sp)             	# [3] 
.LCFI_ldm_sgempasm_adjustsp:
	mov	$19,$23                   	# [3] 
	stl	$17,472($sp)              	# [4] _temp_gra_spill15
	.frame $15,704,$26,0
	.mask 0x4008000,-704
	stl	$21,552($sp)              	# [4] _temp_gra_spill25
	stl	$26,408($sp)              	# [5] _temp_gra_spill7
.LCFI_ldm_sgempasm_store26:
	stl	$15,416($sp)              	# [5] _temp_gra_spill8
	stl	$14,424($sp)              	# [6] _temp_gra_spill9
#	.savesp $26, 0
	stl	$13,432($sp)              	# [6] _temp_gra_spill10
#	.spillsp $15, 0
	stl	$12,440($sp)              	# [7] _temp_gra_spill11
#	.spillsp $14, 0
	stl	$11,448($sp)              	# [7] _temp_gra_spill12
#	.spillsp $13, 0
	stl	$10,392($sp)              	# [8] _temp_gra_spill5
#	.spillsp $12, 0
	stl	$9,400($sp)               	# [8] _temp_gra_spill6
#	.spillsp $11, 0
	ble	$19,.Lt_0_9218            	# [8] 
.BB2_ldm_sgempasm:
#	.spillsp $10, 0
	mov	$19,$17                   	# [0] 
#	.spillsp $9, 0
	addw	$19,3,$4                 	# [0] 
	addw	$19,6,$28                	# [0] 
	cmplt	$31,$21,$27             	# [0] 
	stl	$17,536($sp)              	# [1] _temp_gra_spill23
	sll	$19,4,$17                 	# [1] 
	ldw	$19,704($sp)              	# [1] K
	selge	$4,$4,$28,$4            	# [1] 
	sra	$4,2,$4                   	# [2] 
	addw	$21,3,$25                	# [2] 
	mov	$20,$24                   	# [2] 
	mov	$18,$22                   	# [2] 
	mov	$16,$21                   	# [3] 
	stl	$27,480($sp)              	# [3] _temp_gra_spill16
	mov	$19,$2                    	# [4] 
	s4addl	$19,$31,$0             	# [4] 
	cmplt	$31,$19,$19             	# [4] 
	s4subl	$2,$2,$1               	# [5] 
	sextw	$0,$0                   	# [5] 
	s8addl	$2,$31,$3              	# [5] 
	s4addl	$2,$31,$2              	# [5] 
	s4addl	$1,$31,$1              	# [6] 
	s4addl	$0,$31,$0              	# [6] 
	stw	$31,300($sp)              	# [7] anon15
	stl	$4,384($sp)               	# [7] _temp_gra_spill4
	stl	$25,464($sp)              	# [8] _temp_gra_spill14
	stl	$24,576($sp)              	# [8] _temp_gra_spill28
	stl	$22,488($sp)              	# [9] _temp_gra_spill17
	stl	$21,560($sp)              	# [9] _temp_gra_spill26
	stl	$19,528($sp)              	# [10] _temp_gra_spill22
	stl	$3,512($sp)               	# [10] _temp_gra_spill20
	stl	$2,504($sp)               	# [11] _temp_gra_spill19
	stl	$1,520($sp)               	# [11] _temp_gra_spill21
	stl	$0,544($sp)               	# [12] _temp_gra_spill24
	.align	4
.Lt_0_9730:
#<loop> Loop body line 31, nesting depth: 1, estimated iterations: 100
	.loc	1	40	0
#  36    floatv4 tmp_input[4];
#  37  	floatv4 tmp_output[16];
#  38  
#  39    for(cM = 0; cM < M; cM += 4){
#  40  	  float* output_ptr = output + 4*cM;
	ldl	$1,488($sp)               	# [0] _temp_gra_spill17
	ldl	$0,480($sp)               	# [0] _temp_gra_spill16
	stl	$1,304($sp)               	# [3] _temp_gra_spill0
	beq	$0,.Lt_0_9986             	# [3] 
.BB5_ldm_sgempasm:
#<loop> Part of loop body line 65536, head labeled .Lt_0_9730
	ldl	$0,464($sp)               	# [0] _temp_gra_spill14
	ldl	$2,472($sp)               	# [0] _temp_gra_spill15
	stw	$31,296($sp)              	# [1] anon14
	addw	$0,3,$1                  	# [3] 
	stl	$2,568($sp)               	# [3] _temp_gra_spill27
	selge	$0,$0,$1,$0             	# [4] 
	sra	$0,2,$0                   	# [5] 
	stl	$0,456($sp)               	# [6] _temp_gra_spill13
	.align	4
.Lt_0_10498:
#<loop> Loop body line 40, nesting depth: 2, estimated iterations: 100
	.loc	1	46	0
#  42  	    for(i = 0 ; i < 4; ++i)
#  43  	      for(j = 0; j < 4; ++j)
#  44  	        simd_load( tmp_output[i*4+j],(output_ptr+4*(j+i*M)) );
#  45  	    //FJR trans
#  46  	    float* weight_ptr = weight + cN*K;
	ldl	$0,568($sp)               	# [0] _temp_gra_spill27
	.loc	1	40	0
	ldl	$6,304($sp)               	# [0] _temp_gra_spill0
	mov	$31,$8                    	# [0] 
	mov	$31,$22                   	# [0] 
	.loc	1	47	0
#  47  	    float* input_ptr  = input  + 4*cM;
	ldl	$5,560($sp)               	# [1] _temp_gra_spill26
	.loc	1	40	0
	mov	4,$7                      	# [1] 
	ldi	$16,32($sp)               	# [1] anon21
	.loc	1	46	0
	mov	$0,$19                    	# [3] 
.BB25_ldm_sgempasm:
.BB23_ldm_sgempasm:
.Lt_0_11266:
#<loop> Loop body line 40, nesting depth: 4, iterations: 1
#<loop> unrolled 4 times (fully)
	.loc	1	44	0
	vlds	$4,0($6)                 	# [0] id:194
	vlds	$3,16($6)                	# [0] id:194
	vlds	$2,32($6)                	# [1] id:194
	vlds	$1,48($6)                	# [1] id:194
	vsts	$4,0($16)                	# [4] id:195 anon21+0x0
	vsts	$3,16($16)               	# [4] id:195 anon21+0x0
	vsts	$2,32($16)               	# [5] id:195 anon21+0x0
	vsts	$1,48($16)               	# [5] id:195 anon21+0x0
.BB10_ldm_sgempasm:
.BB24_ldm_sgempasm:
	.loc	1	42	0
	addw	$22,1,$22                	# [0] 
	ldi	$16,64($16)               	# [0] 
	addw	$8,$23,$8                	# [0] 
	addl	$17,$6,$6                	# [0] 
	cmpeq	$22,4,$1                	# [1] 
	addw	$7,$23,$7                	# [1] 
	beq	$1,.Lt_0_11266            	# [2] 
.BB11_ldm_sgempasm:
#<loop> Part of loop body line 65536, head labeled .Lt_0_10498
	.loc	1	47	0
	ldl	$1,528($sp)               	# [0] _temp_gra_spill22
	.loc	1	48	0
#  48        for(cK=0; cK < K; ++cK){
	ldl	$6,304($sp)               	# [0] _temp_gra_spill0
	mov	4,$7                      	# [0] 
	ldi	$16,32($sp)               	# [0] anon21
	.loc	1	47	0
	beq	$1,.BB31_ldm_sgempasm  	# [3] 
.BB12_ldm_sgempasm:
#<loop> Part of loop body line 65536, head labeled .Lt_0_10498
	ldw	$2,704($sp)               	# [0] K
	stl	$7,600($sp)               	# [0] _temp_gra_spill31
	mov	$31,$18                   	# [0] 
	stl	$6,608($sp)               	# [1] _temp_gra_spill32
	ldl	$1,568($sp)               	# [1] _temp_gra_spill27
	ldl	$8,512($sp)               	# [2] _temp_gra_spill20
	ldl	$7,520($sp)               	# [2] _temp_gra_spill21
	stl	$2,496($sp)               	# [3] _temp_gra_spill18
	.loc	1	91	0
#  87  	    }//cK
#  88  	    for(i = 0 ; i < 4; ++i)
#  89  	      for(j = 0; j < 4; ++j)
#  90  	        simd_store(  tmp_output[i*4+j],(output_ptr+4*(j+i*M)) ); 
#  91  	    output_ptr += 16*M;
	ldl	$2,536($sp)               	# [3] _temp_gra_spill23
	.loc	1	47	0
	ldl	$6,504($sp)               	# [4] _temp_gra_spill19
	stl	$17,584($sp)              	# [4] _temp_gra_spill29
	stl	$23,592($sp)              	# [5] _temp_gra_spill30
	vlds	$12,272($sp)             	# [5] anon21+240
	addl	$8,$1,$8                 	# [5] 
	addl	$7,$1,$7                 	# [5] 
	.loc	1	91	0
	sll	$2,6,$2                   	# [6] 
	.loc	1	47	0
	vlds	$21,208($sp)             	# [6] anon21+176
	vlds	$11,144($sp)             	# [6] anon21+112
	.loc	1	91	0
	stl	$2,312($sp)               	# [7] _temp_gra_spill1
	.loc	1	41	0
	ldl	$2,544($sp)               	# [7] _temp_gra_spill24
	.loc	1	47	0
	addl	$6,$1,$6                 	# [7] 
	vlds	$28,80($sp)              	# [8] anon21+48
	vlds	$26,256($sp)             	# [8] anon21+224
	vlds	$27,192($sp)             	# [9] anon21+160
	vlds	$25,64($sp)              	# [9] anon21+32
	vlds	$22,176($sp)             	# [10] anon21+144
	vlds	$15,112($sp)             	# [10] anon21+80
	.loc	1	41	0
	addl	$2,$19,$1                	# [10] 
	.loc	1	47	0
	vlds	$9,48($sp)               	# [11] anon21+16
	vlds	$10,224($sp)             	# [11] anon21+192
	vlds	$24,160($sp)             	# [12] anon21+128
	vlds	$14,96($sp)              	# [12] anon21+64
	vlds	$13,32($sp)              	# [13] anon21
	vlds	$17,240($sp)             	# [13] anon21+208
	vlds	$23,128($sp)             	# [14] anon21+96
	stl	$16,616($sp)              	# [14] _temp_gra_spill33
	.loc	1	41	0
	stl	$1,568($sp)               	# [15] _temp_gra_spill27
	.align	4
.Lt_0_13314:
.BB26_ldm_sgempasm:
#<loop> Loop body line 47, nesting depth: 3, estimated iterations: 100
	ldse	$1,0($6)                 	# [0] id:201
	vlds	$16,48($5)               	# [0] id:199
	addw	$18,1,$18                	# [0] 
	ldi	$6,4($6)                  	# [0] 
	vlds	$4,32($5)                	# [1] id:198
	vlds	$3,16($5)                	# [1] id:197
	vlds	$2,0($5)                 	# [2] id:196
	ldse	$19,0($0)                	# [2] id:200
	ldi	$0,4($0)                  	# [2] 
	ldse	$20,0($7)                	# [3] id:203
	ldi	$7,4($7)                  	# [3] 
	vsts	$1,352($sp)              	# [4] _temp_gra_spill3
	vmas	$1,$16,$11,$11           	# [4] 
	vmas	$1,$4,$23,$23            	# [5] 
	vmas	$1,$3,$15,$15            	# [5] 
	vmas	$2,$1,$14,$14            	# [6] 
	ldse	$1,0($8)                 	# [6] id:202
	vmas	$19,$16,$28,$28          	# [6] 
	ldi	$8,4($8)                  	# [6] 
	vmas	$20,$16,$12,$12          	# [7] 
	vmas	$19,$2,$13,$13           	# [7] 
	vmas	$19,$3,$9,$9             	# [8] 
	vmas	$19,$4,$25,$25           	# [8] 
	vmas	$2,$20,$10,$10           	# [9] 
	vmas	$3,$20,$17,$17           	# [9] 
	vsts	$1,320($sp)              	# [10] _temp_gra_spill2
	vmas	$1,$16,$21,$21           	# [10] 
	vmas	$2,$1,$24,$24            	# [10] 
	ldl	$16,576($sp)              	# [10] _temp_gra_spill28
	vmas	$3,$1,$22,$22            	# [11] 
	vmas	$1,$4,$27,$27            	# [11] 
	ldw	$1,704($sp)               	# [11] K
	vmas	$4,$20,$26,$26           	# [12] 
	sll	$16,4,$16                 	# [13] 
	cmpeq	$18,$1,$1               	# [14] 
	addl	$5,$16,$5                	# [14] 
	vsts	$22,176($sp)             	# [15] anon21+144
	vsts	$27,192($sp)             	# [15] anon21+160
	vsts	$10,224($sp)             	# [16] anon21+192
	vsts	$17,240($sp)             	# [16] anon21+208
	vsts	$12,272($sp)             	# [17] anon21+240
	vsts	$24,160($sp)             	# [17] anon21+128
	vsts	$21,208($sp)             	# [18] anon21+176
	vsts	$14,96($sp)              	# [18] anon21+64
	vsts	$15,112($sp)             	# [19] anon21+80
	vsts	$23,128($sp)             	# [19] anon21+96
	vsts	$11,144($sp)             	# [20] anon21+112
	vsts	$13,32($sp)              	# [20] anon21
	vsts	$9,48($sp)               	# [21] anon21+16
	vsts	$25,64($sp)              	# [21] anon21+32
	vsts	$28,80($sp)              	# [22] anon21+48
	vsts	$26,256($sp)             	# [22] anon21+224
	beq	$1,.Lt_0_13314            	# [22] 
#<loop> Part of loop body line 65536, head labeled .Lt_0_10498
	vlds	$0,352($sp)              	# [0] _temp_gra_spill3
	vlds	$5,320($sp)              	# [0] _temp_gra_spill2
	ldl	$16,616($sp)              	# [1] _temp_gra_spill33
	ldl	$6,608($sp)               	# [1] _temp_gra_spill32
	ldl	$7,600($sp)               	# [2] _temp_gra_spill31
	ldl	$23,592($sp)              	# [2] _temp_gra_spill30
	ldl	$17,584($sp)              	# [3] _temp_gra_spill29
	ldl	$18,304($sp)              	# [3] _temp_gra_spill0
	ldl	$8,312($sp)               	# [4] _temp_gra_spill1
	ldw	$21,296($sp)              	# [4] anon14
.BB15_ldm_sgempasm:
#<loop> Part of loop body line 65536, head labeled .Lt_0_10498
	ldl	$25,.bss($gp)             	!literal	# [0] .bss
	.loc	1	41	0
	ldl	$24,552($sp)              	# [0] _temp_gra_spill25
	addw	$21,4,$21                	# [0] 
	.loc	1	91	0
	addl	$18,$8,$18               	# [0] 
	.loc	1	48	0
	vsts	$19,0($25)               	# [3] id:207 anon25+0x0
	vsts	$0,16($25)               	# [3] id:206 anon26+0x0
	.loc	1	41	0
	cmple	$24,$21,$24             	# [3] 
	.loc	1	48	0
	vsts	$5,32($25)               	# [4] id:205 anon27+0x0
	vsts	$20,48($25)              	# [4] id:204 anon28+0x0
.Lt_0_12802:
#<loop> Part of loop body line 65536, head labeled .Lt_0_10498
	mov	$31,$8                    	# [0] 
	mov	$31,$22                   	# [0] 
.BB30_ldm_sgempasm:
.BB28_ldm_sgempasm:
.Lt_0_14338:
#<loop> Loop body line 48, nesting depth: 4, iterations: 1
#<loop> unrolled 4 times (fully)
	.loc	1	90	0
	vlds	$3,0($16)                	# [0] id:208 anon21+0x0
	vlds	$2,16($16)               	# [0] id:208 anon21+0x0
	vlds	$1,32($16)               	# [1] id:208 anon21+0x0
	vlds	$0,48($16)               	# [1] id:208 anon21+0x0
	vsts	$3,0($6)                 	# [4] id:209
	vsts	$2,16($6)                	# [4] id:209
	vsts	$1,32($6)                	# [5] id:209
	vsts	$0,48($6)                	# [5] id:209
.BB19_ldm_sgempasm:
.BB29_ldm_sgempasm:
	.loc	1	88	0
	addw	$22,1,$22                	# [0] 
	ldi	$16,64($16)               	# [0] 
	addw	$8,$23,$8                	# [0] 
	addl	$17,$6,$6                	# [0] 
	cmpeq	$22,4,$0                	# [1] 
	addw	$7,$23,$7                	# [1] 
	beq	$0,.Lt_0_14338            	# [2] 
.BB20_ldm_sgempasm:
#<loop> Part of loop body line 65536, head labeled .Lt_0_10498
	stw	$21,296($sp)              	# [0] anon14
	stl	$18,304($sp)              	# [0] _temp_gra_spill0
	.loc	1	41	0
	beq	$24,.Lt_0_10498           	# [0] 
.Lt_0_9986:
#<loop> Part of loop body line 65536, head labeled .Lt_0_9730
	.loc	1	39	0
	ldw	$0,300($sp)               	# [0] anon15
	ldl	$2,560($sp)               	# [0] _temp_gra_spill26
	ldl	$1,488($sp)               	# [1] _temp_gra_spill17
	addw	$0,4,$0                  	# [3] 
	ldi	$2,64($2)                 	# [3] 
	stw	$0,300($sp)               	# [4] anon15
	ldi	$1,64($1)                 	# [4] 
	cmple	$23,$0,$0               	# [4] 
	stl	$2,560($sp)               	# [5] _temp_gra_spill26
	stl	$1,488($sp)               	# [5] _temp_gra_spill17
	beq	$0,.Lt_0_9730             	# [5] 
.Lt_0_9218:
	ldl	$26,408($sp)              	# [0] _temp_gra_spill7
	ldl	$9,400($sp)               	# [0] _temp_gra_spill6
	ldl	$10,392($sp)              	# [1] _temp_gra_spill5
#	.body
#	.label_state 1
#	.body
	ldl	$11,448($sp)              	# [1] _temp_gra_spill12
	ldl	$12,440($sp)              	# [2] _temp_gra_spill11
	ldl	$13,432($sp)              	# [2] _temp_gra_spill10
	ldl	$14,424($sp)              	# [3] _temp_gra_spill9
	ldl	$15,416($sp)              	# [3] _temp_gra_spill8
	ldi	$sp,704($sp)              	# [3] 
	ret	$31,($26),1               	# [4] 
	.align	4
.BB31_ldm_sgempasm:
#<loop> Part of loop body line 65536, head labeled .Lt_0_10498
	.loc	1	41	0
	ldw	$21,296($sp)              	# [0] anon14
	.loc	1	91	0
#	.restore $sp
	ldl	$0,536($sp)               	# [0] _temp_gra_spill23
	.loc	1	41	0
	ldl	$25,544($sp)              	# [1] _temp_gra_spill24
#	.body
#	.copy_state 1
	ldl	$24,552($sp)              	# [1] _temp_gra_spill25
	.loc	1	91	0
	sll	$0,6,$0                   	# [3] 
	.loc	1	41	0
	addw	$21,4,$21                	# [3] 
	addl	$25,$19,$25              	# [4] 
	.loc	1	91	0
	addl	$6,$0,$18                	# [4] 
	.loc	1	41	0
	cmple	$24,$21,$24             	# [4] 
	stl	$25,568($sp)              	# [5] _temp_gra_spill27
	.loc	1	39	0
	br	$31,.Lt_0_12802            	# [5] 
.L_CC_ldm_sgempasm:
# PU cycle count: 0.000000
	.end	ldm_sgempasm
	.section .text1
	.align	4
	.section .bss
	.align	5
#	.gpvalue 0

	.section .debug_info, "", "progbits"
	.align	0
	.byte	0xa4, 0x00, 0x00, 0x00, 0x02, 0x00
	.long	.debug_abbrev
	.long	0x67730108, 0x5f6d6d65, 0x6c616d73, 0x00632e6c
	.long	0x6e65706f, 0x35204343, 0x3132342e, 0x2d77732d
	.byte	0x35, 0x30, 0x30, 0x00, 0x01, 0x00
	.long	.debug_line
	.long	0x6c1f0102, 0x735f6d64, 0x6d6d6567, 0x616d735f
	.byte	0x6c, 0x6c, 0x00, 0x01, 0x04, 0x92, 0x1e, 0xc0
	.byte	0x05
	.quad	.BB1_ldm_sgempasm
	.quad	.L_CC_ldm_sgempasm
	.long	0x691f0103, 0x7475706e, 0x40910200, 0x771f0103
	.long	0x68676965, 0x91020074, 0x1f010348, 0x7074756f
	.long	0x02007475, 0x01035091, 0x02004d1f, 0x01035891
	.long	0x646c4d1f, 0x60910200, 0x4e1f0103, 0x68910200
	.byte	0x03, 0x01, 0x1f, 0x4b, 0x00, 0x02, 0x91, 0x00
	.byte	0x00, 0x00, 0x00

	.section .debug_frame, "", "progbits"
	.align	0

	.section .debug_aranges, "", "progbits"
	.align	0
	.byte	0x2c, 0x00, 0x00, 0x00, 0x02, 0x00
	.long	.debug_info
	.byte	0x08, 0x00, 0x00, 0x00, 0x00, 0x00
	.quad	.BB1_ldm_sgempasm
	.quad	.L_CC_ldm_sgempasm - .BB1_ldm_sgempasm
	.long	0x00000000, 0x00000000, 0x00000000, 0x00000000

	.section .debug_pubnames, "", "progbits"
	.align	0
	.byte	0x22, 0x00, 0x00, 0x00, 0x02, 0x00
	.long	.debug_info
	.long	0x000000a8, 0x00000034, 0x5f6d646c, 0x6d656773
	.byte	0x6d, 0x5f, 0x73, 0x6d, 0x61, 0x6c, 0x6c, 0x00
	.byte	0x00, 0x00, 0x00, 0x00

	.section .eh_frame, "a", "progbits"
	.align	0

.LEHCIE:
	.long	.LEHCIE_end - .LEHCIE_begin
.LEHCIE_begin:
	.long 0x0
	.byte	0x01, 0x00, 0x01, 0x78, 0x1a, 0x0c, 0x1e, 0x00
	.align 3
.LEHCIE_end:

	.section .debug_abbrev, "", "progbits"
	.align	0
	.long	0x03011101, 0x13082508, 0x100b420b, 0x02000006
	.long	0x0b3a012e, 0x08030b3b, 0x0a400c3f, 0x01120111
	.long	0x05030000, 0x3b0b3a00, 0x0208030b, 0x0000000a
	.byte	0x00
	.section	.note.GNU-stack,"",@progbits
	.ident	"#SWCC Version 5.421-sw-500 : gemm/sgemm_small.c compiled with : -O2 -msimd "

