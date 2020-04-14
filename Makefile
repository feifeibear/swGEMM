CC = sw5cc.new
LINK = mpif90 
#all: debug_asm 
all:swgemm_test

CFLAGS = -msimd -I.
#uncomment to time inside CPE with rpcc, warning! will get low overall performance

DEBUGFLAGS += -DUSE_COMP
DEBUGFLAGS += -DUSE_RTC
DEBUGFLAGS += -DUSE_DMA
DEBUGFLAGS += -DUSE_FLOAT2DOUBLE

DFLAGS += -D_MEM_128BALIGN_
DFLAGS += -DCHECK_RES

NOCFLAGS = -O0 -msimd
LIBFLAGS = -hybrid -lm_slave -allshare
BLASLIB = ./thirdparty/lib/cblas_LINUX.a
#BLASLIB += ../../swthirdparty/lib/libswblasall-2.a
BLASLIB += ./thirdparty/lib/libswblas_opt.a

gemmsrc=$(wildcard ./gemm/*.c)
slavesrc=$(wildcard ./slave/*.c)
debugslavesrc=$(wildcard ./debug_slave/*.c)
asmsrc=$(wildcard ./asm/*.S)
src=$(wildcard ./src/*.c)
debugsrc=$(wildcard ./debug_src/*.c)
LIBOBJ += ./util/common_slave.o
LIBOBJ += $(patsubst %.c, %.o, $(slavesrc) $(debugslavesrc) $(src) $(debugsrc))
LIBOBJ += $(patsubst %.c, %.o, $(gemmsrc))
LIBOBJ += $(patsubst %.S, %.o, $(asmsrc))
OBJ = ${LIBOBJ}

DGEMM_MINGZHEN_O = 	asm/dgemm_trans_asm.o asm/dgemm_trans_asm_m.o \
					slave/sw_copy_border.o slave/sw_copy_back_border.o \
					slave/sw_slave_gemm_trans_gen.o \
					src/sw_dgemm_std.o

mingzhen:${DGEMM_MINGZHEN_O}
	sw5ar cr ./libswdgemm.a ${DGEMM_MINGZHEN_O}					

debug_asm: ${OBJ} test_asm.o
	$(CC) $(LIBFLAGS) $^ ${BLASLIB} -o $@ 

swgemm_test: ${OBJ} test.o
	#$(CC) $(LIBFLAGS) $^ ${BLASLIB} -o $@ 
	$(LINK) $(LIBFLAGS) $^ ${BLASLIB} -o $@ ${LINK_SPC} 

ar: $(LIBOBJ)
	swar rcs ./libswgemm.a ${BLASLIB} $^

test_asm.o: test_asm.c
	$(CC) $(DFLAGS) -O2 -I./thirdparty/include/ -host -c $^ -o $@ 

test.o: test.c
	$(CC) $(DFLAGS) -O2 -I./thirdparty/include/ -host -c test.c -o test.o -OPT:IEEE_arith=1

./src/%.o: ./src/%.c
	$(CC) $(DFLAGS) $(CFLAGS) -O2 -I./thirdparty/include/ -host -c $^ -o $@ -OPT:IEEE_arith=1

./debug_src/%.o: ./debug_src/%.c
	$(CC) $(DFLAGS) $(CFLAGS) -O2 -I./thirdparty/include/ -host -c $^ -o $@

./gemm/%.o: ./gemm/%.c
	$(CC) $(DFLAGS) $(CFLAGS) -O2 -slave -c $< -o $@ 

haha:./gemm/sgemm_small.c
	$(CC) $(DFLAGS) $(CFLAGS) -O2 -slave -S $< -o ./gemm/sgemm_small.S

./slave/%.o: ./slave/%.c
	$(CC) $(DFLAGS) -O1 -msimd -I. -slave -c $^ -o $@
	#$(CC) $(DEBUGFLAGS) -O1 -msimd -I. -slave -c $^ -o $@

./debug_slave/%.o: ./debug_slave/%.c
	$(CC) $(DEBUGFLAGS) -O2 -msimd -I. -slave -c $^ -o $@

./asm/%.o: ./asm/%.S
	$(CC) -O0 -msimd -I. -slave -c $^ -o $@

./util/common_slave.o: ./util/common_slave.c
	$(CC) $(CFLAGS) -O2 -slave -c $^ -o $@

run: swgemm_test
	sh run.sh 1280 128 128
	#sh run.sh 2560 128 129
	#time bsub -b -I -p -q q_sw_expr -n 1 -o run.log -cgsp 64 ./example_asm
	#time bsub -b -I -m 1 -p -q q_sw_share -host_stack 1024 -sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 -n 1 -o run.log -cgsp 64 ./example_asm

.PHONY: clean
clean:
	rm -rf example_asm *.o ./debug_slave/*.o ./debug_src/*.o ./slave/*.o ./gemm/*.o ./asm/*.o ./swblas.a


