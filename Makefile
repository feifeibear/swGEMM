CC = sw5cc.new
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
BLASLIB = ./thirdparty/lib/cblas_LINUX0324.a
#BLASLIB += ../../swthirdparty/lib/libswblasall-2.a
BLASLIB += ./thirdparty/lib/libswblas0324.a

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

debug_asm: ${OBJ} test_asm.o
	$(CC) $(LIBFLAGS) $^ ${BLASLIB} -o $@ 

swgemm_test: ${OBJ} test.o
	$(CC) $(LIBFLAGS) $^ ${BLASLIB} -o $@ 

ar: $(LIBOBJ)
	swar rcs ./libswgemm.a ${BLASLIB} $^

test_asm.o: test_asm.c
	$(CC) $(DFLAGS) -O2 -I./thirdparty/include/ -host -c $^ -o $@ 

test.o: test.c
	$(CC) $(DFLAGS) -O2 -I./thirdparty/include/ -host -c test.c -o test.o

./src/%.o: ./src/%.c
	$(CC) $(DFLAGS) $(CFLAGS) -O2 -I./thirdparty/include/ -host -c $^ -o $@

./debug_src/%.o: ./debug_src/%.c
	$(CC) $(DFLAGS) $(CFLAGS) -O2 -I./thirdparty/include/ -host -c $^ -o $@

./gemm/%.o: ./gemm/%.c
	$(CC) $(DFLAGS) $(CFLAGS) -O2 -slave -c $< -o $@ 

haha:./gemm/sgemm_small.c
	$(CC) $(DFLAGS) $(CFLAGS) -O2 -slave -S $< -o ./gemm/sgemm_small.S

./slave/%.o: ./slave/%.c
	$(CC) $(DFLAGS) -O1 -msimd -I. -slave -c $^ -o $@

./debug_slave/%.o: ./debug_slave/%.c
	$(CC) $(DEBUGFLAGS) -O2 -msimd -I. -slave -c $^ -o $@

./asm/%.o: ./asm/%.S
	$(CC) -O0 -msimd -I. -slave -c $^ -o $@

./util/common_slave.o: ./util/common_slave.c
	$(CC) $(CFLAGS) -O2 -slave -c $^ -o $@

run: example_asm
	sh run.sh 2560 128 128
	#time bsub -b -I -p -q q_sw_expr -n 1 -o run.log -cgsp 64 ./example_asm
	#time bsub -b -I -m 1 -p -q q_sw_share -host_stack 1024 -sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 -n 1 -o run.log -cgsp 64 ./example_asm

.PHONY: clean
clean:
	rm -rf example_asm *.o ./debug_slave/*.o ./debug_src/*.o ./slave/*.o ./gemm/*.o ./asm/*.o ./swblas.a


