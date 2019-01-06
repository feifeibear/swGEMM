bsub -b -I -q q_sw_expr -host_stack 1024 -n 1 -cgsp 64 -sw3run /home/export/online1/swyf/swdnn/git/sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./swblas 512 512 512 
#bsub -b -I -q q_sw_expr -host_stack 1024 -n 1 -cgsp 64 -sw3run ../sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./swblas 1024 1024 128
#bsub -b -I -q q_sw_expr -host_stack 1024 -n 1 -cgsp 64 -sw3run ../sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./swblas 1022 1023 122
