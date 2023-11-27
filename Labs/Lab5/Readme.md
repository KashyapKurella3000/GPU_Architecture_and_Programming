In this lab, we will experiment with the effect of cache size, memory bandwidth, and scheduler on GPU benchmarks.

 

Step 1: Compilation

We will use 6 benchmarks in GPGPUSIM: 
1) JPEG, 2) BFS, 3) SLA, 4) SCP, 5) TRA, 6) LPS

Their code is here: benchmarks.zip Download benchmarks.zip

 

If you have successfully set up GPGPUSIM in Lab 4, you should be able to make them directly using the included makefile in each folder. If you are not able to compile them and cannot figure out why, please let me know. After compilation, you should be able to run them in the simulator with the corresponding config file provided, which is similar to Lab 4. You can run them with the following arguments:

JPEG:
./gpgpu_ptx_sim__JPEG --decode --file=./cameraman.bmp > out.txt


BFS:
./gpgpu_ptx_sim__BFS ./graph65536.txt > out.txt


SLA:
./gpgpu_ptx_sim__SLA --n=300000 > out.txt


SCP:
./gpgpu_ptx_sim__SCP --vector_n=4096 --element_n=4096 > out.txt


TRA:
./gpgpu_ptx_sim__TRA --size_x=1024 --size_y=1024 > out.txt

 

LPS:
./gpgpu_ptx_sim__LPS --nx=256 --ny=256 --nz=256 > out.txt

 

 

Step 2: Sensitivity study

Perform a sensitivity test on 1)/2)/3)/4) with the benchmark set and report your observations/conclusions (better with graphs). Also, provide reasoning of different behaviors. You have to report statistics related to performance (IPC), cache miss rates, and bandwidth utilization. Make sure all results are normalized to the results of the default configuration (i.e., GTX480). 

1) L1 data cache size: 16KB (default), 1/2 of default, 2x of default, 4x of default, 8x of default

2) L2 data cache size: 768KB (default), 1/2 of default, 2x of default, 4x of default, 8x of default

3) Memory bandwidth: 177.6GB/sec (default), 1/4 of default, 1/2 of default, 2x of default, 4x of default

4) Warp schedulers: GTO (default), LRR, Two Level

 

Note that the baseline is GTX 480 configuration. You can use the result of the last kernel if there are multiple kernels.

Also, you can turn off the power simulation so that you do not need to get unnecessary power log files:
-power_simulation_enabled 1 --> -power_simulation_enabled 0

If you want to change the cache size, you can change the number of ways. If you want to change the memory bandwidth, you can change the dram bus width.

 

 

Step 3: SWL Scheduler

You can enable the static warp limiting schedule in the gpgpusim.config file by using the following line (and comment out other schedulers):

-gpgpu_scheduler warp_limiting:2:x

The values of x can limit the number of warp that can run concurrently.

 

1) Discuss the effects on the IPC, L1, L2 miss-rates, and BW-utilization by changing the number of warps that can concurrently execute by changing the value of x. You can experiment with x being: 1, 2, 4, 8, 16, 24, 32, 48.

 

2) Extra Credit: Design and implement a scheme in GPGPUSIM that can dynamically figure out the best value of x such that IPC is maximum (or at least close to it). Again, x can only take values among: 1, 2, 4, 8, 16, 24, 32, 48.

The dynamic scheme should figure out x on its own and should not use x from gpgpusim.config. For this, you can create an option:

-gpgpu_dyamic_swl

If it is zero, x will be chosen from "-gpgpu_scheduler warp_limiting:2:x". Otherwise, if it is one, your scheme will kick in and find x for you.

 

Since we need to know the IPC on the fly, you may want to implement a profiling mechanism to get the IPC from a time window (e.g., 2K cycles). You may also want to have per-SM profiling since the schedulers on each SM work independently.

If you do not see the loss in IPC with the increase in the value of x, that is fine. The goal then should be to dynamically find as a low value of x as possible without losing out the max performance (IPC) considerably.

 

After successfully implementing and testing your scheme, you should then compare the IPC, L1, L2 miss-rates, and BW-utilization of it with the static scheme that uses a fixed x value. Again, test with the benchmark set and discuss your findings.

 

Note that you should finish as much as you can for this assignment. Your grade depends on the completeness and correctness of your experiment. Depending on your machine configuration, some of the experiments may run for a long time (this will not be the case if you have a powerful machine). You can let them run in the background and in parallel if possible. You should also create scripts to launch them continuously and collect results in a batch to save time. If you find that certain experiments last too long (e.g., more than 24 hours), you can also reduce the input size in the launch command and try again. However, make sure that the run time is at least one hour to generate stable outputs.

 

Things to submit:

1. A report in pdf format with the requirements listed above.

2. Your modified code in GPGPUSIM if you have completed Step 3.2.

Place your modified code part (do not paste the entire file) in the appendix of your report (with notes on the file name, function name, starting line, and other necessary information).
Also, place all your modified files in a zip folder and upload it to Canvas.
