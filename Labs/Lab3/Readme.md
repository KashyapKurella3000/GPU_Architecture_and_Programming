Note: This is a team lab.

In this lab, we are going to install, setup, and run the GPU simulator, GPGPUSIM.

Some instructions are provided in its Github repo (https://github.com/gpgpu-sim/gpgpu-sim_distribution). If you need, you can try building it yourself with your preferred CUDA versions. However, this process can be complicated for first-time users. Therefore, it is recommended that you follow the steps introduced in this lab.

 


Step 1: Setup the environment

In our previous assignment, we have already installed GCC 4.5.1 on an x86 Ubuntu system, which has been verified to work with the simulator. You can go back to assignment 1 if you need instructions on this.

 

Then, install some other dependencies with the following commands:
sudo apt-get install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev

sudo apt-get install doxygen graphviz

sudo apt-get install python-pmw python-ply python-numpy libpng12-dev python-matplotlib

sudo apt-get install libxi-dev libxmu-dev

sudo apt-get install freeglut3 freeglut3-dev

sudo apt install libboost-all-dev

Note that, if the above packages do not solve all your dependency issues, you can also install additional ones by following the prompts. It should be pretty easy on Ubuntu.

 

Next, we also need to install CUDA 4.2 which can be found here (https://developer.nvidia.com/cuda-toolkit-42-archive).

We need both the toolkit (i.e., 1) and sdk (i.e., 3) for Linux. For the toolkit, you can download the 64-bit version for Ubuntu 10.04 (or other versions depending on your environment).

 

First, you should install the toolkit by using:
sh cudatoolkit_4.2.9_linux_64_ubuntu10.04.run

The installation simply tells you to specify the installation path and setup some environment variables.

As an example, you can use the installation path:
[your_path]/toolkitcuda42 

 

And after installation, add the following lines in your .bashrc file in your home folder:

export CUDA_INSTALL_PATH=[your_path]/toolkitcuda42/cuda

export CUDAHOME=$CUDA_INSTALL_PATH

export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export PATH=$CUDA_INSTALL_PATH/bin:$PATH

export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/include:$LD_LIBRARY_PATH

export PATH=$CUDA_INSTALL_PATH/include:$PATH

 

Run the following command after editing the file:
source .bashrc

And also use:
nvcc --version

to verify your installation.


Second, you should install the sdk by using:
sh gpucomputingsdk_4.2.9_linux.run


Again, as an example, you can use the installation path:
[your_path]/sdkcuda42

And when it says "Could not locate CUDA.  Enter the full path to CUDA ...", you should also provide: 
[your_path]/toolkitcuda42/cuda

 

After installation, add the following lines in your .bashrc file and run source .bashrc:

export NVIDIA_CUDA_SDK_LOCATION=[your_path]/sdkcuda42

export NVIDIA_COMPUTE_SDK_LOCATION=$NVIDIA_CUDA_SDK_LOCATION


Then, go to [your_path]/sdkcuda42 and type make.

It is possible that you will see the compilation terminates in the middle because of some errors.

But that is actually fine, go to [your_path]/sdkcuda42/C/lib and libcutil_x86_64.a should be there. That is what we need. If it does not appear, you should let me know.

 

 

Step 2: Installing GPGPUSIM

Go to your preferred path and run the following command:
git clone https://github.com/gpgpu-sim/gpgpu-sim_distribution

What we are using is GPGPUSIM 4.0 which is a recent release of the simulator. Therefore, we first need to modify a few files in the simulator to make it work for our environment and also more convenient to use.


First, go to the gpgpu-sim_distribution folder and open Makefile. 

Find the following line (should be line 146 currently):
g++ -shared -Wl,-soname,libcudart.so -Wl,--version-script=linux-so-version.txt\

And change it to:
g++ -shared -Wl,-soname,libcudart.so.4 -Wl,--version-script=linux-so-version.txt\


Second, go to gpgpu-sim_distribution/src/cuda-sim folder and open cuda-sim.cc.

Find the following line (should be line 1927 currently):
DPRINTF(LIVENESS, "GPGPU-Sim PTX: %u instructions simulated : ctaid=(%u,%u,%u) "

And change it to:
printf("GPGPU-Sim PTX: %u instructions simulated : ctaid=(%u,%u,%u) "


Third, still under the gpgpu-sim_distribution/src/cuda-sim folder, open ptx_loader.cc.

Find the following line (should be line 67 currently):
"saves ptx files embedded in binary as <n>.ptx", "0");

And change it to:
"saves ptx files embedded in binary as <n>.ptx", "1");


Next, we can compile the simulator.

First, go to the gpgpu-sim_distribution folder and do the following:
source setup_environment

 

Then you can type make to compile the simulator. You can use the following command to speed up the compilation:
make -j12

The number after j indicates the number of threads used in the compilation.

 

If your environment is setup correctly by following all the previous steps, the compilation should finish with no errors. Go to [your_path]/gpgpu-sim_distribution/lib/gcc-4.5.1/cuda-4020/release (or a slightly different path depending on your GCC and CUDA versions) and you should see the generated shared libraries (.so files).

If you see any dependency errors, try to install those dependencies. If there are errors that you cannot fix yourself, you can let me know. And again, using an x86 Ubuntu system is the easiest way for a successful installation. Try using a virtual machine if it does not work on your local system.

 

After installation, add the following lines in your .bashrc file and run source .bashrc:

export GPGPUSIM_ROOT=[your_path]/gpgpu-sim_distribution

export LD_LIBRARY_PATH=$GPGPUSIM_ROOT/lib/gcc-4.5.1/cuda-4020/release

 

 

Step 3: Compile and run a CUDA application with GPGPUSIM

To use the simulator, we need to link the shared library generated from the simulator instead of the actual CUDA library when we compile a CUDA program (see lecture 3, Profiling and Simulation).

As an example, we can use the vecadd program previously written for the real GPU to test the simulator.

 

In your working folder, you can use the make command by placing your main.cu and these two files:

Makefile

common.mk Download common.mk

common.mk will take care of the linking process if all the environment variables mentioned earlier are set correctly. And since the simulator is connected through a shared library, we only need to compile the application once for a specific CUDA and GCC version combination. You can modify some implementations in the simulator and recompile it without affecting the compiled applications.

 

After compiling the CUDA application, you should see an executable, gpgpu_ptx_sim__[appname], appear in the folder. 

Use ldd gpgpu_ptx_sim__[appname] command and make sure that libcudart.so.4 is linked and is pointing to the shared library in the simulator's folder. If not, you probably did not make the correct change in the simulator's Makefile. 

 

Next, we will need to provide some configuration files to tell the simulator the GPU hardware details we want to simulate.

The following folder contains a few preset configurations:
gpgpu-sim_distribution\configs\tested-cfgs

For example, to use the GTX480 Fermi architecture that we are familiar with, simply go to SM2_GTX480 and copy all files to your working folder that contains the executable.

 

Then, you can simply run the executable and observe the simulation results.

The output should end with your application's output (e.g., if you print out the vecadd result, but the computation process is simulated) and also some timing information like this:
gpgpu_simulation_time = 0 days, 0 hrs, 0 min, 1 sec (1 sec)

If this is not the case or the simulation is stuck, then you probably did not make the correct change in cuda-sim.cc. You should let me know if you are not able to fix it.

 

It is recommended that you redirect the output to a file (e.g., output.txt) and observe the output from there.

What you will see are some detailed statistics that cannot be easily obtained with real hardware (which is one of the reasons why simulators are useful). You can quickly scan through the different items to see how your application performs under GTX480. Later, we will discuss more details about these outputs.

 

Things to submit:

1. Simulation output for your preferred application and configuration. 

However, it is recommended to start with a simple vecadd and GTX480 configuration for easier debugging. 


2. In a separate text file, answer the following questions by searching in your output file:

What is the value of:

1) gpu_tot_ipc

2) gpu_occupancy

3) L1D_total_cache_miss_rate

4) L2_total_cache_miss_rate

5) gpgpu_simulation_time

6) gpgpu_simulation_rate
