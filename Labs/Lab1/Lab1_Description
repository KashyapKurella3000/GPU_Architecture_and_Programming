In this Lab, you will learn how to set up the Jetson TX1 system and also check its configurations using a CUDA sample program. Please follow the steps below. After you finish, answer the questions and submit them to Canvas. 

 

1. Please get your Jetson board from me and complete the equipment check-out formActions  (will be provided during the class). Please follow the regulations listed on the form. Make sure your board is working properly.

Test your board by connecting it to a monitor via HDMI cable. Connect keyboard and mouse to the USB hub attached to the board. Then turn on the provided Jetson board.

Login credentials:
ID: nvidia
PASSWORD: nvidia


2. If your machine has Ubuntu 16.04 or 18.04 skip this step. Download, install, and setup VMware Workstation Player (https://www.vmware.com/products/workstation-player.html).

After installing VMware Workstation Player. Create a virtual machine with Ubuntu 18.04 (https://releases.ubuntu.com/18.04/) and 256GB disk. Assign more resources to the VM for better performance. Use USB 3.1 and bridged network (make sure to disable routers for other VMs in your system).

Install the OS, and then open a terminal and do:
sudo apt-get install libxml2-utils


After the installation, power off and go to the VM folder. Open your .vmx file and insert those two lines:

usb.autoConnect.device0 = "vid:0x0955 pid:0x7020 autoclean:1"
usb.autoConnect.device1 = "vid:0x0955 pid:0x7721 autoclean:1"

This makes sure the USB cable is working properly during the flashing.


3. Download and install Nvidia SDK Manager (https://docs.nvidia.com/sdk-manager/index.html) in Ubuntu 18.04. Connect your Jetson board to the host machine and turn it to force recovery mode (hold the force recovery button then power up or reset). Then follow the steps in SDK Manager.

In the SDK Manager, you should deselect the host installation and choose Jetpack 4.6.

If the flashing fails, you should debug it. Go to the downloaded Jetpack folder and do:
sudo ./flash.sh jetson-tx1-devkit mmcblk0p1

Observe and fix the errors.


4. After flashing, your board would automatically exit force recovery mode and enter the OS installation stage. Simply follow the instructions to start the OS installation. And then install CUDA tools on the board.

Please use the following login credentials for the board:
ID: nvidia
PASSWORD: nvidia

After the OS installation, you can continue to install the CUDA tools. Use ifconfig command on the board to make sure the IP address is correct on the host.

Due to the time limitation, you can stop after installing CUDA. You can install the rest of the tools later.

You can also try with Virtual Box (but you need to download the extension pack to use USB 3.1 and set filters for the vid and pid) if VMware does not work.


5. Go to the CUDA samples. Compile and run DeviceQuery. Then answer a few questions. 

1) What is the CUDA driver version?

2) What is the CUDA compute capability?

3) What is the number of Multiprocessors?

4) What is the number of CUDA Cores per Multiprocessor?

5) What is the size of constant memory?

6) What is the max size of shared memory per thread block?

7) What is the max register number per thread block?


If you cannot set up your board, DeviceQuery is also available here: DeviceQuery.zip Download DeviceQuery.zip You can run it on Google Colab.

 

6. Submit your results to Canvas.
1) A text file containing your answers.
2) A screenshot of OS and nvcc version on the board.

 

Useful command:
tegrastats
https://docs.nvidia.com/drive/drive_os_5.1.6.1L/nvvib_docs/index.html#page/DRIVE_OS_Linux_SDK_Development_Guide/Utilities/util_tegrastats.html


Useful resources:
Download and Run SDK Manager
https://docs.nvidia.com/sdk-manager/download-run-sdkm/index.html

Install Jetson Software with SDK Manager
https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html

NVIDIA Jetson Linux Driver Package Software Features
https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3261/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/introduction.html#

Jetson/General debug
https://elinux.org/Jetson_TX1
https://elinux.org/Jetson/General_debug

NVIDIA® Jetson™ Linux Driver Package (L4T)
https://developer.nvidia.com/embedded/develop/software

Automatically connecting USB devices at virtual machine power on
https://kb.vmware.com/s/article/1648


Sample location:
TensorRT 
/usr/src/tensorrt/samples/ 

cuDNN 
/usr/src/cudnn_samples_<version>/ 

CUDA 
/usr/local/cuda-<version>/samples/ 

MM API 
/usr/src/jetson_multimedia_api 

VisionWorks 
/usr/share/visionworks/sources/samples/  
/usr/share/visionworks-tracking/sources/samples/  
/usr/share/visionworks-sfm/sources/samples/  

OpenCV 
/usr/share/opencv4/samples/ 

VPI
/opt/nvidia/vpi/samples/
