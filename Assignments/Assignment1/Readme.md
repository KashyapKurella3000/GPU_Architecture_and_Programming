In this assignment, you will get a Linux system (Ubuntu) ready on your machine. Depending on your current OS, finish either item 1 or item 2. If you already have one, go directly to item 3.

1. If you are using Windows 10 or 11, you can install Ubuntu using the Windows Subsystem for Linux (WSL).


If you are using windows 10, follow this tutorial:
https://docs.microsoft.com/en-us/windows/wsl/install-win10 Links to an external site.

There might be risks for joining the Windows Insiders program, so you can do it manually. But either way is fine.

If you want to paste something to the terminal, simply copy that and then right-click your mouse in the terminal. You can also copy something from the terminal by first selecting it, then right-click with your mouse. 

There is a troubleshooting section for manual installation in the tutorial. However, you may encounter some other problems if you are on the latest version of Windows 10:

(1) If the wsl command is not recognized in Powershell. Run it in CMD instead.

(2) If you see "Error: 0x80370102 The virtual machine could not be started because a required feature is not installed." You need to enable virtual machine in BIOS. How to do that depends on your actual hardware. Simply search in Google with your machine's configuration and there should be plenty of answers.

For other issues, Google it first and let me know if you cannot solve them.


If you are using windows 11, this process is quite simple. Simply follow this tutorial:
https://www.windows11.dev/ce7in/how-to-install-wsl2-and-linux-distros-on-windows-11-6od Links to an external site.


2. If you are using Mac OS, you can use a virtual machine to run Ubuntu. If you own previous versions of MacBook with Intel CPUs, follow this guideActions  by Dr. Ronald Mak to install VirtualBox and Ubuntu. However, if you own the current version of MacBook with M1 CPUs, you can only use UTM, Parallels, or Docker. But still, there are no free and good solutions to emulate x86 CPUs on M1. Therefore, if you only have access to a MacBook with M1 CPU, please let me know.


3. Now you should have access to a terminal with Bash. You can get gcc with "sudo apt install gcc" command and use it to compile the C code that you are going to write.


4. Write two CPU versions of matrix multiplication (i.e., M * N = P) code in C.

First with the naive implementation. Second with memory optimization techniques discussed during the lecture (you may add more on top of those). Your programs should be able to take arbitrary input matrix sizes and calculate the output matrix. All the matrices must be arrays stored in the memory (i.e., reserved with malloc). Input matrix values should be randomly generated integers.

Also, measure the performance of your programs by getting their execution times (the higher precision the better). Your code should not produce any errors or warnings during compilation. Experiment with input sizes of 2 * 2, 4 * 4, ..., 8192 * 8192. Draw a figure with your results and explain your observations.


5. Next, you need to know how to install a specific (and old) version of gcc. For now, let us first install gcc 4.5.1. 

(1) To do so, go to:
https://gcc.gnu.org/mirrors.html Links to an external site.

Links to an external site.Select your preferred mirror site and find the gcc-4.5.1 folder. Then download the gcc-4.5.1.tar.gz file and place it in your preferred installation location. In the command line, do the following:

tar xzf gcc-4.5.1.tar.gz
cd gcc-4.5.1

 

(2) Before we start compilation you may want to install the following if you have not. If you see other requirements you can also install them.

sudo apt install gcc-multilib
sudo apt install make

 

(3) Also, do:
find /usr/ -name crti*

The returning path would be ##path that contains crti.o##.

 

Then run "echo $LIBRARY_PATH" to see if it is empty.

If yes:

export LIBRARY_PATH=##path that contains crti.o##

If not:

export LIBRARY_PATH=##path that contains crti.o##:$LIBRARY_PATH

in .bashrc in your home folder (similarly for later exports).


(4) Next, download patch_linux_unwind.patch Download patch_linux_unwind.patch and run:

patch -p0 -b --ignore-whitespace --fuzz 3   < patch_linux_unwind.patch

And specify the file to patch: ##path to gcc-4.5.1##\gcc\config\i386\linux-unwind.h

 

If an error occurred later during make for linux-unwind.h, but not from the i386 folder, you can also look at the content in the patch and try to fix it yourself. 


(5) Then, download and place download_prerequisites Download download_prerequisites inside /contrib. Change permission with "sudo chmod 747 download_prerequisites" if needed. Then:

cd..
./contrib/download_prerequisites


(6) Next, create and go to the objective folder (replace ##your_obj_path## with your path):

cd ..
mkdir ##your_obj_path##
mkdir ##your_installation_path##
cd ##your_obj_path##

 

And specify your installation path (replace ##your_installation_path## with your path):

$PWD/../gcc-4.5.1/configure --prefix=##your_installation_path## --enable-languages=c,c++,fortran 

 

Solve any errors that have appeared. Then type "make". This step will take a long time. 

See if there are any errors. Search online and try to fix them. Also, ask your teammate and other students for help. If no luck, come to me during office hours. However, there is no guarantee that I can fix your problem immediately, especially remotely. If you encounter any error, make sure to run "make distclean" and "rm ./config.cache" and start over.


(7) If the compilation passed in the previous step, you can then type "make install". Again, see if there are any errors at the end. If yes, also fix them.


6. Then, you need to add the path ##your_installation_path##/bin to your $PATH in .bashrc with

export PATH=##your_installation_path##/bin:$PATH

And also add:

export LD_LIBRARY_PATH=##your_installation_path##/lib64
export LD_RUN_PATH=##your_installation_path##/lib64

 

7. Test with "gcc --version". It should show gcc 4.5.1. Otherwise, there might be an error in your $PATH.


There are some other guides you can refer to:

https://gcc.gnu.org/install/ Links to an external site.

https://gcc.gnu.org/wiki/InstallingGCC Links to an external site.

 

Things you need to submit in this assignment, in a PDF file and a zip file:

1. A screenshot of the result of "uname -a" in your command line.

2. Source code for your matrix multiplication programs. Include everything necessary in a folder and upload the zipped version.

3. A chart showing all the experimental results and a detailed discussion about your observations and analysis. 

4. A screenshot of the result of "gcc --version" in your command line.
