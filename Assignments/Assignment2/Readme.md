Note: all assignments should be done individually.

In this assignment, you will write two GPU versions of matrix multiplication code with CUDA C. The first version should use a naive approach (i.e., no optimization). The second version should use a tiled approach and shared memory. More detailed requirements are as follows: 


1. Your programs should be able to calculate the output matrix for two square input matrices. To simplify the design, you can assume that the width of the input matrix can only be selected from a power of 2. Input matrices should be initially stored in the host memory initialized with random numbers.


2. You can run your code with Google Colab or Jetson TX1. You should check what GPU you are assigned to and make sure the maximum size of shared memory is used by using the appropriate allocation method discussed during the class.


3. Use the CPU version of matrix multiplication you wrote in assignment 1 to verify the calculation result. For example, you can iterate through and compare all the elements in the output matrices. If anything does not match, you should print out a warning message "Verification Failed!". Otherwise, print "Verification Success!". Your final program should not produce any errors or warnings during compilation or runtime. 


4. According to the GPU's shared memory capacity and the input matrix size, you need to figure out what is the best tile size, block dim, and grid dim. For example, you can have measurements of your kernel's execution time and compare them for several different configurations. However, note that there is an upper bound for the block dim as well. Therefore, you may not be able to utilize all the available shared memory.


5. Measure the execution time with input sizes of 2 * 2, 4 * 4, ..., 8192 * 8192 for the GPU versions (naive and tiled) of the matrix multiplication with your optimal configurations. Draw a figure with your results together with the CPU result and explain your observations. 


6. According to the discussions during the class, consider the following question: If you have an excessive amount of shared memory compared to your block size, what are the possible optimization you could do? List your ideas and explain why they are beneficial. Then pick your favorite one to implement and evaluate the improvement (extra credit for the implementation). 

 

Things you need to submit in this assignment, in a PDF file and a zip file:

1. A PDF file containing the required writings as listed above.

2. Source code for your implementations. Include everything necessary in a folder and upload the zipped version. Note that you should provide a Makefile and your code should compile with no error or warning.
