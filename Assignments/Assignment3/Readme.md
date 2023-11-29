At the end of CMPE214_Lecture_4_Patterns_02.pdfActions , we discussed a naive implementation of a Reduction kernel and also a possible way to better organize its threads to reduce the waste of threads resources in SMs. Now with the original code and the hint of the optimization provided, implement, test, and compare the performance of Reduction with the following configurations:

1. Serial (CPU)

2. No optimization (naive GPU method)

3. GPU with optimized thread organization (as indicated in the hint)

 

You can use simple patterns to initialize your input array and verify the GPU results with the CPU result. Also, use different grid and block size combinations to observe the performance difference. For example, you can choose grid sizes of 1, 2, 4, 8, ..., 128, and for each of them use block size 32, 64, 128, ..., 1024. Then based on the execution time measurement (which should exclude the data copying part and be on the GPU side for better accuracy), draw a figure and explain your observations.

 

Things to submit:

1. Your working folder in a zip file for your comparison program.

2. A pdf report containing your timing output and a discussion about your findings.
