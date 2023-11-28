In Assignment 3, we compared the CPU, GPU (naive), and an optimized GPU implementation of the Reduction algorithm. We then discussed more possible ways to further improve its performance in CMPE214_Lecture_5_Advanced_02.pdfActions . Now with the same testing configurations, implement, test, and compare the performance of Reduction by adding the following optimizations:

 

1. Loop unrolling

2. Warp shuffle instructions

 

For example, you should first add Loop unrolling on top of the kernel implementation with optimized thread organization. Then, on top of that, add warp shuffle instructions. After measuring their performance with the same launching parameters, you then can include two additional curves in your previously drawn figure in the assignment and explain your observations.

 

Things to submit:

1. Your working folder in a zip file for your comparison program.

2. A pdf report containing your timing output and a discussion about your findings.
