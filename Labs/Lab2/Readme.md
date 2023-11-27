In this Lab, you will have hands-on experience in writing and compiling a CUDA application (probably the first time for some of you) to prepare for future CUDA programming tasks. Simply follow the steps below and submit your results to Canvas. 

1. Download the code: lab2.zip Download lab2.zip

 

2. Complete, compile and run the code (both Jetson and Colab are fine).

 

3. Try with different input sizes and launch dims and measure their timings (you should probably automate this process using loops and you can use CPU clocks for now), what are your observations?

Iterating through combinations between the number of blocks (1, 2, 4, 8, ..., 1024) and the number of threads per block (1, 2, 4, 8, ..., 1024).
For the convenience of observation, you can print out the timing and convert them into an excel sheet. Then draw figures with selected rows and columns.
Keep the number of blocks fixed, how does changing the number of threads per block affect the performance? Observe this from at least three different numbers of blocks from small to large.
Keep the number of threads per block fixed, how does changing the number of blocks affect the performance? Observe this from at least three different numbers of threads per block from small to large.
Keeping the total number of threads fixed, how does the number of blocks and the number of threads per block allocation affect the performance? Observe this from at least three different total numbers of threads from small to large.
Along with the device query result from Lab1, what is your conclusion? And by following what principle can we allocate threads to obtain optimal performance?
 

4. Upload a zip file of your completed working folder (like what is provided to you in lab2.zip) and a report (in PDF) for your results, figures, observations, discussions, and conclusions.
