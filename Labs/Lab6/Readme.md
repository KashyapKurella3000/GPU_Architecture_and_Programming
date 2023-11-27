Compare the performance of using memcpy, pinned memory, and UVM (also with different hints of UVM). You can simply use the code of the previous assignments (e.g., MatrixMul or VectorAdd) with timing measurements and replace their memory allocation method. Make sure you also include the time spent on memcpy for a fair comparison, since pinned memory and UVM do not need that. Rather, their data transfer overhead is amortized on each memory access or each page miss.

Also, you should test with data sets of different sizes (e.g., 32B, 64B, ..., 2GB) for this experiment.

Pinned Example:
  int *CPU_data_in;

  cudaHostAlloc((void**)&CPU_data_in, sizeof(int) * data_size, cudaHostAllocDefault);

  cudaFreeHost(CPU_data_in);

UVM Example:
  cudaMallocManaged(&CPU_data_in, sizeof(int) * data_size);

  cudaFree(CPU_data_in);

Hints:
  cudaMemAdvise(CPU_data_in, sizeof(int) * data_size, cudaMemAdviseSetAccessedBy, dev_id);

  cudaMemAdvise(CPU_data_in, sizeof(long long int) * data_size, cudaMemAdviseSetPreferredLocation, dev_id));

 

Things to submit:

1. Your working folder in a zip file for your comparison program.

2. A PDF file containing your timing output (using tables and figures) and a discussion about your findings.
