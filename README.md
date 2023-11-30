# GPU_Architecture_and_Programming

•	Hands-on experience in CUDA application development, including code download, compilation, and execution on both Jetson and Colab platforms. Conducted comprehensive performance analysis by iterating through block and thread combinations, measuring timings, and creating visualizations in Excel. Investigated the impact of varying block and thread numbers on performance, drawing conclusions aligned with device query results to optimize thread allocation for optimal performance.

•	 Installed and configured GPGPUSIM, a GPU simulator, through a detailed three-step process involving environment setup, CUDA installation, and simulator modification. Implemented and tested the GPGPU simulator with a CUDA application, facilitating performance, providing insights into GPU performance metrics such as IPC, occupancy, cache miss rates, simulation time, and simulation rate.

•	Experimented with p-chasing method by creating few data patterns in arrays to create hits and misses in caches and profiling L1, L2 cache line sizes.

•	Led comprehensive GPU benchmark experimentation, involving compilation and execution of GPGPUSIM benchmarks (JPEG, BFS, SLA, SCP, TRA, LPS) to analyze cache size, memory bandwidth, and scheduler effects. Conducted a sensitivity study, normalizing results to GTX 480 configuration, evaluating L1/L2 cache sizes, memory bandwidth, and warp scheduler variations. Implemented and tested a dynamic warp limiting schedule scheme, optimizing IPC and cache miss rates, and compared its performance with static configurations, providing insightful findings and visualizations.

•	Conducted a performance comparison study by assessing the usage of memcpy, pinned memory, and Unified Virtual Memory (UVM) with various hints, applied to various CUDA programs (e.g., MatrixMul and VectorAdd). Employed timing measurements and diverse data set sizes (32B to 2GB), accounting for data transfer overheads and ensuring fair comparison, offering insights into optimal memory allocation strategies.

•	Enhanced the performance evaluation of the Reduction algorithm by implementing and comparing the impact of optimizations, including loop unrolling and warp shuffle instructions, building upon the CPU, GPU (naive), and optimized GPU implementations. Integrated results into previously analyzed figures, providing insightful observations on the effectiveness of the added optimizations.
