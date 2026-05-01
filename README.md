# pytorch-cuda-kernel-bridge

This repository demonstrates a high-performance bridge between high-level PyTorch and custom CUDA C++ kernels. It features a custom polynomial activation operator ($x^2 + x + 1$) implemented at the hardware level to showcase the efficiency gains of bypassing standard framework abstractions.

## Implementation Strategy

### The CUDA Kernel (polynomial_cuda.cu)

The core computation is handled by a templated CUDA kernel designed for maximum throughput:Vectorized Parallelization: Utilizes 1024 threads per block to process element-wise operations across large tensors.Type Agnostic: Uses AT_DISPATCH_FLOATING_TYPES to ensure compatibility with various floating-point precisions.Memory Efficiency: Employs __restrict__ pointers to signal the compiler for better memory optimization.

### The Python Bridge (load)

Rather than using a traditional setup.py build, this project utilizes torch.utils.cpp_extension.load():Flexibility: The Just-In-Time (JIT) approach handles CUDA and PyTorch version mismatches dynamically, preventing common compilation errors during development.

Optimization: Custom compiler flags like -O3 are passed during runtime to ensure the extension is fully optimized for execution.

## Benchmarking & Results

The custom CUDA implementation was tested against the native PyTorch functional implementation ($x^2 + x + 1$) on a tensor of 1,000,000 elements.### Performance ComparisonImplementationAverage Execution Time (ms)SpeedupPyTorch Built-in0.1375 ms1.0xCustom CUDA Extension0.0374 ms~3.67x Faster### Why is the Custom Kernel Faster?While PyTorch is highly optimized, a custom CUDA kernel allows for a Fused Operation. PyTorch might execute this polynomial as multiple discrete operations ($x \cdot x \rightarrow + x \rightarrow + 1$), involving multiple memory round-trips. The custom kernel performs the entire calculation in a single pass over the data, significantly reducing global memory bandwidth pressure.

## How to RunEnsure you have the CUDA Toolkit and PyTorch installed in your environment.

Open the polynomial_activation.ipynb notebook.The load() function will automatically compile the .cu file on the first run and link it to your Python session.

## Project Significance

This project serves as a technical proof-of-concept for Performance Engineers needing to optimize bottlenecked AI layers. It demonstrates the ability to write low-level GPU code and integrate it seamlessly into modern AI research workflows.
