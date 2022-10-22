#include "Includes/Buffer.cuh"
#include "Includes/Timer.cuh"

void addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    addWithCuda(c, a, b, arraySize);

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    CHECK_CUDA(cudaSetDevice(0));

    // Allocate GPU buffers for three vectors (two input, one output).
    CUDABuffer<int> dev_c;
    dev_c.Allocate(size);

    CUDABuffer<int> dev_a;
    dev_a.Allocate(size);

    CUDABuffer<int> dev_b;
    dev_b.Allocate(size);

    dev_a.CopyFrom(a, dev_a.GetByteSize());
    dev_b.CopyFrom(b, dev_b.GetByteSize());
    dev_c.CopyFrom(c, dev_c.GetByteSize());

    Timer timer;
    
    // Launch a kernel on the GPU with one thread for each element.
    timer.Start();
    addKernel<<<1, size>>>(dev_c.GetBuffer(), dev_a.GetBuffer(), dev_b.GetBuffer());
    timer.End();

    timer.Print("addKernel");

    // Check for any errors launching the kernel
    CHECK_CUDA(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    dev_c.CopyTo(c, dev_c.GetByteSize());
}
