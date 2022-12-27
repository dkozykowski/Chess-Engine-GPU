#include <algorithm>
#include <macros.cuh>

namespace SCAN {

#define THREADS_IN_BLOCK 1024
#define ELEMENTS_PER_BLOCK 2 * THREADS_IN_BLOCK
#define SHARED_MEMORY_SIZE sizeof(int) * ELEMENTS_PER_BLOCK

unsigned int closestPowerOf2(unsigned int value) {
    int power = 1;
    while (power < value) power *= 2;
    return power;
}

__global__ void firstStageScan(unsigned int *input, unsigned int n,
                               unsigned int *sums, unsigned int closestPower) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1;

    temp[2 * tid] = input[2 * globalTid];
    temp[2 * tid + 1] = input[2 * globalTid + 1];

    for (int d = closestPower >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();

    if (tid == 0) {
        sums[blockIdx.x] = temp[closestPower - 1];
        temp[closestPower - 1] = 0;
    }
    for (int d = 1; d < closestPower; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    input[2 * globalTid] = temp[2 * tid];
    input[2 * globalTid + 1] = temp[2 * tid + 1];
}
__global__ void add(unsigned int *input, unsigned int *sums) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockId = blockIdx.x;

    input[2 * globalIdx] += sums[blockId];
    input[2 * globalIdx + 1] += sums[blockId];
}

int getBlockCount(int n) {
    if (n % (ELEMENTS_PER_BLOCK) == 0)
        return n / (ELEMENTS_PER_BLOCK);
    else
        return n / (ELEMENTS_PER_BLOCK) + 1;
}

void scan(unsigned int *input, unsigned int n, unsigned int *sums,
          unsigned int *totalSum)  // not ideal but allows up to 2048^3 elements
{
    if (n > (ELEMENTS_PER_BLOCK)) {
        unsigned int blockCount = getBlockCount(n);
        firstStageScan<<<blockCount, THREADS_IN_BLOCK, SHARED_MEMORY_SIZE>>>(
            input, ELEMENTS_PER_BLOCK, sums, ELEMENTS_PER_BLOCK);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        if (blockCount > (ELEMENTS_PER_BLOCK)) {
            unsigned int offset = closestPowerOf2(blockCount);
            unsigned int newBlockCount = getBlockCount(blockCount);

            firstStageScan<<<newBlockCount, THREADS_IN_BLOCK,
                             sizeof(int) * ELEMENTS_PER_BLOCK>>>(
                sums, ELEMENTS_PER_BLOCK, sums + offset, ELEMENTS_PER_BLOCK);
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaPeekAtLastError());

            int powOf2 = closestPowerOf2(newBlockCount);
            firstStageScan<<<1, (powOf2 + 1) / 2,
                             sizeof(int) * ELEMENTS_PER_BLOCK>>>(
                sums + offset, newBlockCount, totalSum, powOf2);
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaPeekAtLastError());

            add<<<newBlockCount, THREADS_IN_BLOCK>>>(sums, sums + offset);
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaPeekAtLastError());
        } else {
            int powOf2 = closestPowerOf2(blockCount);
            firstStageScan<<<1, (powOf2 + 1) / 2,
                             sizeof(int) * ELEMENTS_PER_BLOCK>>>(
                sums, blockCount, totalSum, powOf2);
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaPeekAtLastError());
        }
        add<<<blockCount, THREADS_IN_BLOCK>>>(input, sums);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
    } else {
        firstStageScan<<<1, (closestPowerOf2(n) + 1) / 2,
                         sizeof(int) * ELEMENTS_PER_BLOCK>>>(
            input, n, totalSum, closestPowerOf2(n));
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
    }
}
} // namespace SCAN