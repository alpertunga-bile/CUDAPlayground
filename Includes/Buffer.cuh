#pragma once

#include "debug.cuh"

template<typename T>
class CUDABuffer
{
public:
	CUDABuffer() : size(0), byteSize(0), buffer(nullptr) {}
	~CUDABuffer() { DeAllocate(); }

	void Allocate(unsigned int _size)
	{
		size = _size;
		byteSize = size * sizeof(T);

		CHECK_CUDA(cudaMalloc((void**)&buffer, byteSize));
	}

	void CopyFrom(const T* from, size_t _size, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
	{
		if (_size > byteSize) return;

		CHECK_CUDA(cudaMemcpy(buffer, from, _size, kind));
	}

	void CopyTo(void* to, size_t _size, cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
	{
		if (_size > byteSize) return;

		CHECK_CUDA(cudaMemcpy(to, buffer, _size, kind));
	}

	void DeAllocate()
	{
		if (buffer == nullptr) return;

		cudaFree(buffer);
	}

	T* GetBuffer() { return buffer; }
	size_t GetBufferSize() { return byteSize; }
	unsigned int GetSize() { return size; }

private:
	unsigned int size;
	size_t byteSize;
	T* buffer;
};