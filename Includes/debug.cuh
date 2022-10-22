#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#if _DEBUG
	#define CHECK_CUDA(error) cuda_check(error, __FILE__, __LINE__)
#else
	#define CHECK_CUDA
#endif

static void print_error_string(const char* file, int line, const char* error)
{
	std::cout << '[' << file << "] " << line << " " << error << '\n';
}

static void cuda_check(cudaError error, const char* file, int line)
{
	switch (error)
	{
		case cudaSuccess:
			print_error_string(file, line, "SUCCESS");
			break;
		case cudaErrorInvalidValue:
			print_error_string(file, line, "Invalid Value");
			break;
		case cudaErrorInitializationError:
			print_error_string(file, line, "Initialization Error");
			break;
		case cudaErrorInvalidConfiguration:
			print_error_string(file, line, "Invalid Configuration");
			break;
		case cudaErrorInvalidHostPointer:
			print_error_string(file, line, "Invalid Host Pointer");
			break;
		case cudaErrorInvalidDevicePointer:
			print_error_string(file, line, "Invalid Device Pointer");
			break;
		case cudaErrorInvalidTexture:
			print_error_string(file, line, "Invalid Texture");
			break;
		case cudaErrorInvalidMemcpyDirection:
			print_error_string(file, line, "Invalid Memcpy Direction");
			break;
		case cudaErrorSynchronizationError:
			print_error_string(file, line, "Synchronization Error");
			break;
		case cudaErrorDuplicateVariableName:
			print_error_string(file, line, "DuplicateVariableName");
			break;
		case cudaErrorDevicesUnavailable:
			print_error_string(file, line, "Devices Unavailable");
			break;
		case cudaErrorIncompatibleDriverContext:
			print_error_string(file, line, "Incompatible Driver Context");
			break;
		case cudaErrorNoDevice:
			print_error_string(file, line, "No Device");
			break;
		case cudaErrorInvalidDevice:
			print_error_string(file, line, "Invalid Device");
			break;
		case cudaErrorDeviceUninitialized:
			print_error_string(file, line, "Device Uninitialized");
			break;
		case cudaErrorAlreadyAcquired:
			print_error_string(file, line, "Already Acquired");
			break;
		case cudaErrorInvalidGraphicsContext:
			print_error_string(file, line, "Invalid Graphics Context");
			break;
		case cudaErrorFileNotFound:
			print_error_string(file, line, "File Not Found");
			break;
		case cudaErrorSharedObjectSymbolNotFound:
			print_error_string(file, line, "Shared Object Symbol Not Found");
			break;
		case cudaErrorOperatingSystem:
			print_error_string(file, line, "OS Call failed");
			break;
		case cudaErrorInvalidResourceHandle:
			print_error_string(file, line, "Invalid Resource Handle");
			break;
		case cudaErrorSymbolNotFound:
			print_error_string(file, line, "Symbol Not Found");
			break;
		default:
		{
			const char* temp = "Error Code " + static_cast<int>(error);
			print_error_string(file, line, temp);
			break;
		}
	} // end of switch
}