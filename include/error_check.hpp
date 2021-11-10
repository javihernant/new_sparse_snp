#ifndef _ERROR_CHECK_HPP_
#define _ERROR_CHECK_HPP_

#include "cuda_runtime.h"

#define cuda_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"cudaError: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

// void cuda_check(cudaError_t status);

#endif