#ifndef _ERROR_CHECK_HPP_
#define _ERROR_CHECK_HPP_

#include "cuda_runtime.h"
#include <cusparse.h> 

#define cuda_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"cudaError: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline void gpuAssert(cublasStatus_t code, const char *file, int line, bool abort=true){
   if (code != CUBLAS_STATUS_SUCCESS)
    {
        switch (code) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
            fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_NOT_INITIALIZED file: %s line: %d ", file, line);
            break; 
            
            case CUBLAS_STATUS_ALLOC_FAILED:
            fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_ALLOC_FAILED file: %s line: %d ", file, line); 
            break; 

            case CUBLAS_STATUS_INVALID_VALUE:
            fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_INVALID_VALUE file: %s line: %d ", file, line); 
            break; 

            case CUBLAS_STATUS_ARCH_MISMATCH:
            fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_ARCH_MISMATCH file: %s line: %d ", file, line); 
            break; 

            case CUBLAS_STATUS_MAPPING_ERROR:
            fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_MAPPING_ERROR file: %s line: %d ", file, line); 
            break; 

            case CUBLAS_STATUS_EXECUTION_FAILED:
            fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_EXECUTION_FAILED file: %s line: %d ", file, line); 
            break; 

            case CUBLAS_STATUS_INTERNAL_ERROR:
            fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_INTERNAL_ERROR file: %s line: %d ", file, line); 
            break; 

            case CUBLAS_STATUS_NOT_SUPPORTED:
            fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_NOT_SUPPORTED file: %s line: %d ", file, line); 
            break; 

            case CUBLAS_STATUS_LICENSE_ERROR:
            fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_LICENSE_ERROR file: %s line: %d ", file, line); 
            break; 
        }
        if (abort) exit(code);
    }
}

inline void gpuAssert(cusparseStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSPARSE_STATUS_SUCCESS) 
   {
      fprintf(stderr,"cuSPARSE: %s (%d) %s %d\n", cusparseGetErrorString(code), code, file, line);
      if (abort) exit(code);
   }
}

#endif