
#ifndef GERYON_H
#define GERYON_H

#include <string>



#ifdef __CUDACC__ // if it's in .cu file
#define __global
#define GLOBAL_ID_X threadIdx.x+mul24(blockIdx.x,blockDim.x)
#define GLOBAL_ID_Y threadIdx.y+mul24(blockIdx.y,blockDim.y)
#define THREAD_ID_X threadIdx.x
#define THREAD_ID_Y threadIdx.y
#define BLOCK_ID_X blockIdx.x
#define BLOCK_ID_Y blockIdx.y
#define BLOCK_SIZE_X blockDim.x
#define BLOCK_SIZE_Y blockDim.y
#define __kernel extern "C" __global__
#define __local __shared__
#define mul24 __mul24
#define __inline __host__ __device__
#define atomic_add atomicAdd

#elif defined(cl_khr_global_int32_base_atomics) // if it's in .cl file

#define GLOBAL_ID_X get_global_id(0)
#define GLOBAL_ID_Y get_global_id(1)
#define THREAD_ID_X get_local_id(0)
#define THREAD_ID_Y get_local_id(1)
#define BLOCK_ID_X get_group_id(0)
#define BLOCK_ID_Y get_group_id(1)
#define BLOCK_SIZE_X get_local_size(0)
#define BLOCK_SIZE_Y get_local_size(1)
#define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE)
#define __inline inline
#define atomicAdd atomic_add

#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#else



#if defined(USE_OPENCL) || defined (UCL_OPENCL) // use OpenCL

const char * OpenCl_AddStr = 
"#define GLOBAL_ID_X get_global_id(0)\n"
"#define GLOBAL_ID_Y get_global_id(1)\n"
"#define THREAD_ID_X get_local_id(0)\n"
"#define THREAD_ID_Y get_local_id(1)\n"
"#define BLOCK_ID_X get_group_id(0)\n"
"#define BLOCK_ID_Y get_group_id(1)\n"
"#define BLOCK_SIZE_X get_local_size(0)\n"
"#define BLOCK_SIZE_Y get_local_size(1)\n"
"#define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE)\n"
"#define __inline inline\n"
"#ifdef cl_khr_fp64\n"
"    #pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#elif defined(cl_amd_fp64)\n"
"    #pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
"#else\n"
"    #error \"Double precision floating point not supported by OpenCL implementation.\"\n"
"#endif\n"
"\n"
;

#include "ocl_device.h"
#include "ocl_mat.h"
#include "ocl_kernel.h"
#include "ocl_macros.h"
#include "ocl_memory.h"
#include "ocl_texture.h"
#include "ocl_timer.h"
using namespace ucl_opencl;


#elif defined(USE_CUDA) || defined(UCL_CUDADR) // use CUDA

#include "nvd_device.h"
#include "nvd_mat.h"
#include "nvd_kernel.h"
#include "nvd_macros.h"
#include "nvd_memory.h"
#include "nvd_texture.h"
#include "nvd_timer.h"
using namespace ucl_cudadr;

#elif defined(UCL_CUDART)

#error "The lammps programmer removed this paragraph and I do not bother to add it back."

#endif

// Standard ucl headers
#include "ucl_basemat.h"
#include "ucl_copy.h"
#include "ucl_d_mat.h"
#include "ucl_d_vec.h"
#include "ucl_h_mat.h"
#include "ucl_h_vec.h"
// #include "ucl_image.h"
#include "ucl_matrix.h"
#include "ucl_nv_kernel.h"
#include "ucl_print.h"
#include "ucl_types.h"
#include "ucl_vector.h"
// #include "ucl_version.h"
// #include "ucl_tracer.h"

/**
 * \brief Converts to human readable error.
 * \param result Code that has been returned by geryon call.
 * \return The string corresponding to the result code.
 */
inline std::string ucl_check(int result)
{
   switch(result) {
      case UCL_ERROR:
         return std::string("UCL_ERROR"); break;
      case UCL_SUCCESS:
         return std::string("UCL_SUCCESS"); break;
      case UCL_COMPILE_ERROR:
         return std::string("UCL_COMPILE_ERROR"); break;
      case UCL_FILE_NOT_FOUND:
         return std::string("UCL_FILE_NOT_FOUND"); break;
      case UCL_FUNCTION_NOT_FOUND:
         return std::string("UCL_FUNCTION_NOT_FOUND"); break;
      case UCL_MEMORY_ERROR:
         return std::string("UCL_MEMORY_ERROR"); break;
   }
   return std::string("Unknown");
}



#endif
#endif