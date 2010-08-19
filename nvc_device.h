/***************************************************************************
                                nvc_device.h
                             -------------------
                               W. Michael Brown

  Utilities for dealing with cuda devices

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Wed Jan 28 2009
    copyright            : (C) 2009 by W. Michael Brown
    email                : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef NVC_DEVICE
#define NVC_DEVICE

#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>
#include "nvc_macros.h"
#include "ucl_types.h"

namespace ucl_cudart {

// --------------------------------------------------------------------------
// - COMMAND QUEUE STUFF
// --------------------------------------------------------------------------
typedef cudaStream_t command_queue; 

inline void ucl_sync(cudaStream_t &stream) {
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
}

/// Class for looking at device properties
/** \note Calls to change the device outside of the class results in incorrect
  *       behavior 
  * \note There is no error checking for indexing past the number of devices **/
class UCL_Device {
 public:
  /// Collect properties for every GPU on the node
  /** \note You must set the active GPU with set() before using the device **/
  UCL_Device();
  
  ~UCL_Device();

  /// Returns 1 (For compatibility with OpenCL)
  inline int num_platforms() { return 1; }

  /// Return a string with name and info of the current platform
  std::string platform_name() { return "NVIDIA Corporation NVIDIA CUDA"; }

  /// Return the number of devices that support CUDA
  inline int num_devices() { return _properties.size(); }

  /// Set the CUDA device to the specified device number
  void set(int num);

  /// Get the current device number
  inline int device_num() { return _device; }

  /// Returns the default stream for the current device
  inline command_queue & cq() { return cq(0); }
  
  /// Returns the stream indexed by i
  inline command_queue & cq(const int i) { return _cq[i]; }

  /// Block until all commands in the default stream have completed
  inline void sync() { sync(0); }
  
  /// Block until all commands in the specified stream have completed
  inline void sync(const int i) { ucl_sync(cq(i)); }
  
  /// Get the number of command queues currently available on device
  inline int num_queues() 
    { if (_device==-1) return 0; else return _cq.size(); }
  
  /// Add a stream for device computations
  inline void push_command_queue() {
    _cq.push_back(cudaStream_t()); 
    CUDA_SAFE_CALL_NS(cudaStreamCreate(&_cq.back())); 
  }

  /// Remove a stream for device computations
  /** \note You cannot delete the default stream **/
  inline void pop_command_queue() {
    if (_cq.size()<2) return;
    CUDA_SAFE_CALL_NS(cudaStreamDestroy(_cq.back()));
    _cq.pop_back();
  }
  
  /// Get the current CUDA device name
  inline std::string name() { return name(_device); }
  /// Get the CUDA device name
  inline std::string name(const int i) 
    { return std::string(_properties[i].name); }

  /// Get a string telling the type of the current device
  inline std::string device_type_name() { return device_type_name(_device); }
  /// Get a string telling the type of the device
  inline std::string device_type_name(const int i) { return "GPU"; }

  /// Get current device type (UCL_CPU, UCL_GPU, UCL_ACCELERATOR, UCL_DEFAULT)
  inline int device_type() { return device_type(_device); }
  /// Get device type (UCL_CPU, UCL_GPU, UCL_ACCELERATOR, UCL_DEFAULT)
  inline int device_type(const int i) { return UCL_GPU; }
  
  /// Returns true if double precision is support for the current device
  bool double_precision() { return double_precision(_device); }
  /// Returns true if double precision is support for the device
  bool double_precision(const int i) {return compute_cap(i)>=1.3;}
  
  /// Get the number of cores in the current device
  inline unsigned cores() { return cores(_device); }
  /// Get the number of cores
  inline unsigned cores(const int i) 
    { return _properties[i].multiProcessorCount*8; }
  
  /// Get the gigabytes of global memory in the current device
  inline double gigabytes() { return gigabytes(_device); }
  /// Get the gigabytes of global memory
  inline double gigabytes(const int i) 
    { return static_cast<double>(_properties[i].totalGlobalMem)/1073741824; }
  
  /// Get the bytes of global memory in the current device
  inline size_t bytes() { return bytes(_device); }
  /// Get the bytes of global memory
  inline size_t bytes(const int i) { return _properties[i].totalGlobalMem; }

  /// Return the GPGPU compute capability for current device
  inline double compute_cap() { return compute_cap(_device); }
  /// Return the GPGPU compute capability
  inline double compute_cap(const int i) 
    { return static_cast<double>(_properties[i].minor)/10+_properties[i].major;}
  
  /// Clock rate in GHz for current device
  inline double clock_rate() { return clock_rate(_device); }
  /// Clock rate in GHz
  inline double clock_rate(const int i) { return _properties[i].clockRate*1e-6;}
               
  /// Get the maximum number of threads per block
  inline size_t group_size() { return group_size(_device); }
  /// Get the maximum number of threads per block
  inline size_t group_size(const int i) 
    { return _properties[i].maxThreadsPerBlock; }
  
  /// List all devices along with all properties
  void print_all(std::ostream &out);
 
 private:
  int _device, _num_devices;
  std::vector<cudaDeviceProp> _properties;
  std::vector<cudaStream_t> _cq;
};

// Grabs the properties for all devices
inline UCL_Device::UCL_Device() {
  CUDA_SAFE_CALL_NS(cudaGetDeviceCount(&_num_devices));
  for (int dev=0; dev<_num_devices; ++dev) {
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL_NS(cudaGetDeviceProperties(&deviceProp, dev));
    if (deviceProp.major == 9999 && deviceProp.minor == 9999)
      break;
    _properties.push_back(deviceProp);
  }
  _device=-1;
  _cq.push_back(cudaStream_t());
  _cq.back()=0;
}

inline UCL_Device::~UCL_Device() {
  for (int i=1; i<num_queues(); i++) pop_command_queue();
}

// Set the CUDA device to the specified device number
inline void UCL_Device::set(int num) {
  if (_device==num)
    return;
  for (int i=1; i<num_queues(); i++) pop_command_queue();
  cudaThreadExit();
  CUDA_SAFE_CALL_NS(cudaSetDevice(num));
  _device=num;
}

// List all devices along with all properties
inline void UCL_Device::print_all(std::ostream &out) {
  if (num_devices() == 0)
    out << "There is no device supporting CUDA\n";
  for (int i=0; i<num_devices(); ++i) {
    out << "\nDevice " << i << ": \"" << name(i).c_str() << "\"\n";
    out << "  Type of device:                                "
        << device_type_name(i).c_str() << std::endl;
    out << "  Compute capability:                            "
        << compute_cap(i) << std::endl;
    out << "  Double precision support:                      ";
    if (double_precision(i))
      out << "Yes\n";
    else
      out << "No\n";
    out << "  Total amount of global memory:                 "
        << gigabytes(i) << " GB\n";
    #if CUDART_VERSION >= 2000
    out << "  Number of compute units/multiprocessors:       "
        << _properties[i].multiProcessorCount << std::endl;
    out << "  Number of cores:                               "
        << cores(i) << std::endl;
    #endif
    out << "  Total amount of constant memory:               "
        << _properties[i].totalConstMem << " bytes\n";
    out << "  Total amount of local/shared memory per block: "
        << _properties[i].sharedMemPerBlock << " bytes\n";
    out << "  Total number of registers available per block: "
        << _properties[i].regsPerBlock << std::endl;
    out << "  Warp size:                                     "
        << _properties[i].warpSize << std::endl;
    out << "  Maximum number of threads per block:           "
        << _properties[i].maxThreadsPerBlock << std::endl;
    out << "  Maximum group size (# of threads per block)    "
        << _properties[i].maxThreadsDim[0] << " x "
        << _properties[i].maxThreadsDim[1] << " x "
        << _properties[i].maxThreadsDim[2] << std::endl;
    out << "  Maximum item sizes (# threads for each dim)    "
        << _properties[i].maxGridSize[0] << " x "
        << _properties[i].maxGridSize[1] << " x "
        << _properties[i].maxGridSize[2] << std::endl;
    out << "  Maximum memory pitch:                          "
        << _properties[i].memPitch << " bytes\n";
    out << "  Texture alignment:                             "
        << _properties[i].textureAlignment << " bytes\n";
    out << "  Clock rate:                                    "
        << clock_rate(i) << " GHz\n";
    #if CUDART_VERSION >= 2000
    out << "  Concurrent copy and execution:                 ";
    if (_properties[i].deviceOverlap)
      out << "Yes\n";
    else
      out << "No\n";
    #endif
  }
}

}

#endif

