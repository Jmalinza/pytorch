#pragma once

#include <c10/cuda/CUDAMacros.h>

namespace c10 {
namespace cuda {

namespace CUDAAffinity {

C10_CUDA_API void init(int device_count);
C10_CUDA_API void set_enabled(bool enabled);
C10_CUDA_API bool enabled(void);
C10_CUDA_API void set_affinity(int device);

} // namespace CUDAAffinity

}} // namespace c10::cuda
