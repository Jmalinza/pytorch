#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC

#include <c10/cuda/CUDAStream.h>
#include <c10/core/Allocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Registry.h>

#include <mutex>

namespace c10 {

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class C10_CUDA_API FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() {};
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeCudaMemoryCallbacksRegistry, name, __VA_ARGS__);

namespace cuda {

// TODO: Turn this into an honest to goodness class. I briefly attempted to do
// this, but it was a bit irritating to figure out how to also correctly
// apply pimpl pattern so I didn't have to leak any internal implementation
// details in the header (CUDACachingAllocator could be made a pimpl, but
// you also need to appropriately define a class which is a subclass
// of Allocator. Not impossible, but required a bit more surgery than
// I wanted to do at the time.)
//
// Why is this using a namespace rather than old-style THCCachingAllocator_
// prefix?  Mostly because it made the HIPify rules easier to write; _ is
// not counted as a word boundary, so you would otherwise have to list each
// of these functions.

namespace CUDACachingAllocator {

C10_CUDA_API void* raw_alloc(size_t nbytes);
C10_CUDA_API void raw_delete(void* ptr);

C10_CUDA_API Allocator* get();
C10_CUDA_API void init(int device_count, at::Allocator* host_allocator);
C10_CUDA_API void emptyCache();
C10_CUDA_API void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock);
C10_CUDA_API void* getBaseAllocation(void *ptr, size_t *size);
C10_CUDA_API void recordStream(void *ptr, CUDAStream stream);
C10_CUDA_API uint64_t currentMemoryAllocated(int device);
C10_CUDA_API uint64_t maxMemoryAllocated(int device);
C10_CUDA_API void     resetMaxMemoryAllocated(int device);
C10_CUDA_API uint64_t currentMemoryCached(int device);
C10_CUDA_API uint64_t maxMemoryCached(int device);
C10_CUDA_API void     resetMaxMemoryCached(int device);
C10_CUDA_API uint64_t currentMemoryActive(int device);
C10_CUDA_API uint64_t maxMemoryActive(int device);
C10_CUDA_API void     resetMaxMemoryActive(int device);
C10_CUDA_API uint64_t currentMemoryReclaimed(int device);
C10_CUDA_API void     resetMemoryReclaimed(int device);
C10_CUDA_API void   setUserEnabledLMS(bool enable);
C10_CUDA_API bool   userEnabledLMS(void);
C10_CUDA_API void   setUserSizeLMS(size_t size);
C10_CUDA_API size_t userSizeLMS(void);
C10_CUDA_API void   setUserLimitLMS(size_t limit);
C10_CUDA_API size_t userLimitLMS(void);
C10_CUDA_API void reclaimInactive();

C10_CUDA_API std::mutex* getFreeMutex();

C10_CUDA_API std::shared_ptr<void> getIpcDevPtr(std::string handle);

enum AllocSource {
  FREELIST,
  CUDAMALLOC_UNDER_LIMIT,
  RECLAIM_ONE,
  RECLAIM_FRAGMENTS,
  CUDAMALLOC_OVER_LIMIT,
  RECLAIM_ALL,
  CUDAMALLOC_PURGE,
  NUM_ALLOC_SOURCES
};

C10_CUDA_API void currentAllocDistribution(int device, uint64_t* distribution);
C10_CUDA_API void resetAllocDistribution(int device);

} // namespace CUDACachingAllocator

}} // namespace c10::cuda

#endif
