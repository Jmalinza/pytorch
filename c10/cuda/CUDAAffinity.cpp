#include <c10/cuda/CUDAAffinity.h>

#include <c10/cuda/CUDAException.h>

#include <cuda_runtime_api.h>
#include <mutex>
#include <nvml.h>
#include <dlfcn.h>

namespace c10 {
namespace cuda {
namespace CUDAAffinity {

// CPU/GPU Affinity

#define CUDA_AFFINITY_DEFAULT true

struct Affinity {
  Affinity(bool enabled) : enabled_(enabled), nvml_lib_(nullptr), nvml_idx_(nullptr) {}

  ~Affinity() {
    if (nvml_lib_) {
      shutdown_();
      dlclose(nvml_lib_);
    }
    if (nvml_idx_) {
      delete[] nvml_idx_;
    }
  }

  void set_enabled(bool enabled) { enabled_ = enabled; }
  bool enabled(void) { return enabled_; }

  void init(int cudaCount) {
    if (nvml_idx_ != nullptr)
      return;

    void *handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (handle == nullptr)
      return;

    nvmlReturn_t (*init)(void);
    nvmlReturn_t (*deviceGetCount)(int *);
    nvmlReturn_t (*deviceGetHandleByIndex)(int, nvmlDevice_t *);
    nvmlReturn_t (*deviceGetPciInfo)(nvmlDevice_t, nvmlPciInfo_t *);
    nvmlReturn_t (*deviceSetCpuAffinity)(nvmlDevice_t);
    nvmlReturn_t (*shutdown)(void);
    init = (nvmlReturn_t (*)(void))dlsym(handle, "nvmlInit_v2");
    deviceGetCount = (nvmlReturn_t (*)(int *))dlsym(handle, "nvmlDeviceGetCount_v2");
    deviceGetHandleByIndex = (nvmlReturn_t (*)(int, nvmlDevice_t *))dlsym(handle, "nvmlDeviceGetHandleByIndex_v2");
    deviceGetPciInfo = (nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t *))dlsym(handle, "nvmlDeviceGetPciInfo_v2");
    deviceSetCpuAffinity = (nvmlReturn_t (*)(nvmlDevice_t))dlsym(handle, "nvmlDeviceSetCpuAffinity");
    shutdown = (nvmlReturn_t (*)(void))dlsym(handle, "nvmlShutdown");

    if (!(init && shutdown && deviceGetCount && deviceGetHandleByIndex &&
          deviceGetPciInfo && deviceSetCpuAffinity)) {
      dlclose(handle);
      return;
    }

    init();
    nvml_lib_ = handle;
    deviceSetCpuAffinity_ = deviceSetCpuAffinity;
    shutdown_ = shutdown;

    // Need to map cuda index to nvml index
    nvml_idx_ = new nvmlDevice_t[cudaCount];

    nvmlDevice_t id;
    nvmlPciInfo_t pciinfo;
    int count, i, j;
    cudaDeviceProp *device_prop = new cudaDeviceProp[cudaCount];
    for (j = 0; j < cudaCount; j++) {
      C10_CUDA_CHECK(cudaGetDeviceProperties(&device_prop[j], j));
    }
    deviceGetCount(&count);
    for (i = 0; i < count; i++) {
      deviceGetHandleByIndex(i, &id);
      deviceGetPciInfo(id, &pciinfo);
      for (j = 0; j < cudaCount; j++) {
        cudaDeviceProp* p = &device_prop[j];
        if ((pciinfo.domain == (unsigned int)p->pciDomainID) &&
            (pciinfo.bus    == (unsigned int)p->pciBusID) &&
            (pciinfo.device == (unsigned int)p->pciDeviceID)) {
          nvml_idx_[j] = id;
          break;
        }
      }
    }
    delete[] device_prop;
  }

  void set_affinity(int device) {
    if (enabled_ && nvml_idx_ != nullptr) {
      std::lock_guard<std::mutex> lock(mutex_); // work around thread safety issues
      deviceSetCpuAffinity_(nvml_idx_[device]);
    }
  }

 private:
  bool enabled_;
  void *nvml_lib_;
  nvmlDevice_t* nvml_idx_;
  nvmlReturn_t (*deviceSetCpuAffinity_)(nvmlDevice_t);
  nvmlReturn_t (*shutdown_)(void);
  std::mutex mutex_;
};

Affinity affinity(CUDA_AFFINITY_DEFAULT);

void init(int device_count) {
  affinity.init(device_count);
}

void set_enabled(bool enabled) {
  affinity.set_enabled(enabled);
}

bool enabled(void) {
  return affinity.enabled();
}

void set_affinity(int device) {
  affinity.set_affinity(device);
}

} // namespace CUDAAffinity

}} // namespace c10::cuda
