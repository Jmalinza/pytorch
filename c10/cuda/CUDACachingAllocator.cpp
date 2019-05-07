#include <c10/cuda/CUDACachingAllocator.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/UniqueVoidPtr.h>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace c10 {

C10_DEFINE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);

namespace cuda {
namespace CUDACachingAllocator {

//
// Yet another caching allocator for CUDA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to cudaMalloc.
// - If the cudaMalloc fails, the allocator will free all cached blocks that
//   are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using cudaMalloc.
//   To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//



namespace {

using stream_set = std::unordered_set<cuda::CUDAStream>;

constexpr size_t kMinBlockSize = 512;       // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;      // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;    // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;   // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;     // round up large allocs to 2 MiB

struct DeviceStats {
  uint64_t   amount_allocated;      // total amount allocated in bytes
  uint64_t   max_amount_allocated;  // max total amount allocated in bytes
  uint64_t   amount_cached;         // total amount in cache in bytes
  uint64_t   max_amount_cached;     // max total amount in cache in bytes
  uint64_t   amount_inactive;       // total amount in reclaim list in bytes
  uint64_t   amount_active()        { return amount_allocated - amount_inactive; }
  uint64_t   max_amount_active;     // max total active in bytes

  DeviceStats() :
      amount_allocated(0), max_amount_allocated(0),
      amount_cached(0), max_amount_cached(0),
      amount_inactive(0), max_amount_active(0) { }

  void increaseAllocated(size_t delta) {
    amount_allocated += delta;
    max_amount_allocated = std::max(max_amount_allocated, amount_allocated);
    max_amount_active = std::max(max_amount_active, amount_active());
  }

  void decreaseAllocated(size_t delta) {
    amount_allocated -= delta;
  }

  void increaseCached(size_t delta) {
    amount_cached += delta;
    max_amount_cached = std::max(max_amount_cached, amount_cached);
  }

  void decreaseCached(size_t delta) {
    amount_cached -= delta;
  }

  void increaseInactive(size_t delta) {
    amount_inactive += delta;
  }

  void decreaseInactive(size_t delta) {
    amount_inactive -= delta;
    max_amount_active = std::max(max_amount_active, amount_active());
  }
};

struct Block;
typedef bool (*Comparison)(const Block*, const Block*);
typedef std::set<Block*, Comparison> BlockPool;

struct Block {
  int           device;      // gpu
  cudaStream_t  stream;      // allocation stream
  stream_set    stream_uses; // streams on which the block was used
  size_t        size;        // block size in bytes
  BlockPool*    pool;        // owning memory pool
  void*         ptr;         // memory address
  bool          allocated;   // in-use flag
  Block*        prev;        // prev block if split from a larger allocation
  Block*        next;        // next block if split from a larger allocation
  int           event_count; // number of outstanding CUDA events

  Block(int device, cudaStream_t stream, size_t size, BlockPool* pool, void* ptr) :
    device(device), stream(stream), stream_uses(), size(size), pool(pool),
    ptr(ptr), allocated(0), prev(nullptr), next(nullptr), event_count(0) { }

  // constructor for search key
  Block(int device, cudaStream_t stream, size_t size) :
    device(device), stream(stream), stream_uses(), size(size), pool(nullptr),
    ptr(nullptr), allocated(0), prev(nullptr), next(nullptr), event_count(0) { }
};

static bool BlockComparator(const Block* a, const Block* b)
{
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

#define LMS_SIZE_DEFAULT (1 << 20) // 1 MB

struct LMSSettings {
  LMSSettings() :
    enabled_(false), size_(LMS_SIZE_DEFAULT), limit_(0), host_allocator_(nullptr) {}

  bool enabled()                 { return enabled_; }
  void set_enabled(bool enabled) { enabled_ = enabled; }
  size_t size()                  { return size_; }
  void set_size(size_t size)     { size_ = size; }
  size_t limit()                 { return limit_; }
  void set_limit(size_t limit)   { limit_ = limit; }
  at::Allocator* host_allocator()                        { return host_allocator_; }
  void set_host_allocator(at::Allocator* host_allocator) { host_allocator_ = host_allocator; }

  bool enabled(size_t size) {
    return enabled_ && size >= size_;
  }
  bool limit_alloc(DeviceStats& stats, size_t alloc_size) {
    return (stats.amount_cached + alloc_size) > limit_;
  }

private:
  bool enabled_;
  size_t size_;
  size_t limit_;
  at::Allocator* host_allocator_;
};

struct AllocParams {
  AllocParams(int device, size_t size, cudaStream_t stream, BlockPool* pool, size_t alloc_size,
              LMSSettings* lms, DeviceStats& stats) :
    search_key(device, stream, size),
    pool(pool),
    alloc_size(alloc_size),
    lms_enabled(lms->enabled(size)),
    limit_alloc(lms_enabled && lms->limit_alloc(stats, alloc_size)),
    block(nullptr),
    err(cudaSuccess) {}

  int device() { return search_key.device; }
  cudaStream_t stream() { return search_key.stream; }
  size_t size() { return search_key.size; }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  bool lms_enabled;
  bool limit_alloc;
  Block* block;
  cudaError_t err;
};

} // namespace

struct DeviceCachingAllocator
{
  // device statistics
  DeviceStats stats;

  // lock around all operations
  std::recursive_mutex mutex;

  // cached blocks larger than 1 MB
  BlockPool large_blocks;

  // cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // outstanding cuda events
  std::deque<std::pair<cudaEvent_t, Block*>> cuda_events;

  at::IntrusiveList reclaim_list;

  DeviceCachingAllocator() :
      large_blocks(BlockComparator),
      small_blocks(BlockComparator) {}

  bool get_free_block(AllocParams& p)
  {
    BlockPool& pool = *p.pool;
    auto it = pool.lower_bound(&p.search_key);
    if (it == pool.end() || (*it)->stream != p.stream())
      return false;
    p.block = *it;
    pool.erase(it);
    return true;
  }

  bool trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
        FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  bool alloc_block(AllocParams& p, bool record_error)
  {
    size_t size = p.alloc_size;
    void* ptr;
    cudaError_t err;
    err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
      if (record_error) p.err = err; else cudaGetLastError();
      return false;
    }

    stats.increaseCached(size);
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
    return (p.block != nullptr);
  }

  bool try_lms_reclaim(AllocParams& p) {
    size_t size = p.size();
    cudaStream_t stream = p.stream();
    cudaEvent_t sync_event;

    AT_ASSERT(stream == cuda::getCurrentCUDAStream().stream());
    C10_CUDA_CHECK(cudaEventCreate(&sync_event));
    C10_CUDA_CHECK(cudaEventRecord(sync_event, stream));

    bool found =
      // a. Search reclaim list for a suitable inactive allocation
      (reclaim_one(size, sync_event) && get_free_block(p))
      // b. Reclaim fragments of suitable allocations
      || (reclaim_fragments(size, sync_event) && get_free_block(p))
      // c. Attempt allocate (if not done earlier due to limit)
      || (p.limit_alloc && alloc_block(p, false))
      // d. Reclaim everything else
      || (reclaim_all(sync_event) && get_free_block(p));

    C10_CUDA_CHECK(cudaEventDestroy(sync_event));

    return found;
  }

  /** allocates a block which is safe to use from the provided stream */
  Block* malloc(int device, size_t size, cudaStream_t stream, LMSSettings* lms)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    // process outstanding cudaEvents
    process_events();

    size = round_size(size);
    auto& pool = get_pool(size);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, lms, stats);

    bool block_found =
      // 1. Search pool
      get_free_block(params)
      // 2. Trigger callbacks and retry search
      || (trigger_free_memory_callbacks(params) && get_free_block(params))
      // 3. Attempt allocate (if not limited by lms settings)
      || (!params.limit_alloc && alloc_block(params, false))
      // 4. If LMS enabled, try to reclaim inactive allocations
      || (params.lms_enabled && try_lms_reclaim(params))
      // 5. Free all non-split cached blocks and retry alloc.
      || (free_cached_blocks() && alloc_block(params, true));

    AT_ASSERT((!block_found && params.err != cudaSuccess) || params.block);
    if (!block_found) {
      if (params.err == cudaErrorMemoryAllocation) {
        cudaGetLastError();  // clear CUDA error

        size_t device_free;
        size_t device_total;
        C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));

        // "total capacity": total global memory on GPU
        // "already allocated": memory allocated by the program using the
        //                      caching allocator
        // "free": free memory as reported by the CUDA API
        // "cached": memory held by the allocator but not used by the program
        //
        // The "allocated" amount  does not include memory allocated outside
        // of the caching allocator, such as memory allocated by other programs
        // or memory held by the driver.
        //
        // The sum of "allocated" + "free" + "cached" may be less than the
        // total capacity due to memory held by the driver and usage by other
        // programs.
        //
        // Note that at this point cuda_malloc_retry has already returned all
        // possible "cached" memory to the driver. The only remaining "cached"
        // memory is split from a larger block that is partially in-use.
        AT_ERROR(
          "CUDA out of memory. Tried to allocate ", format_size(alloc_size),
          " (GPU ", device, "; ",
          format_size(device_total), " total capacity; ",
          format_size(stats.amount_allocated), " already allocated; ",
          format_size(device_free), " free; ",
          format_size(stats.amount_cached - stats.amount_allocated), " cached; ",
          format_size(stats.amount_inactive), " inactive)");
      } else {
        C10_CUDA_CHECK(params.err);
      }
    }

    Block* block = params.block;
    Block* remaining = nullptr;
    AT_ASSERT(block);
    if (should_split(block, size)) {

      remaining = block;

      block = new Block(device, stream, size, &pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      pool.insert(remaining);
    }

    block->allocated = true;

    stats.increaseAllocated(block->size);

    return block;
  }

  void free(Block* block)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    block->allocated = false;

    stats.decreaseAllocated(block->size);
    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block);
    }
  }

  /** returns cached blocks to the system allocator */
  void emptyCache()
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    free_cached_blocks();
  }

  void* getBaseAllocation(Block* block, size_t* outSize)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    while (block->prev) {
      block = block->prev;
    }
    void *basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  // Accumulates sizes of all memory blocks for given device in given pool
  void cacheInfoAux(BlockPool& blocks, size_t* total, size_t* largest)
  {
    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
      size_t blocksize = (*it)->size;
      *total += blocksize;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

  void cacheInfo(size_t* total, size_t* largest)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    cacheInfoAux(large_blocks, total, largest);
    cacheInfoAux(small_blocks, total, largest);
  }

  void recordStream(Block* block, cuda::CUDAStream stream)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(Block* block)
  {
    AT_ASSERT(!block->allocated && block->event_count == 0);
    auto& pool = *block->pool;
    try_merge_blocks(block, block->prev, pool);
    try_merge_blocks(block, block->next, pool);
    pool.insert(block);
  }

  /** combine previously split blocks */
  void try_merge_blocks(Block* dst, Block* src, BlockPool& pool)
  {
    if (!src || src->allocated || src->event_count > 0) {
      return;
    }
    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    dst->size += src->size;
    pool.erase(src);
    delete src;
  }

  BlockPool& get_pool(size_t size) {
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  bool should_split(Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool == &small_blocks) {
      return remaining >= kMinBlockSize;
    } else if (block->pool == &large_blocks) {
      return remaining > kSmallSize;
    } else {
      AT_ERROR("should_split: invalid pool");
    }
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  bool free_cached_blocks()
  {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events();

    // Free all non-split cached blocks
    free_blocks(large_blocks);
    free_blocks(small_blocks);
    return true;
  }

  void free_blocks(BlockPool& blocks)
  {
    // Frees all non-split blocks
    std::lock_guard<std::mutex> lock(*CUDACachingAllocator::getFreeMutex());
    auto it = blocks.begin();
    while (it != blocks.end()) {
      Block* block = *it;
      if (!block->prev && !block->next) {
        C10_CUDA_CHECK(cudaFree((void*)block->ptr));
        stats.decreaseCached(block->size);
        auto cur = it;
        ++it;
        blocks.erase(cur);
        delete block;
      } else {
        ++it;
      }
    }
  }

  void synchronize_and_free_events() {
    // Synchronize on outstanding events and then free associated blocks.

    for (auto& e : cuda_events) {
      cudaEvent_t event = e.first;
      Block* block = e.second;

      C10_CUDA_CHECK(cudaEventSynchronize(event));
      C10_CUDA_CHECK(cudaEventDestroy(event));

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
    }

    cuda_events.clear();
  }

  void insert_events(Block* block)
  {
    int prev_device;
    C10_CUDA_CHECK(cudaGetDevice(&prev_device));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto it = streams.begin(); it != streams.end(); ++it) {
      C10_CUDA_CHECK(cudaSetDevice(it->device_index()));

      cudaEvent_t event;
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventRecord(event, it->stream()));

      block->event_count++;
      cuda_events.emplace_back(event, block);
    }

    C10_CUDA_CHECK(cudaSetDevice(prev_device));
  }

  void process_events()
  {
    // Process outstanding cudaEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!cuda_events.empty()) {
      auto& e = cuda_events.front();
      cudaEvent_t event = e.first;
      Block* block = e.second;

      cudaError_t err = cudaEventQuery(event);
      if (err == cudaErrorNotReady) {
        // ignore and clear the error if not ready
        cudaGetLastError();
        break;
      } else if (err != cudaSuccess) {
        C10_CUDA_CHECK(err);
      }

      C10_CUDA_CHECK(cudaEventDestroy(event));

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
      cuda_events.pop_front();
    }
  }

  void reclaim_list_add(StorageImpl* storage) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    size_t storage_size = round_size(storage->capacity());
    stats.increaseInactive(storage_size);
    storage->lms_list_add(&reclaim_list);
  }

  bool reclaim_list_remove(StorageImpl* storage) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (!storage->lms_list_remove())
      return false;

    size_t storage_size = round_size(storage->capacity());
    stats.decreaseInactive(storage_size);
    return true;
  }

  bool reclaim_one(size_t size, cudaEvent_t sync_event) {
    StorageImpl *best = nullptr;
    size_t best_size = ULONG_MAX;

    if (!reclaim_list.empty()) {
      auto hook = reclaim_list.head();
      auto end = reclaim_list.terminator();
      do {
        StorageImpl *storage = at::StorageImpl::from_list_hook(hook);
        hook = hook->next();

        size_t storage_size = round_size(storage->capacity());
        if (storage_size >= size && storage_size < best_size) {
          best = storage;
          best_size = storage_size;
          if (storage_size == size)
            break;
        }
      } while (hook != end);
    }

    if (best == nullptr)
      return false;

    stats.decreaseInactive(best_size);
    best->lms_list_remove();
    best->lms_pageout(sync_event);
    best->lms_pageout_sync();
    return true;
  }

  static inline void process_pageout_sync(at::IntrusiveList* iodone_queue) {
    while (!iodone_queue->empty()) {
      auto hook = iodone_queue->head();
      StorageImpl *storage = at::StorageImpl::from_list_hook(hook);
      storage->lms_pageout_sync(iodone_queue);
    }
  }

  bool reclaim_fragments(size_t size, cudaEvent_t sync_event) {
    at::IntrusiveList iodone_queue;
    size_t alloc_size;
    int count = 0;

    if (!reclaim_list.empty()) {
      auto hook = reclaim_list.head();
      auto end = reclaim_list.terminator();
      do {
        StorageImpl *storage = at::StorageImpl::from_list_hook(hook);
        hook = hook->next();

        CUDACachingAllocator::getBaseAllocation(storage->allocation_ptr(), &alloc_size);
        if (alloc_size >= size) {
          size_t storage_size = round_size(storage->capacity());
          stats.decreaseInactive(storage_size);
          storage->lms_list_remove();
          storage->lms_pageout(sync_event, &iodone_queue);
          count++;
        }
      } while (hook != end);
    }

    if (count == 0)
      return false;

    process_pageout_sync(&iodone_queue);
    return true;
  }

  bool reclaim_all(cudaEvent_t sync_event) {
    at::IntrusiveList iodone_queue;
    int count = 0;

    if (!reclaim_list.empty()) {
      auto hook = reclaim_list.head();
      auto end = reclaim_list.terminator();
      do {
        StorageImpl *storage = at::StorageImpl::from_list_hook(hook);
        hook = hook->next();

        size_t storage_size = round_size(storage->capacity());
        stats.decreaseInactive(storage_size);
        storage->lms_list_remove();
        storage->lms_pageout(sync_event, &iodone_queue);
        count++;
      } while (hook != end);
    }

    if (count == 0)
      return false;

    process_pageout_sync(&iodone_queue);
    return true;
  }

  void reclaimInactive()
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    if (!reclaim_list.empty()) {
      cudaStream_t stream = cuda::getCurrentCUDAStream().stream();
      cudaEvent_t sync_event;

      C10_CUDA_CHECK(cudaEventCreate(&sync_event));
      C10_CUDA_CHECK(cudaEventRecord(sync_event, stream));
      reclaim_all(sync_event);
      C10_CUDA_CHECK(cudaEventDestroy(sync_event));
    }
  }
};

struct THCCachingAllocator {
  std::mutex mutex;
  std::vector<DeviceCachingAllocator*> device_allocator;

  // allocated blocks by device pointer
  std::unordered_map<void*, Block*> allocated_blocks;

  // lock around calls to cudaFree (to prevent deadlocks with NCCL)
  std::mutex cuda_free_mutex;

  LMSSettings lms_settings;

  void init(int device_count, at::Allocator* host_allocator) {
    int size = device_allocator.size();
    if (size < device_count) {
      device_allocator.resize(device_count);
      for (int i = size; i < device_count; i++) {
        device_allocator[i] = new DeviceCachingAllocator();
      }
    }
    lms_settings.set_host_allocator(host_allocator);
  }

  void malloc(void** devPtr, size_t size, cudaStream_t stream) {
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    Block* block = device_allocator[device]->malloc(device, size, stream, &lms_settings);
    {
      std::lock_guard<std::mutex> lock(mutex);
      allocated_blocks[block->ptr] = block;
    }
    *devPtr = (void*)block->ptr;
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = allocated_blocks.find(ptr);
      if (it == allocated_blocks.end()) {
        AT_ERROR("invalid device pointer: ", ptr);
      }
      block = it->second;
      allocated_blocks.erase(it);
    }
    device_allocator[block->device]->free(block);
  }

  void emptyCache() {
    int count = device_allocator.size();
    for (int i = 0; i < count; i++)
      device_allocator[i]->emptyCache();
  }

  Block* find_allocated_block(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    return it->second;
  }

  void* getBaseAllocation(void* ptr, size_t* outSize)
  {
    Block* block = find_allocated_block(ptr);
    if (!block) {
      AT_ERROR("invalid device pointer: ", ptr);
    }
    return device_allocator[block->device]->getBaseAllocation(block, outSize);
  }

  void recordStream(void* ptr, cuda::CUDAStream stream)
  {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (ptr) {
      Block* block = find_allocated_block(ptr);
      if (!block) {
        AT_ERROR("invalid device pointer: ", ptr);
      }
      device_allocator[block->device]->recordStream(block, stream);
    }
  }

  void cacheInfo(int dev_id, size_t* total, size_t* largest) {
    device_allocator[dev_id]->cacheInfo(total, largest);
  }

  void reclaimInactive() {
    int count = device_allocator.size();
    for (int i = 0; i < count; i++)
      device_allocator[i]->reclaimInactive();
  }
};

THCCachingAllocator caching_allocator;


#define LMS_INVALID_STREAM ((cudaStream_t)-1)

struct CudaLMSImpl : public at::LMSImpl {
  CudaLMSImpl() :
    at::LMSImpl(caching_allocator.lms_settings.host_allocator()),
    stream_(LMS_INVALID_STREAM) {}
  ~CudaLMSImpl() {
    destroy_stream();
  }

  void release_resources() {
    at::LMSImpl::release_resources();
    destroy_stream();
  }

  void reclaim_list_add(at::IntrusiveListHook* hook) {
    at::StorageImpl* storage = at::StorageImpl::from_list_hook(hook);
    size_t size = storage->capacity();
    size_t storage_size = DeviceCachingAllocator::round_size(size);
    if (size == 0 || !caching_allocator.lms_settings.enabled(storage_size))
      return;
    int device = storage->device().index();
    caching_allocator.device_allocator[device]->reclaim_list_add(storage);
  }

  bool reclaim_list_remove(at::IntrusiveListHook* hook) {
    at::StorageImpl* storage = at::StorageImpl::from_list_hook(hook);
    int device = storage->device().index();
    return caching_allocator.device_allocator[device]->reclaim_list_remove(storage);
  }

 protected:
  cudaStream_t stream() const {
    AT_ASSERT(stream_ != LMS_INVALID_STREAM);
    return stream_;
  }

  void create_stream() {
    if (stream_ == LMS_INVALID_STREAM) {
      const unsigned int kFlags = cudaStreamDefault;
      C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, kFlags));
    }
  }

  void destroy_stream() {
    if (stream_ != LMS_INVALID_STREAM) {
      C10_CUDA_CHECK(cudaStreamDestroy(stream_));
      stream_ = LMS_INVALID_STREAM;
    }
  }

  void do_pagein(void* dst, void* src, size_t size) {
    C10_CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream()));
  }

  void do_pagein_sync() {
    C10_CUDA_CHECK(cudaStreamSynchronize(stream()));
  }

  void do_pageout(void* dst, void* src, size_t size, at::LMSSyncEvent_t sync_event) {
    create_stream();
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream(), (cudaEvent_t)sync_event, 0));
    C10_CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream()));
  }

  void do_pageout_sync() {
    C10_CUDA_CHECK(cudaStreamSynchronize(stream()));
  }

  cudaStream_t stream_;
};


static void CudaCachingDeleter(void* ptr) {
  caching_allocator.free(ptr);
}

// NB: I decided not to fold this into THCCachingAllocator, because the latter
// has a lot more methods and it wasn't altogether clear that they should
// actually be publically exposed
struct CudaCachingAllocator : public Allocator {
  DataPtr allocate(size_t size) const override {
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* r = nullptr;
    if (size != 0) {
      caching_allocator.malloc(&r, size, cuda::getCurrentCUDAStream(device));
    }
    return {r, r, &CudaCachingDeleter, Device(DeviceType::CUDA, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    return &CudaCachingDeleter;
  }
  at::LMSImpl* lms() const {
    return caching_allocator.lms_settings.enabled() ? new CudaLMSImpl() : nullptr;
  }
};

CudaCachingAllocator device_allocator;

Allocator* get(void)
{
  return &device_allocator;
}

void init(int device_count, at::Allocator* host_allocator) {
  caching_allocator.init(device_count, host_allocator);
}

void emptyCache(void) {
  caching_allocator.emptyCache();
}

void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
  caching_allocator.cacheInfo(dev_id, cachedAndFree, largestBlock);
}

void* getBaseAllocation(void *ptr, size_t *size)
{
  return caching_allocator.getBaseAllocation(ptr, size);
}

void recordStream(void *ptr, cuda::CUDAStream stream)
{
  caching_allocator.recordStream(ptr, stream);
}

std::mutex* getFreeMutex()
{
  return &caching_allocator.cuda_free_mutex;
}

static inline void assertValidDevice(int device) {
  int device_num = device_count();
  AT_ASSERTM(0 <= device && device < device_num, "Invalid device argument.");
}

uint64_t currentMemoryAllocated(int device)
{
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->stats.amount_allocated;
}

uint64_t maxMemoryAllocated(int device) {
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->stats.max_amount_allocated;
}

void resetMaxMemoryAllocated(int device) {
  assertValidDevice(device);
  DeviceStats& stats = caching_allocator.device_allocator[device]->stats;
  stats.max_amount_allocated = stats.amount_allocated;
}

uint64_t currentMemoryCached(int device)
{
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->stats.amount_cached;
}

uint64_t maxMemoryCached(int device) {
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->stats.max_amount_cached;
}

void resetMaxMemoryCached(int device) {
  assertValidDevice(device);
  DeviceStats& stats = caching_allocator.device_allocator[device]->stats;
  stats.max_amount_cached = stats.amount_cached;
}

uint64_t currentMemoryActive(int device)
{
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->stats.amount_active();
}

uint64_t maxMemoryActive(int device) {
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->stats.max_amount_active;
}

void resetMaxMemoryActive(int device) {
  assertValidDevice(device);
  DeviceStats& stats = caching_allocator.device_allocator[device]->stats;
  stats.max_amount_active = stats.amount_active();
}

void setUserEnabledLMS(bool enable) {
  caching_allocator.lms_settings.set_enabled(enable);
}

bool userEnabledLMS(void) {
  return caching_allocator.lms_settings.enabled();
}

void setUserSizeLMS(size_t size) {
  caching_allocator.lms_settings.set_size(size);
}

size_t userSizeLMS(void) {
  return caching_allocator.lms_settings.size();
}

void setUserLimitLMS(size_t limit) {
  caching_allocator.lms_settings.set_limit(limit);
}

size_t userLimitLMS(void) {
  return caching_allocator.lms_settings.limit();
}

void reclaimInactive(void) {
  caching_allocator.reclaimInactive();
}

//
// In CUDA IPC, sender sends a tensor to receiver, getIpcDevPtr
// is called by the receiving process to map the CUDA memory from the sending
// process into its own address space.
//
// CUDA IPC only allows sharing a big memory block associated with a cudaIpcMemHandle_t
// and it can be opened only **once** per context per process. There can be
// multiple types of storage in the same IPC mem block, so we must cache the
// device ptr to construct typed storage as it comes.
//
// ipcMemHandle_to_devptr maps a cudaIpcMemHandle_t to a device pointer in the process
// that can be used to access the memory block in the sender process.
// It only saves a weak_ptr of the device pointer in the map, the shared_ptr
// will be used to reconstruct all storages in this CudaMalloc allocation.
// And it will deleted in cudaIpcCloseMemHandle when its reference count is 0.
//
namespace {
  std::mutex IpcMutex;
  std::unordered_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr;
}

std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  std::lock_guard<std::mutex> lock(IpcMutex);

  auto iter = ipcMemHandle_to_devptr.find(handle);
  if (iter != ipcMemHandle_to_devptr.end()) {
    auto devptr = iter->second.lock();
    if (devptr) return devptr;
  }
  // This ipcMemHandle hasn't been opened, or already expired, open it to
  // enable IPC access to that mem block.
  void *dev = nullptr;
  auto ipc_handle = reinterpret_cast<const cudaIpcMemHandle_t*>(handle.c_str());
  C10_CUDA_CHECK(cudaIpcOpenMemHandle(&dev, *ipc_handle, cudaIpcMemLazyEnablePeerAccess));
  // devPtr has to be deleted in same device when created.
  int curr_device;
  C10_CUDA_CHECK(cudaGetDevice(&curr_device));
  auto sp = std::shared_ptr<void>(
      dev,
      [handle, curr_device](void *ptr) {
        cuda::CUDAGuard device_guard(curr_device);
        std::lock_guard<std::mutex> deleter_lock(IpcMutex);
        C10_CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
        ipcMemHandle_to_devptr.erase(handle);});
  std::weak_ptr<void> wp = sp;
  // To eliminate an additional search, we can use insert().
  // It doesn't overwrite when key already exists(ptr expired).
  // But in the deleter for sp we erased the entry,
  // this should be safe to do now.
  ipcMemHandle_to_devptr.insert(iter, {handle, wp});

  return sp;
}

void* raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(&r, nbytes, cuda::getCurrentCUDAStream(device));
  return r;
}

void raw_delete(void* ptr) {
  caching_allocator.free(ptr);
}

} // namespace CUDACachingAllocator

}} // namespace c10::cuda
