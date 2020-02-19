#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/IntrusiveList.h>

#include <atomic>
#include <cstring>

namespace c10 {

struct StorageImpl;
struct LmsStorageImpl;

typedef void* LMSSyncEvent_t;
typedef IntrusiveList<LmsStorageImpl> LmsStorageList;
typedef IntrusiveListHook<LmsStorageImpl> LmsStorageListHook;

struct LmsStorageImpl {
  LmsStorageImpl(StorageImpl* storage, Allocator* host_allocator) :
    storage_(storage), host_allocator_(host_allocator), reclaimed_(false), pincount_(0), list_hook_(this) {}
  LmsStorageImpl() = delete;
  virtual ~LmsStorageImpl() {
    reclaim_list_remove_internal();
  }

  virtual void release_resources() {
    reclaim_list_remove_internal();
    host_data_ptr_.clear();
    reclaimed_ = false;
  }

  bool reclaimed() const {
    return reclaimed_;
  };

  bool pin() {
    bool initial = (++pincount_ == 1);
    if (initial) {
      ensure_data_internal();
    }
    return initial;
  }

  bool unpin() {
    bool final = (--pincount_ == 0);
    if (final) {
      if (reclaimed_) pagein_sync(); // in case data was never accessed
      reclaim_list_add();
    }
    return final;
  }

  void ensure_data() {
    ensure_data_internal();
    if (reclaimed_)
      pagein_sync();
  }

  void pageout(LMSSyncEvent_t sync_event, LmsStorageList* async_queue = nullptr) {
    AT_ASSERT(reclaimed_ == false);
    void* src = ptr();
    size_t size = capacity();
    void* dst = host_data_ptr_.get();
    if (!dst) {
      host_data_ptr_ = host_allocator_->allocate(size);
      dst = host_data_ptr_.get();
    }
    AT_ASSERT(src);
    AT_ASSERT(dst);

    do_pageout(dst, src, size, sync_event);

    if (async_queue)
      list_add(async_queue);
  }

  void pageout_sync(LmsStorageList* async_queue = nullptr) {
    if (async_queue)
      list_remove();
    do_pageout_sync();
    set_data_ptr(at::DataPtr(nullptr, device()));
    reclaimed_ = true;
  }

  void copy_reclaimed_data(void* dst, size_t size) const {
    AT_ASSERT(reclaimed_ == true);
    memcpy(dst, host_data_ptr_.get(), size);
  }

  void list_add(LmsStorageList* list) {
    list->append(&list_hook_);
  }

  bool list_remove() {
    return list_hook_.remove();
  }

  // StorageImpl accessors defined in StorageImpl.h to avoid circular depencencies
  const Allocator* allocator() const;
  size_t capacity() const;
  Device device() const;
  void* ptr() const;
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr);

protected:
  virtual void reclaim_list_add() = 0;
  virtual bool reclaim_list_remove() = 0;
  virtual void do_pagein(void* dst, void* src, size_t size) = 0;
  virtual void do_pagein_sync() = 0;
  virtual void do_pageout(void* dst, void* src, size_t size, LMSSyncEvent_t sync_event) = 0;
  virtual void do_pageout_sync() = 0;

  void ensure_data_internal() {
    if (reclaim_list_remove_internal() || !reclaimed_)
      return;

    if (!ptr())
      pagein();
  }

  bool reclaim_list_remove_internal() {
    if (!list_hook_.attached()) return false;
    return reclaim_list_remove();
  }

  void pagein() {
    AT_ASSERT(reclaimed_);
    AT_ASSERT(!ptr());
    size_t size = capacity();
    auto dst = allocator()->allocate(size);
    do_pagein(dst.get(), host_data_ptr_.get(), size);
    set_data_ptr(std::move(dst));
  }

  void pagein_sync() {
    do_pagein_sync();
    host_data_ptr_.clear();
    reclaimed_ = false;
  }

  StorageImpl* const storage_;
  Allocator* const host_allocator_;
  bool reclaimed_;
  DataPtr host_data_ptr_;
  mutable std::atomic<size_t> pincount_;
  LmsStorageListHook list_hook_;
};
} // namespace c10
