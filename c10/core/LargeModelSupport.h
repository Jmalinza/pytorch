#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/IntrusiveList.h>

#include <atomic>
#include <cstring>

namespace c10 {

typedef void* LMSSyncEvent_t;

struct LMSImpl {
  LMSImpl(Allocator* allocator) : allocator_(allocator), reclaimed_(false), pincount_(0) {}
  LMSImpl() = delete;
  virtual ~LMSImpl() {}

  virtual void release_resources() {
    data_ptr_.clear();
    reclaimed_ = false;
  }

  virtual void reclaim_list_add(IntrusiveListHook* list_hook) = 0;
  virtual bool reclaim_list_remove(IntrusiveListHook* list_hook) = 0;

  bool reclaimed() const {
    return reclaimed_;
  };

  bool pin() {
    bool initial = (++pincount_ == 1);
    return initial;
  }

  bool unpin() {
    bool final = (--pincount_ == 0);
    if (final && reclaimed_)
      pagein_sync();
    return final;
  }

  void pagein(void* dst, size_t size) {
    AT_ASSERT(reclaimed_ == true);
    void* src = data_ptr_.get();
    AT_ASSERT(dst);
    AT_ASSERT(src);

    do_pagein(dst, src, size);
  }

  void pagein_sync() {
    do_pagein_sync();
    reclaimed_ = false;
  }

  void pageout(void* src, size_t size, LMSSyncEvent_t sync_event) {
    AT_ASSERT(reclaimed_ == false);

    void* dst = data_ptr_.get();
    if (!dst) {
      data_ptr_ = allocator_->allocate(size);
      dst = data_ptr_.get();
    }
    AT_ASSERT(src);
    AT_ASSERT(dst);

    do_pageout(dst, src, size, sync_event);
  }

  void pageout_sync() {
    do_pageout_sync();
    reclaimed_ = true;
  }

  void copy_reclaimed_data(void* dst, size_t size) const {
    AT_ASSERT(reclaimed_ == true);
    memcpy(dst, data_ptr_.get(), size);
  }

protected:
  virtual void do_pagein(void* dst, void* src, size_t size) = 0;
  virtual void do_pagein_sync() = 0;
  virtual void do_pageout(void* dst, void* src, size_t size, LMSSyncEvent_t sync_event) = 0;
  virtual void do_pageout_sync() = 0;

  Allocator* allocator_ = nullptr;
  bool reclaimed_ = false;
  DataPtr data_ptr_;
  mutable std::atomic<size_t> pincount_;
};


struct LMS {
  LMS(LMSImpl* lms) { set(lms); }
  LMS() = delete;
  ~LMS() { unset(); }

  LMS& operator=(LMS&& other) = default;
  LMS(LMS&& other) = default;

  static LMS* from_list_hook(IntrusiveListHook *hook);

  bool enabled() const {
    return lms_ != nullptr;
  };

  void set(LMSImpl* lms) {
    AT_ASSERT(lms_ == nullptr);
    lms_ = lms;
  }

  void unset() {
    if (enabled()) {
      reclaim_list_remove();
      delete lms_;
      lms_ = nullptr;
    }
  }

  void release_resources() {
    if (enabled()) {
      reclaim_list_remove();
      lms_->release_resources();
    }
  }

  bool reclaimed() const {
    return enabled() && lms_->reclaimed();
  };

  void list_add(IntrusiveList* list) {
    list->append(&list_hook_);
  }

  bool list_remove() {
    return list_hook_.remove();
  }

  bool pin() {
    bool initial = enabled() && lms_->pin();
    if (initial)
      reclaim_list_remove();
    return initial;
  }

  bool unpin() {
    bool final = enabled() && lms_->unpin();
    if (final)
      reclaim_list_add();
    return final;
  }

  void pagein(void* data_ptr, size_t size) const {
    lms_->pagein(data_ptr, size);
  }

  void pagein_sync() const {
    lms_->pagein_sync();
  }

  void pageout(void* data_ptr, size_t size, LMSSyncEvent_t sync_event, IntrusiveList *async_queue = nullptr) {
    lms_->pageout(data_ptr, size, sync_event);
    if (async_queue)
      list_add(async_queue);
  }

  void pageout_sync(IntrusiveList *async_queue = nullptr) {
    if (async_queue)
      list_remove();
    lms_->pageout_sync();
  }

  void copy_reclaimed_data(void* dst, size_t size) const {
    lms_->copy_reclaimed_data(dst, size);
  }

  void reclaim_list_add() {
    lms_->reclaim_list_add(&list_hook_);
  }

  bool reclaim_list_remove() {
    if (!list_hook_.attached()) return false;

    return lms_->reclaim_list_remove(&list_hook_);
  }

 private:
  IntrusiveListHook list_hook_;
  LMSImpl* lms_ = nullptr;
};
} // namespace c10
