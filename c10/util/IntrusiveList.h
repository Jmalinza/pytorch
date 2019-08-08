//===--- IntrusiveList.h - --------------------------------------*- C++ -*-===//

#pragma once

#include "c10/util/Exception.h"

namespace c10 {
  class IntrusiveListHook {
  public:
    IntrusiveListHook() {
      next_ = prev_ = this;
    }
    ~IntrusiveListHook() {
      remove();
    }

    IntrusiveListHook(IntrusiveListHook&& other) : IntrusiveListHook() {}
    IntrusiveListHook& operator=(IntrusiveListHook&& other) { return *this; }

    bool attached() const { return next_ != this; }
    bool detached() const { return next_ == this; }

    void insertbefore(IntrusiveListHook *x) {
      if (x->attached()) {
        AT_ERROR("Double insertion of IntrusiveListHook");
      }
      x->prev_ = prev_;
      x->next_ = this;
      prev_->next_ = x;
      prev_ = x;
    }

    bool remove() {
      if (!attached()) return false;

      prev_->next_ = next_;
      next_->prev_ = prev_;
      next_ = prev_ = this;
      return true;
    }
    IntrusiveListHook *next() const { return next_; }
    IntrusiveListHook *prev() const { return prev_; }

  private:
    IntrusiveListHook *next_;
    IntrusiveListHook *prev_;
  };

  class IntrusiveList {
  public:
    IntrusiveList() {}
    ~IntrusiveList() {}
    bool empty() const { return anchor_.detached(); }
    void append(IntrusiveListHook *x) { anchor_.insertbefore(x); }
    void prepend(IntrusiveListHook *x) { anchor_.next()->insertbefore(x); }
    IntrusiveListHook *head() const { return anchor_.next(); }
    IntrusiveListHook *tail() const { return anchor_.prev(); }
    const IntrusiveListHook *terminator() const { return &anchor_; }

  private:
    IntrusiveListHook anchor_;
  };

} // end namespace c10
