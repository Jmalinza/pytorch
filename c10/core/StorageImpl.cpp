#include <c10/core/StorageImpl.h>

namespace c10 {

c10::StorageImpl* c10::StorageImpl::from_list_hook(c10::IntrusiveListHook *hook) {
  return (StorageImpl *)((char *)c10::LMS::from_list_hook(hook) - offsetof(StorageImpl, lms_));
}

} // namespace at
