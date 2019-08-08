#include <c10/core/LargeModelSupport.h>

namespace c10 {

c10::LMS* c10::LMS::from_list_hook(c10::IntrusiveListHook *hook) {
  return (LMS *)((char *)hook - offsetof(LMS, list_hook_));
}

} // namespace at
