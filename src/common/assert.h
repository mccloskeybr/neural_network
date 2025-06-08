#ifndef SRC_COMMON_ASSERT_H_
#define SRC_COMMON_ASSERT_H_

#include <cassert>

#ifdef DEBUG
#define ASSERT(exp) assert(exp)
#else
#define ASSERT(exp)
#endif

#define UNREACHABLE() assert(false)

#endif
