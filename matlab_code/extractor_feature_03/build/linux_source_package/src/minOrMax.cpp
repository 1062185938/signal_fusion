//
// minOrMax.cpp
//
// Code generation for function 'minOrMax'
//

// Include files
#include "minOrMax.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
namespace internal {
float b_maximum(const float x[512])
{
  float ex;
  int idx;
  if (!std::isnan(x[0])) {
    idx = 1;
  } else {
    int k;
    boolean_T exitg1;
    idx = 0;
    k = 2;
    exitg1 = false;
    while ((!exitg1) && (k <= 512)) {
      if (!std::isnan(x[k - 1])) {
        idx = k;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }
  if (idx == 0) {
    ex = x[0];
  } else {
    ex = x[idx - 1];
    idx++;
    for (int b_k{idx}; b_k < 513; b_k++) {
      float f;
      f = x[b_k - 1];
      if (ex < f) {
        ex = f;
      }
    }
  }
  return ex;
}

float b_maximum(const float x[512], int &idx)
{
  float ex;
  int b_idx;
  if (!std::isnan(x[0])) {
    b_idx = 1;
  } else {
    int k;
    boolean_T exitg1;
    b_idx = 0;
    k = 2;
    exitg1 = false;
    while ((!exitg1) && (k < 513)) {
      if (!std::isnan(x[k - 1])) {
        b_idx = k;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }
  if (b_idx == 0) {
    ex = x[0];
    idx = 1;
  } else {
    ex = x[b_idx - 1];
    idx = b_idx;
    b_idx++;
    for (int b_k{b_idx}; b_k < 513; b_k++) {
      float f;
      f = x[b_k - 1];
      if (ex < f) {
        ex = f;
        idx = b_k;
      }
    }
  }
  return ex;
}

float maximum(const float x[4096], int &idx)
{
  float ex;
  int b_idx;
  if (!std::isnan(x[0])) {
    b_idx = 1;
  } else {
    int k;
    boolean_T exitg1;
    b_idx = 0;
    k = 2;
    exitg1 = false;
    while ((!exitg1) && (k < 4097)) {
      if (!std::isnan(x[k - 1])) {
        b_idx = k;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }
  if (b_idx == 0) {
    ex = x[0];
    idx = 1;
  } else {
    ex = x[b_idx - 1];
    idx = b_idx;
    b_idx++;
    for (int b_k{b_idx}; b_k < 4097; b_k++) {
      float f;
      f = x[b_k - 1];
      if (ex < f) {
        ex = f;
        idx = b_k;
      }
    }
  }
  return ex;
}

float maximum(const float x[4096])
{
  float ex;
  int idx;
  if (!std::isnan(x[0])) {
    idx = 1;
  } else {
    int k;
    boolean_T exitg1;
    idx = 0;
    k = 2;
    exitg1 = false;
    while ((!exitg1) && (k <= 4096)) {
      if (!std::isnan(x[k - 1])) {
        idx = k;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }
  if (idx == 0) {
    ex = x[0];
  } else {
    ex = x[idx - 1];
    idx++;
    for (int b_k{idx}; b_k < 4097; b_k++) {
      float f;
      f = x[b_k - 1];
      if (ex < f) {
        ex = f;
      }
    }
  }
  return ex;
}

float maximum(const array<float, 1U> &x)
{
  float ex;
  int last;
  last = x.size(0);
  if (x.size(0) <= 2) {
    if (x.size(0) == 1) {
      ex = x[0];
    } else {
      ex = x[x.size(0) - 1];
      if ((!(x[0] < ex)) && ((!std::isnan(x[0])) || std::isnan(ex))) {
        ex = x[0];
      }
    }
  } else {
    int idx;
    if (!std::isnan(x[0])) {
      idx = 1;
    } else {
      int k;
      boolean_T exitg1;
      idx = 0;
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= last)) {
        if (!std::isnan(x[k - 1])) {
          idx = k;
          exitg1 = true;
        } else {
          k++;
        }
      }
    }
    if (idx == 0) {
      ex = x[0];
    } else {
      ex = x[idx - 1];
      idx++;
      for (int b_k{idx}; b_k <= last; b_k++) {
        float f;
        f = x[b_k - 1];
        if (ex < f) {
          ex = f;
        }
      }
    }
  }
  return ex;
}

float maximum(const array<float, 1U> &x, int &idx)
{
  float ex;
  int last;
  last = x.size(0);
  if (x.size(0) <= 2) {
    if (x.size(0) == 1) {
      ex = x[0];
      idx = 1;
    } else {
      ex = x[x.size(0) - 1];
      if ((x[0] < ex) || (std::isnan(x[0]) && (!std::isnan(ex)))) {
        idx = x.size(0);
      } else {
        ex = x[0];
        idx = 1;
      }
    }
  } else {
    int b_idx;
    if (!std::isnan(x[0])) {
      b_idx = 1;
    } else {
      int k;
      boolean_T exitg1;
      b_idx = 0;
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= last)) {
        if (!std::isnan(x[k - 1])) {
          b_idx = k;
          exitg1 = true;
        } else {
          k++;
        }
      }
    }
    if (b_idx == 0) {
      ex = x[0];
      idx = 1;
    } else {
      ex = x[b_idx - 1];
      idx = b_idx;
      b_idx++;
      for (int b_k{b_idx}; b_k <= last; b_k++) {
        float f;
        f = x[b_k - 1];
        if (ex < f) {
          ex = f;
          idx = b_k;
        }
      }
    }
  }
  return ex;
}

} // namespace internal
} // namespace coder

// End of code generation (minOrMax.cpp)
