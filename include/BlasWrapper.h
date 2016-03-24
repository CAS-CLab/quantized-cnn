/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#ifndef INCLUDE_BLASWRAPPER_H_
#define INCLUDE_BLASWRAPPER_H_

/*
* functions to be wrapped:
*   1. vsAdd
*   2. vsSub
*   3. vsMul
*   4. vsSqr
*   5. vsPowx
*   6. cblas_sscal
*   7. cblas_saxpy
*   8. cblas_sgemm
*
* functions provided by extern libraries:
*   |          | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
*   |==========|===|===|===|===|===|===|===|===|
*   | ATLAS    | - | - | - | - | - | + | + | + |
*   | MKL      | + | + | + | + | + | + | + | + |
*   | OpenBLAS | - | - | - | - | - | + | + | + |
*   | OpenVML  | + | + | - | - | + | - | - | - |
*
* NOTE: the vsPowx() function in MKL and OpenVML have different definitions,
*       so a wrapper is always needed.
*/

#include "../include/Common.h"

// define macros for all functions to be wrapped (may be undefined later)
#define IMPL_VSADD
#define IMPL_VSSUB
#define IMPL_VSMUL
#define IMPL_VSSQR
#define IMPL_SSCAL
#define IMPL_SAXPY
#define IMPL_SGEMM

// when ATLAS/OpenBLAS is enabled
#if defined(ENABLE_ATLAS) || defined(ENABLE_OPENBLAS)
  extern "C" {
    #include <cblas.h>
  }
#endif  // defined

// when ATLAS is enabled
#ifdef ENABLE_ATLAS
  typedef int CBLAS_INT;
  #undef IMPL_SSCAL
  #undef IMPL_SAXPY
  #undef IMPL_SGEMM
#endif  // ENABLE_ATLAS

// when MKL is enabled
#ifdef ENABLE_MKL
  #include <mkl.h>
  typedef MKL_INT CBLAS_INT;
  typedef CBLAS_LAYOUT CBLAS_ORDER;
  #undef IMPL_VSADD
  #undef IMPL_VSSUB
  #undef IMPL_VSMUL
  #undef IMPL_VSSQR
  #undef IMPL_SSCAL
  #undef IMPL_SAXPY
  #undef IMPL_SGEMM
#endif  // ENABLE_MKL

// when OpenBLAS is enabled
#ifdef ENABLE_OPENBLAS
  typedef blasint CBLAS_INT;
  #undef IMPL_SSCAL
  #undef IMPL_SAXPY
  #undef IMPL_SGEMM
#endif  // ENABLE_OPENBLAS

// when OpenVML is enabled
#ifdef ENABLE_OPENVML
  #include <openvml.h>
  #undef IMPL_VSADD
  #undef IMPL_VSSUB
#endif  // ENABLE_OPENVML

// define variable types for BLAS functions
#if defined(IMPL_VSADD) || defined(IMPL_VSSUB) || defined(IMPL_VSMUL) || \
    defined(IMPL_VSSQR) || defined(IMPL_VSPOWX) || defined(IMPL_SSCAL) || \
    defined(IMPL_SAXPY) || defined(IMPL_SGEMM)
  typedef int CBLAS_INT;
#endif  // defined

#if defined(IMPL_SSCAL) || defined(IMPL_SAXPY) || defined(IMPL_SGEMM)
  typedef enum {CblasRowMajor, CblasColMajor} CBLAS_ORDER;
  typedef enum {CblasNoTrans, CblasTrans} CBLAS_TRANSPOSE;
#endif  // defined

#ifdef IMPL_VSADD
inline void vsAdd(const CBLAS_INT n, const float* a, const float* b, float* y) {
  for (CBLAS_INT i = n - 1; i >= 0; --i) {
    y[i] = a[i] + b[i];
  }  // ENDFOR: i
}
#endif  // IMPL_VSADD

#ifdef IMPL_VSSUB
inline void vsSub(const CBLAS_INT n, const float* a, const float* b, float* y) {
  for (CBLAS_INT i = n - 1; i >= 0; --i) {
    y[i] = a[i] - b[i];
  }  // ENDFOR: i
}
#endif  // IMPL_VSSUB

#ifdef IMPL_VSMUL
inline void vsMul(const CBLAS_INT n, const float* a, const float* b, float* y) {
  for (CBLAS_INT i = n - 1; i >= 0; --i) {
    y[i] = a[i] * b[i];
  }  // ENDFOR: i
}
#endif  // IMPL_VSMUL

#ifdef IMPL_VSSQR
inline void vsSqr(const CBLAS_INT n, const float* a, float* y) {
  for (CBLAS_INT i = n - 1; i >= 0; --i) {
    y[i] = a[i] * a[i];
  }  // ENDFOR: i
}
#endif  // IMPL_VSSQR

// this wrapper is always needed, since MKL's and OpenVML's interface differ
inline void vsPowx_m(const CBLAS_INT n,
    const float* a, const float b, float* y) {
  #ifdef ENABLE_MKL
    vsPowx(n, a, b, y);
  #else
    #ifdef ENABLE_OPENVML
      vsPowx(n, a, &b, y);
    #else
      for (CBLAS_INT i = n - 1; i >= 0; --i) {
        y[i] = exp(b * log(a[i]));
      }  // ENDFOR: i
    #endif  // ENABLE_OPENVML
  #endif  // ENABLE_MKL
}

#ifdef IMPL_SSCAL
inline void cblas_sscal(const CBLAS_INT n,
    const float a, float* x, const CBLAS_INT incx) {
  if (incx == 1) {  // case-by-case optimization
    for (CBLAS_INT i = n - 1; i >= 0; --i) {
      x[i] *= a;
    }  // ENDFOR: i
  } else {
    for (CBLAS_INT i = (n - 1) * incx; i >= 0; i -= incx) {
      x[i] *= a;
    }  // ENDFOR: i
  }  // ENDIF: incx
}
#endif  // IMPL_SSCAL

#ifdef IMPL_SAXPY
inline void cblas_saxpy(const CBLAS_INT n, const float a, const float* x,
    const CBLAS_INT incx, float* y, const CBLAS_INT incy) {
  if ((incx == 1) && (incy == 1)) {  // case-by-case optmization
    for (CBLAS_INT i = n - 1; i >= 0; --i) {
      y[i] += a * x[i];
    }  // ENDFOR: i
  } else if (incx == incy) {  // case-by-case optimization
    for (CBLAS_INT i = (n - 1) * incx; i >= 0; ) {
      y[i] += a * x[i];
      i -= incx;
    }  // ENDFOR: i
  } else {
    for (CBLAS_INT i = (n - 1) * incx, j = (n - 1) * incy; i >= 0; ) {
      y[j] += a * x[i];
      i -= incx;
      j -= incy;
    }  // ENDFOR: i
  }  // ENDIF: incx
}
#endif  // IMPL_SAXPY

#ifdef IMPL_SGEMM
void cblas_sgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const CBLAS_INT m, const CBLAS_INT n,
    const CBLAS_INT k, const float alpha, const float* a,
    const CBLAS_INT lda, const float* b, const CBLAS_INT ldb,
    const float beta, float* c, const CBLAS_INT ldc);
#endif  // IMPL_SGEMM

#endif  // INCLUDE_BLASWRAPPER_H_
