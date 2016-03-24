/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#include "../include/BlasWrapper.h"

#ifdef IMPL_SGEMM

// detailed implementation of matrix-matrix multiplication
void cblas_sgemm_nn(const CBLAS_INT m, const CBLAS_INT n, const CBLAS_INT k,
    const float alpha, const float *a, const CBLAS_INT lda, const float *b,
    const CBLAS_INT ldb, const float beta, float *c, const CBLAS_INT ldc);
void cblas_sgemm_nt(const CBLAS_INT m, const CBLAS_INT n, const CBLAS_INT k,
    const float alpha, const float *a, const CBLAS_INT lda, const float *b,
    const CBLAS_INT ldb, const float beta, float *c, const CBLAS_INT ldc);
void cblas_sgemm_tn(const CBLAS_INT m, const CBLAS_INT n, const CBLAS_INT k,
    const float alpha, const float *a, const CBLAS_INT lda, const float *b,
    const CBLAS_INT ldb, const float beta, float *c, const CBLAS_INT ldc);
void cblas_sgemm_tt(const CBLAS_INT m, const CBLAS_INT n, const CBLAS_INT k,
    const float alpha, const float *a, const CBLAS_INT lda, const float *b,
    const CBLAS_INT ldb, const float beta, float *c, const CBLAS_INT ldc);

// matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C
void cblas_sgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transA,
    const CBLAS_TRANSPOSE transB, const CBLAS_INT m, const CBLAS_INT n,
    const CBLAS_INT k, const float alpha, const float* a,
    const CBLAS_INT lda, const float* b, const CBLAS_INT ldb,
    const float beta, float* c, const CBLAS_INT ldc) {
  // validate <Layout> parameter
  if (order != CblasRowMajor) {
    printf("[ERROR] only <CblasRowMajor> is supported\n");
    return;
  }  // ENDIF: order

  // choose the proper entry according to <transA> and <transB>
  if (transA == CblasNoTrans) {
    if (transB == CblasNoTrans) {
      cblas_sgemm_nn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
      cblas_sgemm_nt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }  // ENDIF: transB
  } else {
    if (transB == CblasNoTrans) {
      cblas_sgemm_tn(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
      cblas_sgemm_tt(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }  // ENDIF: transB
  }  // ENDIF: transA
}

// matrix-matrix multiplication (A: not transposed; B: not transposed)
void cblas_sgemm_nn(const CBLAS_INT m, const CBLAS_INT n, const CBLAS_INT k,
    const float alpha, const float *a, const CBLAS_INT lda, const float *b,
    const CBLAS_INT ldb, const float beta, float *c, const CBLAS_INT ldc) {
  // compute matrix-matrix multiplication
  for (CBLAS_INT im = 0; im < m; im++) {
    const float* pa = a + im * lda;
    float* pc = c + im * ldc;

    for (CBLAS_INT in = 0; in < n; in++) {
      pc[in] *= beta;
    }  // ENDFOR: in
    for (CBLAS_INT ik = 0; ik < k; ik++) {
      const float va = pa[ik] * alpha;
      const float* pb = b + ik * ldb;
      for (CBLAS_INT in = 0; in < n; in++) {
        pc[in] += va * pb[in];
      }  // ENDFOR: in
    }  // ENDFOR: ik
  }  // ENDFOR: im
}

// matrix-matrix multiplication (A: not transposed; B: transposed)
void cblas_sgemm_nt(const CBLAS_INT m, const CBLAS_INT n, const CBLAS_INT k,
    const float alpha, const float *a, const CBLAS_INT lda, const float *b,
    const CBLAS_INT ldb, const float beta, float *c, const CBLAS_INT ldc) {
  // compute matrix-matrix multiplication
  for (CBLAS_INT im = 0; im < m; im++) {
    const float* pa = a + im * lda;
    float* pc = c + im * ldc;

    for (CBLAS_INT in = 0; in < n; in++) {
      pc[in] *= beta;
    }  // ENDFOR: in
    for (CBLAS_INT in = 0; in < n; in++) {
      const float* pb = b + in * ldb;
      float val = 0.0;
      for (CBLAS_INT ik = 0; ik < k; ik++) {
        val += pa[ik] * pb[ik];
      }  // ENDFOR: ik
      pc[in] += val * alpha;
    }  // ENDFOR: in
  }  // ENDFOR: im
}

// matrix-matrix multiplication (A: transposed; B: not transposed)
void cblas_sgemm_tn(const CBLAS_INT m, const CBLAS_INT n, const CBLAS_INT k,
    const float alpha, const float *a, const CBLAS_INT lda, const float *b,
    const CBLAS_INT ldb, const float beta, float *c, const CBLAS_INT ldc) {
  printf("[ERROR] trans(A) * B is not supported\n");
}

// matrix-matrix multiplication (A: transposed; B: transposed)
void cblas_sgemm_tt(const CBLAS_INT m, const CBLAS_INT n, const CBLAS_INT k,
    const float alpha, const float *a, const CBLAS_INT lda, const float *b,
    const CBLAS_INT ldb, const float beta, float *c, const CBLAS_INT ldc) {
  printf("[ERROR] trans(A) * trans(B) is not supported\n");
}

#endif  // IMPL_SGEMM
