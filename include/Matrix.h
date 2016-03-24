/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#ifndef INCLUDE_MATRIX_H_
#define INCLUDE_MATRIX_H_

#include <algorithm>

#include "../include/Common.h"

const int kMatDimCntMax = 4;

template<typename T>
class Matrix {
 public:
  // constructor function
  Matrix(void);
  Matrix(const Matrix<T>& mat);
  explicit Matrix(const int m);
  Matrix(const int m, const int n);
  Matrix(const int m, const int n, const int p);
  Matrix(const int m, const int n, const int p, const int q);
  Matrix(const int dimCnt, const int* dimLenLst);
  // de-constructor function
  ~Matrix(void);
  // create matrix
  inline void Create(const int m);
  inline void Create(const int m, const int n);
  inline void Create(const int m, const int n, const int p);
  inline void Create(const int m, const int n, const int p, const int q);
  inline void Create(const int dimCnt, const int* dimLenLst);
  // release memory
  inline void Destroy(void);
  // obtain the data pointer
  inline T* GetDataPtr(void) const;
  inline T* GetDataPtr(const int im) const;
  inline T* GetDataPtr(const int im, const int in) const;
  inline T* GetDataPtr(const int im, const int in, const int ip) const;
  inline T* GetDataPtr(
      const int im, const int in, const int ip, const int iq) const;
  // obtain the number of dimensions
  inline int GetDimCnt(void) const;
  // obtain the length of x-th dimension
  inline int GetDimLen(const int dimIdx) const;
  // obtain the pointer step of x-th dimension
  inline int GetDimStp(const int dimIdx) const;
  // obtain the total number of elements
  inline int GetEleCnt(void) const;
  // display matrix size information
  inline void DispSizInfo(void) const;
  // set the element value at the specified location
  inline void SetEleAt(const T val, const int im);
  inline void SetEleAt(const T val, const int im, const int in);
  inline void SetEleAt(const T val, const int im, const int in, const int ip);
  inline void SetEleAt(
      const T val, const int im, const int in, const int ip, const int iq);
  // get the element value at the specified location
  inline T GetEleAt(const int im) const;
  inline T GetEleAt(const int im, const int in) const;
  inline T GetEleAt(const int im, const int in, const int ip) const;
  inline T GetEleAt(
      const int im, const int in, const int ip, const int iq) const;
  // resize matrix
  inline void Resize(const int m);
  inline void Resize(const int m, const int n);
  inline void Resize(const int m, const int n, const int p);
  inline void Resize(const int m, const int n, const int p, const int q);
  // permute dimensions
  void Permute(const int mSdx, const int nSdx);
  void Permute(const int mSdx, const int nSdx, const int pSdx);
  void Permute(const int mSdx, const int nSdx, const int pSdx, const int qSdx);
  // get a sub-matrix at the specified location
  void GetSubMat(const int imBeg, Matrix<T>* pMatDst) const;
  void GetSubMat(const int imBeg, const int inBeg, Matrix<T>* pMatDst) const;
  void GetSubMat(const int imBeg,
      const int inBeg, const int ipBeg, Matrix<T>* pMatDst) const;
  void GetSubMat(const int imBeg, const int inBeg,
      const int ipBeg, const int iqBeg, Matrix<T>* pMatDst) const;

 private:
  // number of dimensions
  int dimCnt_;
  // length of the 1st dimension
  int m_;
  // length of the 2nd dimension
  int n_;
  // length of the 3rd dimension
  int p_;
  // length of the 4th dimension
  int q_;
  // data pointer
  T* data_;
};

// implementation of member functions

template<typename T>
Matrix<T>::Matrix(void) {
  // initialize all variables
  dimCnt_ = 0;
  data_ = nullptr;
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& mat) : data_(nullptr) {
  // declare a static array to specify each dimension's length
  static int dimLenLst[kMatDimCntMax];

  // create a new matrix and copy data
  for (int dimIdx = 0; dimIdx < mat.GetDimCnt(); dimIdx++) {
    dimLenLst[dimIdx] = mat.GetDimLen(dimIdx);
  }  // ENDFOR: dimIdx
  Create(mat.GetDimCnt(), dimLenLst);
  memcpy(data_, mat.GetDataPtr(), sizeof(T) * GetEleCnt());
}

template<typename T>
Matrix<T>::Matrix(const int m) : data_(nullptr) {
  Create(m);
}

template<typename T>
Matrix<T>::Matrix(const int m, const int n) : data_(nullptr) {
  Create(m, n);
}

template<typename T>
Matrix<T>::Matrix(const int m, const int n, const int p) : data_(nullptr) {
  Create(m, n, p);
}

template<typename T>
Matrix<T>::Matrix(
    const int m, const int n, const int p, const int q) : data_(nullptr) {
  Create(m, n, p, q);
}

template<typename T>
Matrix<T>::Matrix(const int dimCnt, const int* dimLenLst) : data_(nullptr) {
  Create(dimCnt, dimLenLst);
}

template<typename T>
Matrix<T>::~Matrix(void) {
  Destroy();
}

template<typename T>
inline void Matrix<T>::Create(const int m) {
  // release previously allocated memory
  Destroy();

  // create a new matrix
  dimCnt_ = 1;
  m_ = m;
  data_ = new T[GetEleCnt()];
}

template<typename T>
inline void Matrix<T>::Create(const int m, const int n) {
  // release previously allocated memory
  Destroy();

  // create a new matrix
  dimCnt_ = 2;
  m_ = m;
  n_ = n;
  data_ = new T[GetEleCnt()];
}

template<typename T>
inline void Matrix<T>::Create(const int m, const int n, const int p) {
  // release previously allocated memory
  Destroy();

  // create a new matrix
  dimCnt_ = 3;
  m_ = m;
  n_ = n;
  p_ = p;
  data_ = new T[GetEleCnt()];
}

template<typename T>
inline void Matrix<T>::Create(
    const int m, const int n, const int p, const int q) {
  // release previously allocated memory
  Destroy();

  // create a new matrix
  dimCnt_ = 4;
  m_ = m;
  n_ = n;
  p_ = p;
  q_ = q;
  data_ = new T[GetEleCnt()];
}

template<typename T>
inline void Matrix<T>::Create(const int dimCnt, const int* dimLenLst) {
  switch (dimCnt) {
  case 1:
    Create(dimLenLst[0]);
    break;
  case 2:
    Create(dimLenLst[0], dimLenLst[1]);
    break;
  case 3:
    Create(dimLenLst[0], dimLenLst[1], dimLenLst[2]);
    break;
  case 4:
    Create(dimLenLst[0], dimLenLst[1], dimLenLst[2], dimLenLst[3]);
    break;
  default:
    printf("[ERROR] invalid number of dimensions: %d\n", dimCnt);
    return;
  }  // ENDSWITCH: dimCnt
}

template<typename T>
inline void Matrix<T>::Destroy(void) {
  // release memory
  if (data_ != nullptr) {
    delete[] data_;
    data_ = nullptr;
  }  // ENDIF: data_

  // re-initialize variables
  dimCnt_ = 0;
}

template<typename T>
inline T* Matrix<T>::GetDataPtr(void) const {
  return data_;
}

template<typename T>
inline T* Matrix<T>::GetDataPtr(const int im) const {
  return (data_ + im);
}

template<typename T>
inline T* Matrix<T>::GetDataPtr(const int im, const int in) const {
  return (data_ + (im * n_ + in));
}

template<typename T>
inline T* Matrix<T>::GetDataPtr(
    const int im, const int in, const int ip) const {
  return (data_ + ((im * n_ + in) * p_ + ip));
}

template<typename T>
inline T* Matrix<T>::GetDataPtr(
    const int im, const int in, const int ip, const int iq) const {
  return (data_ + (((im * n_ + in) * p_ + ip) * q_ + iq));
}

template<typename T>
inline int Matrix<T>::GetDimCnt(void) const {
  return dimCnt_;
}

template<typename T>
inline int Matrix<T>::GetDimLen(const int dimIdx) const {
  switch (dimIdx) {
  case 0:
    return m_;
  case 1:
    return n_;
  case 2:
    return p_;
  case 3:
    return q_;
  default:
    printf("[ERROR] invalid index of dimension: %d\n", dimIdx);
    return -1;
  }  // ENDSWITCH: dimIdx
}

template<typename T>
inline int Matrix<T>::GetDimStp(const int dimIdx) const {
  // compute the pointer step iteratively
  if (dimIdx == dimCnt_ - 1) {
    return 1;
  } else {
    return GetDimLen(dimIdx + 1) * GetDimStp(dimIdx + 1);
  }  // ENDIF: dimIdx
}

template<typename T>
inline int Matrix<T>::GetEleCnt(void) const {
  int eleCnt = 1;
  switch (dimCnt_) {
  case 4:
    eleCnt *= q_;
    // fall through
  case 3:
    eleCnt *= p_;
    // fall through
  case 2:
    eleCnt *= n_;
    // fall through
  case 1:
    eleCnt *= m_;
    break;
  default:
    eleCnt = 0;
    break;
  }  // ENDSWITCH: dimCnt

  return eleCnt;
}

template<typename T>
inline void Matrix<T>::DispSizInfo(void) const {
  switch (dimCnt_) {
  case 1:
    printf("[INFO] matrix size: %d\n", m_);
    break;
  case 2:
    printf("[INFO] matrix size: %d x %d\n", m_, n_);
    break;
  case 3:
    printf("[INFO] matrix size: %d x %d x %d\n", m_, n_, p_);
    break;
  case 4:
    printf("[INFO] matrix size: %d x %d x %d x %d\n", m_, n_, p_, q_);
    break;
  }  // ENDSWITCH: dimCnt
}

template<typename T>
inline void Matrix<T>::SetEleAt(const T val, const int im) {
  data_[im] = val;
}

template<typename T>
inline void Matrix<T>::SetEleAt(const T val, const int im, const int in) {
  data_[im * n_ + in] = val;
}

template<typename T>
inline void Matrix<T>::SetEleAt(
    const T val, const int im, const int in, const int ip) {
  data_[(im * n_ + in) * p_ + ip] = val;
}

template<typename T>
inline void Matrix<T>::SetEleAt(
    const T val, const int im, const int in, const int ip, const int iq) {
  data_[((im * n_ + in) * p_ + ip) * q_ + iq] = val;
}

template<typename T>
inline T Matrix<T>::GetEleAt(const int im) const {
  return data_[im];
}

template<typename T>
inline T Matrix<T>::GetEleAt(const int im, const int in) const {
  return data_[im * n_ + in];
}

template<typename T>
inline T Matrix<T>::GetEleAt(const int im, const int in, const int ip) const {
  return data_[(im * n_ + in) * p_ + ip];
}

template<typename T>
inline T Matrix<T>::GetEleAt(
    const int im, const int in, const int ip, const int iq) const {
  return data_[((im * n_ + in) * p_ + ip) * q_ + iq];
}

template<typename T>
inline void Matrix<T>::Resize(const int m) {
  // check matrix size
  if (GetEleCnt() != m) {
    Create(m);
  }  // ENDIF: GetEleCnt
}

template<typename T>
inline void Matrix<T>::Resize(const int m, const int n) {
  // check matrix size
  if (GetEleCnt() != m * n) {
    Create(m, n);
  } else {
    dimCnt_ = 2;
    m_ = m;
    n_ = n;
  }  // ENDIF: GetEleCnt
}

template<typename T>
inline void Matrix<T>::Resize(const int m, const int n, const int p) {
  // check matrix size
  if (GetEleCnt() != m * n * p) {
    Create(m, n, p);
  } else {
    dimCnt_ = 3;
    m_ = m;
    n_ = n;
    p_ = p;
  }  // ENDIF: GetEleCnt
}

template<typename T>
inline void Matrix<T>::Resize(
    const int m, const int n, const int p, const int q) {
  // check matrix size
  if (GetEleCnt() != m * n * p * q) {
    Create(m, n, p, q);
  } else {
    dimCnt_ = 4;
    m_ = m;
    n_ = n;
    p_ = p;
    q_ = q;
  }  // ENDIF: GetEleCnt
}

template<typename T>
void Matrix<T>::Permute(const int mSdx, const int nSdx) {
  // create a temporary array and copy all data
  int eleCnt = GetEleCnt();
  T* dataTmp = new T[eleCnt];
  memcpy(dataTmp, data_, sizeof(T) * eleCnt);

  // determine the length of each dimension after updating
  int dimLenLstSrc[kMatDimCntMax];
  int dimStpLstSrc[kMatDimCntMax];
  for (int dimIdx = 0; dimIdx < dimCnt_; dimIdx++) {
    dimLenLstSrc[dimIdx] = GetDimLen(dimIdx);
    dimStpLstSrc[dimIdx] = GetDimStp(dimIdx);
  }

  // update each dimension's length
  m_ = dimLenLstSrc[mSdx];
  n_ = dimLenLstSrc[nSdx];

  // copy data from the temporary array
  for (int im = 0; im < m_; im++) {
    int mOffset = im * dimStpLstSrc[mSdx];

    // determine the source/destination data pointer
    int stpSrc = dimStpLstSrc[nSdx];
    const T* pSrc = dataTmp + mOffset;
    T* pDst = GetDataPtr(im, 0);

    // copy data to (im, in, ip, :)
    for (int in = 0; in < n_; in++, pSrc += stpSrc) {
      *(pDst++) = *pSrc;
    }  // ENDFOR: in
  }  // ENDFOR: im

  // release the temporary array
  delete[] dataTmp;
}

template<typename T>
void Matrix<T>::Permute(const int mSdx, const int nSdx, const int pSdx) {
  // create a temporary array and copy all data
  int eleCnt = GetEleCnt();
  T* dataTmp = new T[eleCnt];
  memcpy(dataTmp, data_, sizeof(T) * eleCnt);

  // determine the length of each dimension after updating
  int dimLenLstSrc[kMatDimCntMax];
  int dimStpLstSrc[kMatDimCntMax];
  for (int dimIdx = 0; dimIdx < dimCnt_; dimIdx++) {
    dimLenLstSrc[dimIdx] = GetDimLen(dimIdx);
    dimStpLstSrc[dimIdx] = GetDimStp(dimIdx);
  }

  // update each dimension's length
  m_ = dimLenLstSrc[mSdx];
  n_ = dimLenLstSrc[nSdx];
  p_ = dimLenLstSrc[pSdx];

  // copy data from the temporary array
  for (int im = 0; im < m_; im++) {
    int mOffset = im * dimStpLstSrc[mSdx];
    for (int in = 0; in < n_; in++) {
      int nOffset = in * dimStpLstSrc[nSdx];

      // determine the source/destination data pointer
      int stpSrc = dimStpLstSrc[pSdx];
      const T* pSrc = dataTmp + mOffset + nOffset;
      T* pDst = GetDataPtr(im, in, 0);

      // copy data to (im, in, ip, :)
      for (int ip = 0; ip < p_; ip++, pSrc += stpSrc) {
        *(pDst++) = *pSrc;
      }  // ENDFOR: ip
    }  // ENDFOR: in
  }  // ENDFOR: im

  // release the temporary array
  delete[] dataTmp;
}

template<typename T>
void Matrix<T>::Permute(
    const int mSdx, const int nSdx, const int pSdx, const int qSdx) {
  // create a temporary array and copy all data
  int eleCnt = GetEleCnt();
  T* dataTmp = new T[eleCnt];
  memcpy(dataTmp, data_, sizeof(T) * eleCnt);

  // determine the length of each dimension after updating
  int dimLenLstSrc[kMatDimCntMax];
  int dimStpLstSrc[kMatDimCntMax];
  for (int dimIdx = 0; dimIdx < dimCnt_; dimIdx++) {
    dimLenLstSrc[dimIdx] = GetDimLen(dimIdx);
    dimStpLstSrc[dimIdx] = GetDimStp(dimIdx);
  }  // ENDFOR: dimIdx

  // update each dimension's length
  m_ = dimLenLstSrc[mSdx];
  n_ = dimLenLstSrc[nSdx];
  p_ = dimLenLstSrc[pSdx];
  q_ = dimLenLstSrc[qSdx];

  // copy data from the temporary array
  for (int im = 0; im < m_; im++) {
    int mOffset = im * dimStpLstSrc[mSdx];
    for (int in = 0; in < n_; in++) {
      int nOffset = in * dimStpLstSrc[nSdx];
      for (int ip = 0; ip < p_; ip++) {
        int pOffset = ip * dimStpLstSrc[pSdx];

        // determine the source/destination data pointer
        int stpSrc = dimStpLstSrc[qSdx];
        const T* pSrc = dataTmp + mOffset + nOffset + pOffset;
        T* pDst = GetDataPtr(im, in, ip, 0);

        // copy data to (im, in, ip, :)
        for (int iq = 0; iq < q_; iq++, pSrc += stpSrc) {
          *(pDst++) = *pSrc;
        }  // ENDFOR: iq
      }  // ENDFOR: ip
    }  // ENDFOR: in
  }  // ENDFOR: im

  // release the temporary array
  delete[] dataTmp;
}

template<typename T>
void Matrix<T>::GetSubMat(const int imBeg, Matrix<T>* pMatDst) const {
  // determine the starting/ending height/width indexes
  int imDstBeg = std::max(0, 0 - imBeg);
  int imDstEnd = std::min(pMatDst->GetDimLen(0) - 1, m_ - 1 - imBeg);
  int featVecLen = imDstEnd - imDstBeg + 1;

  // reset all elements in <featMapDst> to zeros
  memset(pMatDst->GetDataPtr(), 0, sizeof(T) * pMatDst->GetEleCnt());

  // copy the selected part in the feature map
  const T* featVecSrc = GetDataPtr(imDstBeg + imBeg);
  T* featVecDst = pMatDst->GetDataPtr(imDstBeg);
  memcpy(featVecDst, featVecSrc, sizeof(T) * featVecLen);
}

template<typename T>
void Matrix<T>::GetSubMat(
    const int imBeg, const int inBeg, Matrix<T>* pMatDst) const {
  // determine the starting/ending height/width indexes
  int imDstBeg = std::max(0, 0 - imBeg);
  int imDstEnd = std::min(pMatDst->GetDimLen(0) - 1, m_ - 1 - imBeg);
  int inDstBeg = std::max(0, 0 - inBeg);
  int inDstEnd = std::min(pMatDst->GetDimLen(1) - 1, n_ - 1 - inBeg);
  int featVecLen = inDstEnd - inDstBeg + 1;

  // reset all elements in <featMapDst> to zeros
  memset(pMatDst->GetDataPtr(), 0, sizeof(T) * pMatDst->GetEleCnt());

  // copy the selected part in the feature map
  for (int imDst = imDstBeg; imDst <= imDstEnd; imDst++) {
    int im = imDst + imBeg;
    const T* featVecSrc = GetDataPtr(im, inDstBeg + inBeg);
    T* featVecDst = pMatDst->GetDataPtr(imDst, inDstBeg);
    memcpy(featVecDst, featVecSrc, sizeof(T) * featVecLen);
  }  // ENDFOR: imDst
}

template<typename T>
void Matrix<T>::GetSubMat(const int imBeg,
    const int inBeg, const int ipBeg, Matrix<T>* pMatDst) const {
  // determine the starting/ending height/width indexes
  int imDstBeg = std::max(0, 0 - imBeg);
  int imDstEnd = std::min(pMatDst->GetDimLen(0) - 1, m_ - 1 - imBeg);
  int inDstBeg = std::max(0, 0 - inBeg);
  int inDstEnd = std::min(pMatDst->GetDimLen(1) - 1, n_ - 1 - inBeg);
  int ipDstBeg = std::max(0, 0 - ipBeg);
  int ipDstEnd = std::min(pMatDst->GetDimLen(2) - 1, p_ - 1 - ipBeg);
  int featVecLen = ipDstEnd - ipDstBeg + 1;

  // reset all elements in <featMapDst> to zeros
  memset(pMatDst->GetDataPtr(), 0, sizeof(T) * pMatDst->GetEleCnt());

  // copy the selected part in the feature map
  for (int imDst = imDstBeg; imDst <= imDstEnd; imDst++) {
    int im = imDst + imBeg;
    for (int inDst = inDstBeg; inDst <= inDstEnd; inDst++) {
      int in = inDst + inBeg;
      const T* featVecSrc = GetDataPtr(im, in, ipDstBeg + ipBeg);
      T* featVecDst = pMatDst->GetDataPtr(imDst, inDst, ipDstBeg);
      memcpy(featVecDst, featVecSrc, sizeof(T) * featVecLen);
    }  // ENDFOR: inDst
  }  // ENDFOR: imDst
}

template<typename T>
void Matrix<T>::GetSubMat(const int imBeg, const int inBeg,
    const int ipBeg, const int iqBeg, Matrix<T>* pMatDst) const {
  // determine the starting/ending height/width indexes
  int imDstBeg = std::max(0, 0 - imBeg);
  int imDstEnd = std::min(pMatDst->GetDimLen(0) - 1, m_ - 1 - imBeg);
  int inDstBeg = std::max(0, 0 - inBeg);
  int inDstEnd = std::min(pMatDst->GetDimLen(1) - 1, n_ - 1 - inBeg);
  int ipDstBeg = std::max(0, 0 - ipBeg);
  int ipDstEnd = std::min(pMatDst->GetDimLen(2) - 1, p_ - 1 - ipBeg);
  int iqDstBeg = std::max(0, 0 - iqBeg);
  int iqDstEnd = std::min(pMatDst->GetDimLen(3) - 1, q_ - 1 - iqBeg);
  int featVecLen = iqDstEnd - iqDstBeg + 1;

  // reset all elements in <featMapDst> to zeros
  memset(pMatDst->GetDataPtr(), 0, sizeof(T) * pMatDst->GetEleCnt());

  // copy the selected part in the feature map
  for (int imDst = imDstBeg; imDst <= imDstEnd; imDst++) {
    int im = imDst + imBeg;
    for (int inDst = inDstBeg; inDst <= inDstEnd; inDst++) {
      int in = inDst + inBeg;
      for (int ipDst = ipDstBeg; ipDst <= ipDstEnd; ipDst++) {
        int ip = ipDst + ipBeg;
        const T* featVecSrc = GetDataPtr(im, in, ip, iqDstBeg + iqBeg);
        T* featVecDst = pMatDst->GetDataPtr(imDst, inDst, ipDst, iqDstBeg);
        memcpy(featVecDst, featVecSrc, sizeof(T) * featVecLen);
      }  // ENDFOR: ipDst
    }  // ENDFOR: inDst
  }  // ENDFOR: imDst
}

#endif  // INCLUDE_MATRIX_H_
