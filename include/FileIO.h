/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#ifndef INCLUDE_FILEIO_H_
#define INCLUDE_FILEIO_H_

#include <string>
#include <algorithm>

#include "../include/Common.h"
#include "../include/Matrix.h"

typedef struct {
  std::string typeName;
  std::string inputFrmtStr;
  std::string outputFrmtStr;
} TypeInfo;

class FileIO {
 public:
  // read from a binary-encoded file
  template<typename T>
  static bool ReadBinFile(const std::string& filePath, Matrix<T>* pDataLst);
  // read from a compact-binary-encoded file
  template<typename T>
  static bool ReadCbnFile(const std::string& filePath, Matrix<T>* pDataLst);
  // read from a plain text file
  template<typename T>
  static bool ReadTxtFile(const std::string& filePath, Matrix<T>* pDataLst);
  // write to a binary-encoded file
  template<typename T>
  static bool WriteBinFile(
      const std::string& filePath, const Matrix<T>& dataLst);
  // write to a compact-binary-encoded file
  template<typename T>
  static bool WriteCbnFile(const std::string& filePath,
      const Matrix<T>& dataLst, const int bitCntPerEle);
  // write to a plain text file
  template<typename T>
  static bool WriteTxtFile(
      const std::string& filePath, const Matrix<T>& dataLst);

 private:
  // get <T>'s type name
  template<typename T>
  static void GetTypeInfo(const Matrix<T>& dataLst, TypeInfo* pTypeInfo);
};

// implementation of member functions

template<typename T>
bool FileIO::ReadBinFile(const std::string& filePath, Matrix<T>* pDataLst) {
  // declare auxiliary variables
  std::size_t rtnVal;
  int32_t dimCnt;
  int32_t* dimLenLst;
  const int kBufferLenData = 4096;

  // open file
  FILE* inFile = fopen(filePath.c_str(), "rb");
  if (inFile == nullptr) {
    printf("[ERROR] could not open file at %s\n", filePath.c_str());
    return false;
  }  // ENDIF: inFile

  // read basic variables
  rtnVal = fread(&dimCnt, sizeof(int32_t), 1, inFile);
  dimLenLst = new int32_t[dimCnt];
  rtnVal = fread(dimLenLst, sizeof(int32_t), dimCnt, inFile);

  // allocate memory space for <pDataLst>
  pDataLst->Create(dimCnt, dimLenLst);

  // read data matrix from file
  printf("[INFO] reading data matrix from %s\n", filePath.c_str());
  int dataCnt = pDataLst->GetEleCnt();
  int dataCntInBuffer = kBufferLenData / sizeof(T);
  int bufferCntData = (dataCnt + dataCntInBuffer - 1) / dataCntInBuffer;
  int dataCntInBufferLast = dataCnt - dataCntInBuffer * (bufferCntData - 1);
  for (int bufferIndData = 0; bufferIndData < bufferCntData; bufferIndData++) {
    // obtain the data pointer for data copying
    T* dataVec = pDataLst->GetDataPtr() + dataCntInBuffer * bufferIndData;

    // check whether is the last data buffer
    if (bufferIndData < bufferCntData - 1) {  // not the last data buffer
      rtnVal = fread(dataVec, sizeof(T), dataCntInBuffer, inFile);
      assert(rtnVal == dataCntInBuffer);
    } else {  // the last data buffer
      rtnVal = fread(dataVec, sizeof(T), dataCntInBufferLast, inFile);
      assert(rtnVal == dataCntInBufferLast);
    }  // ENDIF: bufferIndData
  }  // ENDFOR: bufferIndData
  pDataLst->DispSizInfo();
  printf("[INFO] reading data matrix from %s [COMPLETED]\n", filePath.c_str());

  // close file
  fclose(inFile);

  // free pointers
  delete[] dimLenLst;

  return true;
}

template<typename T>
bool FileIO::ReadCbnFile(const std::string& filePath, Matrix<T>* pDataLst) {
  // declare auxiliary variables
  std::size_t rtnVal;
  int32_t dimCnt;
  int32_t* dimLenLst;
  int32_t bitCntPerEle;  // each data element's length (in bits)
  const int kBufferLen = 4096;
  uint8_t bufferArray[kBufferLen];
  const int kBitCntPerBuf = 8;  // each buffer element has 8 bits

  // open file
  FILE* inFile = fopen(filePath.c_str(), "rb");
  if (inFile == nullptr) {
    printf("[ERROR] could not open file at %s\n", filePath.c_str());
    return false;
  }  // ENDIF: inFile

  // read basic variables
  rtnVal = fread(&dimCnt, sizeof(int32_t), 1, inFile);
  dimLenLst = new int32_t[dimCnt];
  rtnVal = fread(dimLenLst, sizeof(int32_t), dimCnt, inFile);
  rtnVal = fread(&bitCntPerEle, sizeof(int32_t), 1, inFile);

  // allocate memory space for <pDataLst>
  pDataLst->Create(dimCnt, dimLenLst);

  // read data matrix from file
  printf("[INFO] reading data matrix from %s\n", filePath.c_str());
  int dataCnt = pDataLst->GetEleCnt();
  int dataCntInBuf = kBufferLen * kBitCntPerBuf / bitCntPerEle;
  int bufferCnt = (dataCnt + dataCntInBuf - 1) / dataCntInBuf;
  for (int bufferInd = 0; bufferInd < bufferCnt; bufferInd++) {
    // determine the starting/ending indices of the selected data elements
    int dataIndBeg = dataCntInBuf * bufferInd;
    int dataIndEnd = std::min(dataCnt, dataIndBeg + dataCntInBuf) - 1;
    int dataCntSel = dataIndEnd - dataIndBeg + 1;

    // read file and store data in the buffer
    rtnVal = fread(bufferArray, sizeof(uint8_t), kBufferLen, inFile);
    assert(rtnVal == kBufferLen);

    // copy data to the buffer
    uint8_t* pBuffer = bufferArray;
    T* dataVec = pDataLst->GetDataPtr() + dataIndBeg;
    int bitCntLeft = kBitCntPerBuf;
    for (int dataIndSel = 0; dataIndSel < dataCntSel; dataIndSel++) {
      if (bitCntLeft >= bitCntPerEle) {
        bitCntLeft -= bitCntPerEle;
        dataVec[dataIndSel] = *pBuffer >> bitCntLeft;
      } else {
        dataVec[dataIndSel] = *(pBuffer++) << (bitCntPerEle - bitCntLeft);
        bitCntLeft = kBitCntPerBuf - (bitCntPerEle - bitCntLeft);
        dataVec[dataIndSel] |= *pBuffer >> bitCntLeft;
      }  // ENDIF: bitCntLeft
      *pBuffer %= (1 << bitCntLeft);
      dataVec[dataIndSel] += 1;  // append the offset
    }  // ENDFOR: dataIndSel
  }  // ENDFOR: bufferInd
  pDataLst->DispSizInfo();
  printf("[INFO] reading data matrix from %s [COMPLETED]\n", filePath.c_str());

  // close file
  fclose(inFile);

  // free pointers
  delete[] dimLenLst;

  return true;
}

template<typename T>
bool FileIO::ReadTxtFile(const std::string& filePath, Matrix<T>* pDataLst) {
  // declare auxiliary variables
  std::size_t rtnVal;
  int32_t dimCnt;
  int32_t* dimLenLst;
  TypeInfo typeInfo;

  // open file
  FILE* inFile = fopen(filePath.c_str(), "r");
  if (inFile == nullptr) {
    printf("[ERROR] could not open file at %s\n", filePath.c_str());
    return false;
  }  // ENDIF: inFile

  // read basic variables
  rtnVal = fscanf(inFile, "%d", &dimCnt);
  assert(rtnVal == 1);
  dimLenLst = new int32_t[dimCnt];
  for (int dimInd = 0; dimInd < dimCnt; dimInd++) {
    rtnVal = fscanf(inFile, "%d", dimLenLst + dimInd);
    assert(rtnVal == 1);
  }  // ENDFOR: dimInd

  // allocate memory space for <pDataLst>
  pDataLst->Create(dimCnt, dimLenLst);

  // read data matrix from file
  printf("[INFO] reading data matrix from %s\n", filePath.c_str());
  GetTypeInfo(*pDataLst, &typeInfo);
  int dataCnt = pDataLst->GetEleCnt();
  T* dataPtr = pDataLst->GetDataPtr();
  for (int dataInd = 0; dataInd < dataCnt; dataInd++, dataPtr++) {
    rtnVal = fscanf(inFile, typeInfo.inputFrmtStr.c_str(), dataPtr);
    assert(rtnVal == 1);
  }  // ENDFOR: dataInd
  pDataLst->DispSizInfo();
  printf("[INFO] reading data matrix from %s [COMPLETED]\n", filePath.c_str());

  // close file
  fclose(inFile);

  // free pointers
  delete[] dimLenLst;

  return true;
}

template<typename T>
bool FileIO::WriteBinFile(
    const std::string& filePath, const Matrix<T>& dataLst) {
  // declare auxiliary variables
  int rtnVal;
  int32_t dimCnt;
  int32_t dimLen;
  const int kBufferLenData = 4096;

  // open file
  FILE* outFile = fopen(filePath.c_str(), "wb");
  if (outFile == nullptr) {
    printf("[ERROR] could not open file at %s\n", filePath.c_str());
    return false;
  }  // ENDIF: outFile

  // write basic variables
  dimCnt = dataLst.GetDimCnt();
  fwrite(&dimCnt, sizeof(int32_t), 1, outFile);
  for (int dimInd = 0; dimInd < dimCnt; dimInd++) {
    dimLen = dataLst.GetDimLen(dimInd);
    fwrite(&dimLen, sizeof(int32_t), 1, outFile);
  }  // ENDFOR: dimInd

  // write data matrix to file
  printf("[INFO] writing data matrix from %s\n", filePath.c_str());
  int dataCnt = dataLst.GetEleCnt();
  int dataCntInBuffer = kBufferLenData / sizeof(T);
  int bufferCntData = (dataCnt + dataCntInBuffer - 1) / dataCntInBuffer;
  int dataCntInBufferLast = dataCnt - dataCntInBuffer * (bufferCntData - 1);
  for (int bufferIndData = 0; bufferIndData < bufferCntData; bufferIndData++) {
    // obtain the data pointer for data copying
    T* dataVec = dataLst.GetDataPtr() + dataCntInBuffer * bufferIndData;

    // check whether is the last data buffer
    if (bufferIndData < bufferCntData - 1) {  // not the last data buffer
      rtnVal = fwrite(dataVec, sizeof(T), dataCntInBuffer, outFile);
      assert(rtnVal == dataCntInBuffer);
    } else {  // the last data buffer
      rtnVal = fwrite(dataVec, sizeof(T), dataCntInBufferLast, outFile);
      assert(rtnVal == dataCntInBufferLast);
    }  // ENDIF: bufferIndData
  }  // ENDFOR: bufferIndData
  dataLst.DispSizInfo();
  printf("[INFO] writing data matrix from %s [COMPLETED]\n", filePath.c_str());

  // close file
  fclose(outFile);

  return true;
}

template<typename T>
bool FileIO::WriteCbnFile(const std::string& filePath,
    const Matrix<T>& dataLst, const int bitCntPerEle) {
  // declare auxiliary variables
  int rtnVal;
  int32_t dimCnt;
  int32_t dimLen;
  const int kBufferLen = 4096;
  uint8_t bufferArray[kBufferLen];
  const int kBitCntPerBuf = 8;  // each buffer element hass 8 bits

  // open file
  FILE* outFile = fopen(filePath.c_str(), "wb");
  if (outFile == nullptr) {
    printf("[ERROR] could not open file at %s\n", filePath.c_str());
    return false;
  }  // ENDIF: outFile

  // write basic variables
  dimCnt = dataLst.GetDimCnt();
  fwrite(&dimCnt, sizeof(int32_t), 1, outFile);
  for (int dimInd = 0; dimInd < dimCnt; dimInd++) {
    dimLen = dataLst.GetDimLen(dimInd);
    fwrite(&dimLen, sizeof(int32_t), 1, outFile);
  }  // ENDFOR: dimInd

  // write the number of bits for each data element
  fwrite(&bitCntPerEle, sizeof(int32_t), 1, outFile);

  // write data matrix to file
  printf("[INFO] writing data matrix from %s\n", filePath.c_str());
  int dataCnt = dataLst.GetEleCnt();
  int dataCntInBuf = kBufferLen * kBitCntPerBuf / bitCntPerEle;
  int bufferCnt = (dataCnt + dataCntInBuf - 1) / dataCntInBuf;
  for (int bufferInd = 0; bufferInd < bufferCnt; bufferInd++) {
    // determine the starting/ending indices of the selected data elements
    int dataIndBeg = dataCntInBuf * bufferInd;
    int dataIndEnd = std::min(dataCnt, dataIndBeg + dataCntInBuf) - 1;
    int dataCntSel = dataIndEnd - dataIndBeg + 1;

    // reset all elements in the buffer to zeros
    memset(bufferArray, 0, sizeof(uint8_t) * kBufferLen);

    // copy data to the buffer
    const T* dataVec = dataLst.GetDataPtr() + dataIndBeg;
    uint8_t* pBuffer = bufferArray;
    int bitCntLeft = kBitCntPerBuf;
    for (int dataIndSel = 0; dataIndSel < dataCntSel; dataIndSel++) {
      if (bitCntLeft >= bitCntPerEle) {
        bitCntLeft -= bitCntPerEle;
        *pBuffer |= ((dataVec[dataIndSel] - 1) << bitCntLeft);
      } else {
        *(pBuffer++) |=
            ((dataVec[dataIndSel] - 1) >> (bitCntPerEle - bitCntLeft));
        bitCntLeft = kBitCntPerBuf - (bitCntPerEle - bitCntLeft);
        *pBuffer |= ((dataVec[dataIndSel] - 1) << bitCntLeft);
      }  // ENDIF: bitCntLeft
    }  // ENDFOR: dataIndSel

    // write the buffer to the file
    rtnVal = fwrite(bufferArray, sizeof(uint8_t), kBufferLen, outFile);
    assert(rtnVal == kBufferLen);
  }  // ENDFOR: bufferInd
  dataLst.DispSizInfo();
  printf("[INFO] writing data matrix from %s [COMPLETED]\n", filePath.c_str());

  // close file
  fclose(outFile);

  return true;
}

template<typename T>
bool FileIO::WriteTxtFile(
    const std::string& filePath, const Matrix<T>& dataLst) {
  // declare auxiliary variables
  int32_t dimCnt;
  TypeInfo typeInfo;

  // open file
  FILE* outFile = fopen(filePath.c_str(), "w");
  if (outFile == nullptr) {
    printf("[ERROR] could not open file at %s\n", filePath.c_str());
    return false;
  }  // ENDIF: outFile

  // write basic variables
  dimCnt = dataLst.GetDimCnt();
  fprintf(outFile, "%d", dimCnt);
  for (int dimInd = 0; dimInd < dimCnt; dimInd++) {
    fprintf(outFile, " %d", dataLst.GetDimLen(dimInd));
  }  // ENDFOR: dimInd
  fprintf(outFile, "\n");

  // write data matrix to file
  printf("[INFO] writing data matrix to %s\n", filePath.c_str());
  GetTypeInfo(dataLst, &typeInfo);
  int dataCnt = dataLst.GetEleCnt();
  int dimLenLast = dataLst.GetDimLen(dimCnt - 1);
  const T* dataPtr = dataLst.GetDataPtr();
  for (int dataInd = 0; dataInd < dataCnt; dataInd++, dataPtr++) {
    fprintf(outFile, typeInfo.outputFrmtStr.c_str(), *dataPtr);
    fprintf(outFile, "%s", ((dataInd + 1) % dimLenLast != 0) ? " " : "\n");
  }  // ENDFOR: dataInd
  dataLst.DispSizInfo();
  printf("[INFO] writing data matrix to %s [COMPLETED]\n", filePath.c_str());

  // close file
  fclose(outFile);

  return true;
}

template<typename T>
void FileIO::GetTypeInfo(const Matrix<T>& dataLst, TypeInfo* pTypeInfo) {
  // declare auxiliary variables to obtain the type name
  static uint8_t valUInt8;
  static int8_t valInt8;
  static uint16_t valUInt16;
  static int16_t valInt16;
  static uint32_t valUInt32;
  static int32_t valInt32;
  static float valFlt;
  static double valDbl;

  // obtain the name of the input type
  std::string tName = typeid(T).name();

  // compare the type name with known ones
  if (tName == typeid(valUInt8).name()) {
    pTypeInfo->typeName = "uint8_t";
    pTypeInfo->inputFrmtStr = "%hhu";
    pTypeInfo->outputFrmtStr = "%hhu";
  } else if (tName == typeid(valInt8).name()) {
    pTypeInfo->typeName = "int8_t";
    pTypeInfo->inputFrmtStr = "%hhd";
    pTypeInfo->outputFrmtStr = "%hhd";
  } else if (tName == typeid(valUInt16).name()) {
    pTypeInfo->typeName = "uint16_t";
    pTypeInfo->inputFrmtStr = "%hu";
    pTypeInfo->outputFrmtStr = "%hu";
  } else if (tName == typeid(valInt16).name()) {
    pTypeInfo->typeName = "int16_t";
    pTypeInfo->inputFrmtStr = "%hd";
    pTypeInfo->outputFrmtStr = "%hd";
  } else if (tName == typeid(valUInt32).name()) {
    pTypeInfo->typeName = "uint32_t";
    pTypeInfo->inputFrmtStr = "%u";
    pTypeInfo->outputFrmtStr = "%u";
  } else if (tName == typeid(valInt32).name()) {
    pTypeInfo->typeName = "int32_t";
    pTypeInfo->inputFrmtStr = "%d";
    pTypeInfo->outputFrmtStr = "%d";
  } else if (tName == typeid(valFlt).name()) {
    pTypeInfo->typeName = "float";
    pTypeInfo->inputFrmtStr = "%f";
    pTypeInfo->outputFrmtStr = "%.4f";
  } else if (tName == typeid(valDbl).name()) {
    pTypeInfo->typeName = "double";
    pTypeInfo->inputFrmtStr = "%lf";
    pTypeInfo->outputFrmtStr = "%.4f";
  } else {
    printf("[ERROR] unrecognized type name: %s\n", tName.c_str());
    return;
  }  // ENDIF: tName
}

#endif  // INCLUDE_FILEIO_H_
