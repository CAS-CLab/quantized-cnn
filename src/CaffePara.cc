/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#include "../include/CaffePara.h"

#include "../include/Common.h"
#include "../include/FileIO.h"

void CaffePara::Init(
    const std::string& dirPathSrc, const std::string& filePfxSrc) {
  // set-up basic parameters
  dirPath = dirPathSrc;
  filePfx = filePfxSrc;
}

void CaffePara::ConfigLayer_AlexNet(void) {
  // configure the basic information
  layerCnt = 23;
  imgChnIn = 3;
  imgHeiIn = 227;
  imgWidIn = 227;
  layerInfoLst.resize(layerCnt);

  // configure each layer
  ConfigConvLayer(&(layerInfoLst[ 0]), 0, 11, 96, 1, 4);
  ConfigReLuLayer(&(layerInfoLst[ 1]));
  ConfigLoRNLayer(&(layerInfoLst[ 2]), 5, 0.0001, 0.75, 1.0);
  ConfigPoolLayer(&(layerInfoLst[ 3]), 0, 3, 2);
  ConfigConvLayer(&(layerInfoLst[ 4]), 2, 5, 256, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[ 5]));
  ConfigLoRNLayer(&(layerInfoLst[ 6]), 5, 0.0001, 0.75, 1.0);
  ConfigPoolLayer(&(layerInfoLst[ 7]), 0, 3, 2);
  ConfigConvLayer(&(layerInfoLst[ 8]), 1, 3, 384, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[ 9]));
  ConfigConvLayer(&(layerInfoLst[10]), 1, 3, 384, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[11]));
  ConfigConvLayer(&(layerInfoLst[12]), 1, 3, 256, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[13]));
  ConfigPoolLayer(&(layerInfoLst[14]), 0, 3, 2);
  ConfigFCntLayer(&(layerInfoLst[15]), 4096);
  ConfigReLuLayer(&(layerInfoLst[16]));
  ConfigDrptLayer(&(layerInfoLst[17]), 0.50);
  ConfigFCntLayer(&(layerInfoLst[18]), 4096);
  ConfigReLuLayer(&(layerInfoLst[19]));
  ConfigDrptLayer(&(layerInfoLst[20]), 0.50);
  ConfigFCntLayer(&(layerInfoLst[21]), 1000);
  ConfigSMaxLayer(&(layerInfoLst[22]));
}

void CaffePara::ConfigLayer_CaffeNet(void) {
  // configure the basic information
  layerCnt = 23;
  imgChnIn = 3;
  imgHeiIn = 227;
  imgWidIn = 227;
  layerInfoLst.resize(layerCnt);

  // configure each layer
  ConfigConvLayer(&(layerInfoLst[ 0]), 0, 11, 96, 1, 4);
  ConfigReLuLayer(&(layerInfoLst[ 1]));
  ConfigPoolLayer(&(layerInfoLst[ 2]), 0, 3, 2);
  ConfigLoRNLayer(&(layerInfoLst[ 3]), 5, 0.0001, 0.75, 1.0);
  ConfigConvLayer(&(layerInfoLst[ 4]), 2, 5, 256, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[ 5]));
  ConfigPoolLayer(&(layerInfoLst[ 6]), 0, 3, 2);
  ConfigLoRNLayer(&(layerInfoLst[ 7]), 5, 0.0001, 0.75, 1.0);
  ConfigConvLayer(&(layerInfoLst[ 8]), 1, 3, 384, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[ 9]));
  ConfigConvLayer(&(layerInfoLst[10]), 1, 3, 384, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[11]));
  ConfigConvLayer(&(layerInfoLst[12]), 1, 3, 256, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[13]));
  ConfigPoolLayer(&(layerInfoLst[14]), 0, 3, 2);
  ConfigFCntLayer(&(layerInfoLst[15]), 4096);
  ConfigReLuLayer(&(layerInfoLst[16]));
  ConfigDrptLayer(&(layerInfoLst[17]), 0.50);
  ConfigFCntLayer(&(layerInfoLst[18]), 4096);
  ConfigReLuLayer(&(layerInfoLst[19]));
  ConfigDrptLayer(&(layerInfoLst[20]), 0.50);
  ConfigFCntLayer(&(layerInfoLst[21]), 1000);
  ConfigSMaxLayer(&(layerInfoLst[22]));
}

void CaffePara::ConfigLayer_VggCnnS(void) {
  // configure the basic information
  layerCnt = 22;
  imgChnIn = 3;
  imgHeiIn = 224;
  imgWidIn = 224;
  layerInfoLst.resize(layerCnt);

  // configure each layer
  ConfigConvLayer(&(layerInfoLst[ 0]), 0, 7, 96, 1, 2);
  ConfigReLuLayer(&(layerInfoLst[ 1]));
  ConfigLoRNLayer(&(layerInfoLst[ 2]), 5, 0.0005, 0.75, 2.0);
  ConfigPoolLayer(&(layerInfoLst[ 3]), 0, 3, 3);
  ConfigConvLayer(&(layerInfoLst[ 4]), 1, 5, 256, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[ 5]));
  ConfigPoolLayer(&(layerInfoLst[ 6]), 0, 2, 2);
  ConfigConvLayer(&(layerInfoLst[ 7]), 1, 3, 512, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[ 8]));
  ConfigConvLayer(&(layerInfoLst[ 9]), 1, 3, 512, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[10]));
  ConfigConvLayer(&(layerInfoLst[11]), 1, 3, 512, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[12]));
  ConfigPoolLayer(&(layerInfoLst[13]), 0, 3, 3);
  ConfigFCntLayer(&(layerInfoLst[14]), 4096);
  ConfigReLuLayer(&(layerInfoLst[15]));
  ConfigDrptLayer(&(layerInfoLst[16]), 0.50);
  ConfigFCntLayer(&(layerInfoLst[17]), 4096);
  ConfigReLuLayer(&(layerInfoLst[18]));
  ConfigDrptLayer(&(layerInfoLst[19]), 0.50);
  ConfigFCntLayer(&(layerInfoLst[20]), 1000);
  ConfigSMaxLayer(&(layerInfoLst[21]));
}

void CaffePara::ConfigLayer_VGG16(void) {
  // configure the basic information
  layerCnt = 39;
  imgChnIn = 3;
  imgHeiIn = 224;
  imgWidIn = 224;
  layerInfoLst.resize(layerCnt);

  // configure each layer
  ConfigConvLayer(&(layerInfoLst[ 0]), 1, 3, 64, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[ 1]));
  ConfigConvLayer(&(layerInfoLst[ 2]), 1, 3, 64, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[ 3]));
  ConfigPoolLayer(&(layerInfoLst[ 4]), 0, 2, 2);
  ConfigConvLayer(&(layerInfoLst[ 5]), 1, 3, 128, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[ 6]));
  ConfigConvLayer(&(layerInfoLst[ 7]), 1, 3, 128, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[ 8]));
  ConfigPoolLayer(&(layerInfoLst[ 9]), 0, 2, 2);
  ConfigConvLayer(&(layerInfoLst[10]), 1, 3, 256, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[11]));
  ConfigConvLayer(&(layerInfoLst[12]), 1, 3, 256, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[13]));
  ConfigConvLayer(&(layerInfoLst[14]), 1, 3, 256, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[15]));
  ConfigPoolLayer(&(layerInfoLst[16]), 0, 2, 2);
  ConfigConvLayer(&(layerInfoLst[17]), 1, 3, 512, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[18]));
  ConfigConvLayer(&(layerInfoLst[19]), 1, 3, 512, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[20]));
  ConfigConvLayer(&(layerInfoLst[21]), 1, 3, 512, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[22]));
  ConfigPoolLayer(&(layerInfoLst[23]), 0, 2, 2);
  ConfigConvLayer(&(layerInfoLst[24]), 1, 3, 512, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[25]));
  ConfigConvLayer(&(layerInfoLst[26]), 1, 3, 512, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[27]));
  ConfigConvLayer(&(layerInfoLst[28]), 1, 3, 512, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[29]));
  ConfigPoolLayer(&(layerInfoLst[30]), 0, 2, 2);
  ConfigFCntLayer(&(layerInfoLst[31]), 4096);
  ConfigReLuLayer(&(layerInfoLst[32]));
  ConfigDrptLayer(&(layerInfoLst[33]), 0.50);
  ConfigFCntLayer(&(layerInfoLst[34]), 4096);
  ConfigReLuLayer(&(layerInfoLst[35]));
  ConfigDrptLayer(&(layerInfoLst[36]), 0.50);
  ConfigFCntLayer(&(layerInfoLst[37]), 1000);
  ConfigSMaxLayer(&(layerInfoLst[38]));
}

void CaffePara::ConfigLayer_CaffeNetFGB(void) {
  // configure the basic information
  layerCnt = 23;
  imgChnIn = 3;
  imgHeiIn = 227;
  imgWidIn = 227;
  layerInfoLst.resize(layerCnt);

  // configure each layer
  ConfigConvLayer(&(layerInfoLst[ 0]), 0, 11, 96, 1, 4);
  ConfigReLuLayer(&(layerInfoLst[ 1]));
  ConfigPoolLayer(&(layerInfoLst[ 2]), 0, 3, 2);
  ConfigLoRNLayer(&(layerInfoLst[ 3]), 5, 0.0001, 0.75, 1.0);
  ConfigConvLayer(&(layerInfoLst[ 4]), 2, 5, 256, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[ 5]));
  ConfigPoolLayer(&(layerInfoLst[ 6]), 0, 3, 2);
  ConfigLoRNLayer(&(layerInfoLst[ 7]), 5, 0.0001, 0.75, 1.0);
  ConfigConvLayer(&(layerInfoLst[ 8]), 1, 3, 384, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[ 9]));
  ConfigConvLayer(&(layerInfoLst[10]), 1, 3, 384, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[11]));
  ConfigConvLayer(&(layerInfoLst[12]), 1, 3, 256, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[13]));
  ConfigPoolLayer(&(layerInfoLst[14]), 0, 3, 2);
  ConfigFCntLayer(&(layerInfoLst[15]), 4096);
  ConfigReLuLayer(&(layerInfoLst[16]));
  ConfigDrptLayer(&(layerInfoLst[17]), 0.70);
  ConfigFCntLayer(&(layerInfoLst[18]), 4096);
  ConfigReLuLayer(&(layerInfoLst[19]));
  ConfigDrptLayer(&(layerInfoLst[20]), 0.70);
  ConfigFCntLayer(&(layerInfoLst[21]), 518);
  ConfigSMaxLayer(&(layerInfoLst[22]));
}

void CaffePara::ConfigLayer_CaffeNetFGD(void) {
  // configure the basic information
  layerCnt = 23;
  imgChnIn = 3;
  imgHeiIn = 227;
  imgWidIn = 227;
  layerInfoLst.resize(layerCnt);

  // configure each layer
  ConfigConvLayer(&(layerInfoLst[ 0]), 0, 11, 96, 1, 4);
  ConfigReLuLayer(&(layerInfoLst[ 1]));
  ConfigPoolLayer(&(layerInfoLst[ 2]), 0, 3, 2);
  ConfigLoRNLayer(&(layerInfoLst[ 3]), 5, 0.0001, 0.75, 1.0);
  ConfigConvLayer(&(layerInfoLst[ 4]), 2, 5, 256, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[ 5]));
  ConfigPoolLayer(&(layerInfoLst[ 6]), 0, 3, 2);
  ConfigLoRNLayer(&(layerInfoLst[ 7]), 5, 0.0001, 0.75, 1.0);
  ConfigConvLayer(&(layerInfoLst[ 8]), 1, 3, 384, 1, 1);
  ConfigReLuLayer(&(layerInfoLst[ 9]));
  ConfigConvLayer(&(layerInfoLst[10]), 1, 3, 384, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[11]));
  ConfigConvLayer(&(layerInfoLst[12]), 1, 3, 256, 2, 1);
  ConfigReLuLayer(&(layerInfoLst[13]));
  ConfigPoolLayer(&(layerInfoLst[14]), 0, 3, 2);
  ConfigFCntLayer(&(layerInfoLst[15]), 4096);
  ConfigReLuLayer(&(layerInfoLst[16]));
  ConfigDrptLayer(&(layerInfoLst[17]), 0.50);
  ConfigFCntLayer(&(layerInfoLst[18]), 4096);
  ConfigReLuLayer(&(layerInfoLst[19]));
  ConfigDrptLayer(&(layerInfoLst[20]), 0.50);
  ConfigFCntLayer(&(layerInfoLst[21]), 200);
  ConfigSMaxLayer(&(layerInfoLst[22]));
}

bool CaffePara::LoadLayerPara(const bool enblAprx, const ENUM_AsmtEnc asmtEnc) {
  // declare auxiliary variables
  const int kStrBufLen = 256;
  char strBuf[kStrBufLen];
  bool succFlg = true;

  // NOTE
  // <succFlg> will be updated when attempting loading a file from disk.
  // If <succFlg> is set to <false> at sometime, the remaining file loading
  // operations may be skipped by the compiler. However, this won't be an
  // problem, since the function will eventually return <succFlg>, which is
  // <false>, to indicate that the file loading has failed at some point.

  // load parameters for each layer
  layerParaLst.resize(layerCnt);
  for (int layerInd = 0; layerInd < layerCnt; layerInd++) {
    const LayerInfo& layerInfo = layerInfoLst[layerInd];
    LayerPara& layerPara = layerParaLst[layerInd];

    // load parameters for the convolutional (or fully-connected) layer
    if ((layerInfo.type == ENUM_LyrType::Conv) ||
        (layerInfo.type == ENUM_LyrType::FCnt)) {
      // load the bias vector
      snprintf(strBuf, kStrBufLen, "%s/%s.biasVec.%02d.bin",
          dirPath.c_str(), filePfx.c_str(), layerInd + 1);
      succFlg &= FileIO::ReadBinFile(strBuf, &(layerPara.biasVec));

      // load convolutional kernels or fully-connected weighting matrix
      if (enblAprx) {  // load quantized parameters
        // load sub-codebooks
        snprintf(strBuf, kStrBufLen, "%s/%s.ctrdLst.%02d.bin",
            dirPath.c_str(), filePfx.c_str(), layerInd + 1);
        succFlg &= FileIO::ReadBinFile(strBuf, &(layerPara.ctrdLst));

        // load sub-codewords' assignment
        if (asmtEnc == ENUM_AsmtEnc::Raw) {
          snprintf(strBuf, kStrBufLen, "%s/%s.asmtLst.%02d.bin",
              dirPath.c_str(), filePfx.c_str(), layerInd + 1);
          succFlg &= FileIO::ReadBinFile(strBuf, &(layerPara.asmtLst));
        } else {
          snprintf(strBuf, kStrBufLen, "%s/%s.asmtLst.%02d.cbn",
              dirPath.c_str(), filePfx.c_str(), layerInd + 1);
          succFlg &= FileIO::ReadCbnFile(strBuf, &(layerPara.asmtLst));
        }  // ENDIF: asmtEnc

        // fix the 1/0 index difference between MATLAB and C++
        uint8_t* asmtVec = layerPara.asmtLst.GetDataPtr();
        for (int eleInd = 0; eleInd < layerPara.asmtLst.GetEleCnt(); eleInd++) {
          asmtVec[eleInd]--;
        }  // ENDFOR: eleInd
      } else {  // load original parameters
        if (layerInfo.type == ENUM_LyrType::Conv) {
          // load the convolutional kernels
          snprintf(strBuf, kStrBufLen, "%s/%s.convKnl.%02d.bin",
              dirPath.c_str(), filePfx.c_str(), layerInd + 1);
          succFlg &= FileIO::ReadBinFile(strBuf, &(layerPara.convKnlLst));
        } else {
          // load the fully-connected weighting matrix
          snprintf(strBuf, kStrBufLen, "%s/%s.fcntWei.%02d.bin",
              dirPath.c_str(), filePfx.c_str(), layerInd + 1);
          succFlg &= FileIO::ReadBinFile(strBuf, &(layerPara.fcntWeiMat));
        }  // ENDIF: layerInfo
      }  // ENDIF: enblAprx
    }  // ENDIF: layerInfo
  }  // ENDFOR: layerInd

  return succFlg;
}

bool CaffePara::CvtAsmtEnc(
    const ENUM_AsmtEnc asmtEncSrc, const ENUM_AsmtEnc asmtEncDst) {
  // early return if the source/target encoding is the same
  if (asmtEncSrc == asmtEncDst) {
    printf("[INFO] no encoding conversion is required\n");
    return true;
  }  // ENDIF: asmtEncSrc

  // declare auxiliary variables
  const int kStrBufLen = 256;
  char strBufBin[kStrBufLen];
  char strBufCbn[kStrBufLen];
  bool succFlg = true;
  Matrix<uint8_t> asmtLst;
  int bitCntPerEle;

  // NOTE
  // <succFlg> will be updated when attempting loading a file from disk.
  // If <succFlg> is set to <false> at sometime, the remaining file loading
  // operations may be skipped by the compiler. However, this won't be an
  // problem, since the function will eventually return <succFlg>, which is
  // <false>, to indicate that the file loading has failed at some point.

  // convert assignment encoding for each layer
  for (int layerInd = 0; layerInd < layerCnt; layerInd++) {
    const LayerInfo& layerInfo = layerInfoLst[layerInd];

    // convert assignment encoding for the convolutional layer
    if ((layerInfo.type == ENUM_LyrType::Conv) ||
        (layerInfo.type == ENUM_LyrType::FCnt)) {
      // generate file path for the assignment data
      snprintf(strBufBin, kStrBufLen, "%s/%s.asmtLst.%02d.bin",
          dirPath.c_str(), filePfx.c_str(), layerInd + 1);
      snprintf(strBufCbn, kStrBufLen, "%s/%s.asmtLst.%02d.cbn",
          dirPath.c_str(), filePfx.c_str(), layerInd + 1);

      // convert the encoding from <Raw> to <Compact>, or vice versa
      if (asmtEncSrc == ENUM_AsmtEnc::Raw) {
        succFlg &= FileIO::ReadBinFile(strBufBin, &asmtLst);
        bitCntPerEle = CalcBitCntPerEle(asmtLst);
        printf("layer #%d: bitCntPerEle = %d\n", layerInd + 1, bitCntPerEle);
        succFlg &= FileIO::WriteCbnFile(strBufCbn, asmtLst, bitCntPerEle);
      } else {
        succFlg &= FileIO::ReadCbnFile(strBufCbn, &asmtLst);
        succFlg &= FileIO::WriteBinFile(strBufBin, asmtLst);
      }  // ENDIF: asmtEncSrc
    }  // ENDIF: layerInfo
  }  // ENDFOR: layerInd

  return succFlg;
}

int CaffePara::CalcBitCntPerEle(const Matrix<uint8_t>& asmtLst) {
  // find the maximal value in <asmtLst>
  int eleCnt = asmtLst.GetEleCnt();
  const uint8_t* asmtVec = asmtLst.GetDataPtr();
  uint8_t maxVal = 0;
  for (int eleInd = 0; eleInd < eleCnt; eleInd++) {
    maxVal = std::max(maxVal, *(asmtVec++));
  }  // ENDFOR: eleInd
  maxVal -= 1;  // remove the offset

  // determine the proper value for <bitCntPerEle>
  int bitCntPerEle = 0;
  while (maxVal != 0) {
    maxVal /= 2;
    bitCntPerEle++;
  }  // ENDWHILE: maxVal

  return bitCntPerEle;
}

void CaffePara::ConfigConvLayer(LayerInfo* pLayerInfo, const int padSiz,
    const int knlSiz, const int knlCnt, const int grpCnt, const int stride) {
  pLayerInfo->type = ENUM_LyrType::Conv;
  pLayerInfo->padSiz = padSiz;
  pLayerInfo->knlSiz = knlSiz;
  pLayerInfo->knlCnt = knlCnt;
  pLayerInfo->grpCnt = grpCnt;
  pLayerInfo->stride = stride;
}

void CaffePara::ConfigPoolLayer(LayerInfo* pLayerInfo,
    const int padSiz, const int knlSiz, const int stride) {
  pLayerInfo->type = ENUM_LyrType::Pool;
  pLayerInfo->padSiz = padSiz;
  pLayerInfo->knlSiz = knlSiz;
  pLayerInfo->stride = stride;
}

void CaffePara::ConfigFCntLayer(LayerInfo* pLayerInfo, const int nodCnt) {
  pLayerInfo->type = ENUM_LyrType::FCnt;
  pLayerInfo->nodCnt = nodCnt;
}

void CaffePara::ConfigReLuLayer(LayerInfo* pLayerInfo) {
  pLayerInfo->type = ENUM_LyrType::ReLU;
}

void CaffePara::ConfigLoRNLayer(LayerInfo* pLayerInfo, const int lrnSiz,
    const float lrnAlp, const float lrnBet, const float lrnIni) {
  pLayerInfo->type = ENUM_LyrType::LoRN;
  pLayerInfo->lrnSiz = lrnSiz;
  pLayerInfo->lrnAlp = lrnAlp;
  pLayerInfo->lrnBet = lrnBet;
  pLayerInfo->lrnIni = lrnIni;
}

void CaffePara::ConfigDrptLayer(LayerInfo* pLayerInfo, const float drpRat) {
  pLayerInfo->type = ENUM_LyrType::Drpt;
  pLayerInfo->drpRat = drpRat;
}

void CaffePara::ConfigSMaxLayer(LayerInfo* pLayerInfo) {
  pLayerInfo->type = ENUM_LyrType::SMax;
}
