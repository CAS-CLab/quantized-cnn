/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#include "../include/CaffeEva.h"

#include <immintrin.h>

#include "../include/BlasWrapper.h"
#include "../include/CaffePara.h"
#include "../include/Common.h"
#include "../include/FileIO.h"

#ifdef ENBL_ANDROID_DBGR
  #include "share.h"
  #define SHR_PRINTF LOGI
#else
  #define SHR_PRINTF printf
#endif  // ENDIF: ENBL_ANDROID_DBG

// define MACRO for SSE-based table look-up operation
#define SSE_LOOKUP(pTbl, pIdx, vTbl, pos) { \
  vTbl = _mm256_set_ps(                     \
      pTbl[pIdx[pos + 7]],                  \
      pTbl[pIdx[pos + 6]],                  \
      pTbl[pIdx[pos + 5]],                  \
      pTbl[pIdx[pos + 4]],                  \
      pTbl[pIdx[pos + 3]],                  \
      pTbl[pIdx[pos + 2]],                  \
      pTbl[pIdx[pos + 1]],                  \
      pTbl[pIdx[pos + 0]]);                 \
}

// initialize constant variables
const int kDataCntInBatch = 1;  // number of images in each batch
const int kBatchCntProc = 100;  // number of batches
const int kLablCntPerData = 5;  // number of predicted labels per image

CaffeEva::~CaffeEva(void) {
  // release dynamically allocated memory
  delete[] featMapLst;
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++) {
    FeatBufStrLst& featBufStrLst = featBufStrMat[layerInd];
    for (std::size_t bufInd = 0; bufInd < featBufStrLst.size(); bufInd++) {
      // skip if no memory is allocated to the current buffer
      if (featBufStrLst[bufInd].pFeatBuf != nullptr) {
        delete featBufStrLst[bufInd].pFeatBuf;
        featBufStrLst[bufInd].pFeatBuf = nullptr;
      }  // ENDIF: featBufStrLst
    }  // ENDFOR: bufInd
  }  // ENDFOR: layerInd

  // destory objects: <caffeParaObj>
}

void CaffeEva::Init(const bool enblAprxSrc) {
  // initialize <enblAprx>
  enblAprx = enblAprxSrc;

  // initialize all stop-watches
  swAllLayers.Reset();
  swConvLayer.Reset();
  swPoolLayer.Reset();
  swFCntLayer.Reset();
  swReLuLayer.Reset();
  swLoRNLayer.Reset();
  swDrptLayer.Reset();
  swSMaxLayer.Reset();
  swCompLkupTblConv.Reset();
  swEstiInPdValConv.Reset();
  swCompLkupTblFCnt.Reset();
  swEstiInPdValFCnt.Reset();
  swDebugTimePri.Reset();
  swDebugTimeSec.Reset();
}

void CaffeEva::SetModelName(const std::string& modelNameSrc) {
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::SetDirPath()\n");

  // specify the model name
  modelName = modelNameSrc;
}

void CaffeEva::SetModelPath(
    const std::string& dirPathMainSrc, const std::string& fileNamePfxSrc) {
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::SetDirPath()\n");

  // specify the main directory path and file name prefix
  dirPathMain = dirPathMainSrc;
  fileNamePfx = fileNamePfxSrc;
}

bool CaffeEva::LoadDataset(const std::string& dirPathData) {
  // declare auxiliary variables
  const int kStrBufLen = 256;
  char strBuf[kStrBufLen];
  bool succFlg;

  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::LoadDataset()\n");

  // load samples in the evaluation subset
  snprintf(strBuf, kStrBufLen, "%s/dataMatTst.single.bin", dirPathData.c_str());
  succFlg = FileIO::ReadBinFile(strBuf, &dataLst);
  if (!succFlg) {  // failed
    return false;
  }  // ENDIF: succFlg

  // load samples' ground-truth labels in the evaluation subset
  snprintf(strBuf, kStrBufLen, "%s/lablVecTst.uint16.bin", dirPathData.c_str());
  succFlg = FileIO::ReadBinFile(strBuf, &lablVecGrth);
  if (!succFlg) {  // failed
    return false;
  }  // ENDIF: succFlg

  return true;
}

bool CaffeEva::LoadCaffePara(void) {
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::LoadCaffePara()\n");

  // initialize <caffeParaObj>
  caffeParaObj.Init(dirPathMain, fileNamePfx);

  // load each layer's basic information
  if (modelName == "AlexNet") {
    caffeParaObj.ConfigLayer_AlexNet();
  } else if (modelName == "CaffeNet") {
    caffeParaObj.ConfigLayer_CaffeNet();
  } else if (modelName == "VggCnnS") {
    caffeParaObj.ConfigLayer_VggCnnS();
  } else if (modelName == "VGG16") {
    caffeParaObj.ConfigLayer_VGG16();
  } else if (modelName == "CaffeNetFGB") {
    caffeParaObj.ConfigLayer_CaffeNetFGB();
  } else if (modelName == "CaffeNetFGD") {
    caffeParaObj.ConfigLayer_CaffeNetFGD();
  } else {
    printf("[ERROR] unrecognized caffe model name: %s\n", modelName.c_str());
    return false;
  }  // ENDIF: modelName

  // load each layer's detailed parameters
  bool succFlg = caffeParaObj.LoadLayerPara(enblAprx, ENUM_AsmtEnc::Compact);
  if (!succFlg) {  // failed
    return false;
  }  // ENDIF: succFlg

  // prepare feature map and buffers for each layer
  PrepFeatMap();
  PrepFeatBuf();
  if (enblAprx) {
    PrepCtrdBuf();
    PrepAsmtBuf();
  }  // ENDIF: enblAprx

  return true;
}

void CaffeEva::ExecForwardPass(void) {
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::ExecForwardPass()\n");

  // initialize stop-watches for each layer
  swIndvLayerLst.resize(caffeParaObj.layerCnt);
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++) {
    swIndvLayerLst[layerInd].Reset();
  }  // ENDFOR: layerInd

  // pack samples into batches and then execute the forward pass
  int dataCnt = dataLst.GetDimLen(0);
  int dataIndL;
  int dataIndU;
  int batchCnt = (dataCnt + kDataCntInBatch - 1) / kDataCntInBatch;
  lablVecPred.Create(dataCnt, kLablCntPerData, 1, 1);
  for (int batchInd = 0; batchInd < kBatchCntProc; batchInd++) {
    printf("processing the %d-th batch\n", batchInd + 1);

    // check whether is the last batch
    if (batchInd < batchCnt - 1) {
      dataIndL = kDataCntInBatch * batchInd;
      dataIndU = dataIndL + (kDataCntInBatch - 1);
    } else {
      dataIndU = dataCnt - 1;
      dataIndL = dataIndU - (kDataCntInBatch - 1);
    }  // ENDIF: batchInd

    // convert samples' feature vectors into the input feature map
    CvtDataLstToFeatMap(dataIndL, dataIndU, dataLst, &(featMapLst[0]));

    // execute the forward pass
    bool isFirstFCnt = true;
    for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++) {
      // permute dimensions for the first fully-connected layer
      const LayerInfo& layerInfo = caffeParaObj.layerInfoLst[layerInd];
      if (isFirstFCnt && (layerInfo.type == ENUM_LyrType::FCnt)) {
        featMapLst[layerInd].Permute(0, 3, 1, 2);
      }  // ENDIF: isFirstFCnt

      // compute the target layer's activation
      swIndvLayerLst[layerInd].Resume();
      CalcFeatMap(featMapLst[layerInd], layerInd, &(featMapLst[layerInd + 1]));
      swIndvLayerLst[layerInd].Pause();

      // permute dimensions for the first fully-connected layer
      if (isFirstFCnt && (layerInfo.type == ENUM_LyrType::FCnt)) {
        isFirstFCnt = false;
        int m = featMapLst[layerInd].GetDimLen(0);
        int n = featMapLst[layerInd].GetDimLen(1);
        int p = featMapLst[layerInd].GetDimLen(2);
        int q = featMapLst[layerInd].GetDimLen(3);
        featMapLst[layerInd].Resize(m, p, q, n);
      }  // ENDIF: isFirstFCnt
    }  // ENDIF: layerInd

    // convert the output feature map into samples' predicted labels
    CvtFeatMapToLablVec(dataIndL,
        dataIndU, featMapLst[caffeParaObj.layerCnt], &lablVecPred);
  }  // ENDFOR: batchInd
}

void CaffeEva::ExecForwardPass(
    const Matrix<float>& imgDataIn, Matrix<float>* pProbVecOut) {
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::ExecForwardPass()\n");

  // initialize stop-watches for each layer
  swIndvLayerLst.resize(caffeParaObj.layerCnt);
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++) {
    swIndvLayerLst[layerInd].Reset();
  }  // ENDFOR: layerInd

  // copy <dataLstIn> to the input feature map
  featMapLst[0].Permute(0, 3, 1, 2);
  memcpy(featMapLst[0].GetDataPtr(),
      imgDataIn.GetDataPtr(), sizeof(float) * imgDataIn.GetEleCnt());
  featMapLst[0].Permute(0, 2, 3, 1);

  // execute the forward pass
  bool isFirstFCnt = true;
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++) {
    printf("layerInd = %d\n", layerInd);
    // permute dimensions for the first fully-connected layer
    const LayerInfo& layerInfo = caffeParaObj.layerInfoLst[layerInd];
    if (isFirstFCnt && (layerInfo.type == ENUM_LyrType::FCnt)) {
      featMapLst[layerInd].Permute(0, 3, 1, 2);
    }  // ENDIF: isFirstFCnt

    // compute the target layer's activation
    swIndvLayerLst[layerInd].Resume();
    CalcFeatMap(featMapLst[layerInd], layerInd, &(featMapLst[layerInd + 1]));
    swIndvLayerLst[layerInd].Pause();

    // permute dimensions for the first fully-connected layer
    if (isFirstFCnt && (layerInfo.type == ENUM_LyrType::FCnt)) {
      isFirstFCnt = false;
      int m = featMapLst[layerInd].GetDimLen(0);
      int n = featMapLst[layerInd].GetDimLen(1);
      int p = featMapLst[layerInd].GetDimLen(2);
      int q = featMapLst[layerInd].GetDimLen(3);
      featMapLst[layerInd].Resize(m, p, q, n);
    }  // ENDIF: isFirstFCnt
  }  // ENDIF: layerInd

  // extract <dataLstOut> from the output feature map
  pProbVecOut->Resize(featMapLst[caffeParaObj.layerCnt].GetEleCnt());
  memcpy(pProbVecOut->GetDataPtr(),
      featMapLst[caffeParaObj.layerCnt].GetDataPtr(),
      sizeof(float) * pProbVecOut->GetEleCnt());
}

void CaffeEva::CalcPredAccu(void) {
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::CalcPredAccu()\n");

  // initialize counters for accuracy computation
  Matrix<uint32_t> accuCntLst(kLablCntPerData);
  Matrix<float> accuScrLst(kLablCntPerData);
  memset(accuCntLst.GetDataPtr(), 0, sizeof(uint32_t) * accuCntLst.GetEleCnt());
  memset(accuScrLst.GetDataPtr(), 0, sizeof(float) * accuScrLst.GetEleCnt());

  // compute the total number of correctly predicted class labels
  int dataCnt = kDataCntInBatch * kBatchCntProc;
  const uint16_t* lablPtrGrth = lablVecGrth.GetDataPtr();
  const uint16_t* lablPtrPred = lablVecPred.GetDataPtr();
  uint32_t* accuCntVec = accuCntLst.GetDataPtr();
  float* accuScrVec = accuScrLst.GetDataPtr();
  for (int dataInd = 0; dataInd < dataCnt; dataInd++) {
    for (int lablInd = 0; lablInd < kLablCntPerData; lablInd++) {
      uint16_t lablVal_Pred = lablPtrPred[dataInd * kLablCntPerData + lablInd];
      if (lablPtrGrth[dataInd] == lablVal_Pred) {
        accuCntVec[lablInd]++;
      }  // ENDIF: lablPtrGrth
    }  // ENDFOR: lablInd
  }  // ENDFOR: dataInd
  for (int lablInd = 1; lablInd < kLablCntPerData; lablInd++) {
    accuCntVec[lablInd] += accuCntVec[lablInd - 1];
  }  // ENDFOR: lablInd
  for (int lablInd = 0; lablInd < kLablCntPerData; lablInd++) {
    accuScrVec[lablInd] = static_cast<double>(accuCntVec[lablInd]) / dataCnt;
    printf("ACCURACY@%d: %d, %.2f%%\n",
        lablInd + 1, accuCntVec[lablInd], accuScrVec[lablInd] * 100);
  }  // ENDFOR: lablInd
}

float CaffeEva::DispElpsTime(void) {
  // get total computation time
  float timeTotal = swAllLayers.GetTime();

  // display the elapsed time of each stop-watch
  SHR_PRINTF("swAllLayers: %.4f (s)\n", timeTotal);
  SHR_PRINTF("swConvLayer: %.4f (s)\n", swConvLayer.GetTime());
  SHR_PRINTF("swPoolLayer: %.4f (s)\n", swPoolLayer.GetTime());
  SHR_PRINTF("swFCntLayer: %.4f (s)\n", swFCntLayer.GetTime());
  SHR_PRINTF("swReLuLayer: %.4f (s)\n", swReLuLayer.GetTime());
  SHR_PRINTF("swLoRNLayer: %.4f (s)\n", swLoRNLayer.GetTime());
  SHR_PRINTF("swDrptLayer: %.4f (s)\n", swDrptLayer.GetTime());
  SHR_PRINTF("swSMaxLayer: %.4f (s)\n", swSMaxLayer.GetTime());
  SHR_PRINTF("swCompLkupTblConv: %.4f (s)\n", swCompLkupTblConv.GetTime());
  SHR_PRINTF("swEstiInPdValConv: %.4f (s)\n", swEstiInPdValConv.GetTime());
  SHR_PRINTF("swCompLkupTblFCnt: %.4f (s)\n", swCompLkupTblFCnt.GetTime());
  SHR_PRINTF("swEstiInPdValFCnt: %.4f (s)\n", swEstiInPdValFCnt.GetTime());
  SHR_PRINTF("swDebugTimePri: %.4f (s)\n", swDebugTimePri.GetTime());
  SHR_PRINTF("swDebugTimeSec: %.4f (s)\n", swDebugTimeSec.GetTime());

  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++) {
    SHR_PRINTF("swIndvLayerLst #%2d: %.4f (s)\n",
        layerInd + 1, swIndvLayerLst[layerInd].GetTime());
  }  // ENDFOR: layerInd

  // re-initialize each stop-watch
  Init(enblAprx);

  return timeTotal;
}

void CaffeEva::PrepFeatMap(void) {
  // allocate memory space for <featMapSizLst>
  featMapSizLst.resize(caffeParaObj.layerCnt + 1);

  // determine the size of the input feature map
  featMapSizLst[0].dataCnt = kDataCntInBatch;
  featMapSizLst[0].imgHei = caffeParaObj.imgHeiIn;
  featMapSizLst[0].imgWid = caffeParaObj.imgWidIn;
  featMapSizLst[0].imgChn = caffeParaObj.imgChnIn;

  // determine the size of the remaining feature maps
  for (int layerInd = 1; layerInd <= caffeParaObj.layerCnt; layerInd++) {
    // obtain reference to previous/current feature map size
    const FeatMapSiz& featMapSizPrev = featMapSizLst[layerInd - 1];
    FeatMapSiz& featMapSizCurr = featMapSizLst[layerInd];
    const LayerInfo& layerInfo = caffeParaObj.layerInfoLst[layerInd - 1];

    // obtain basic variables
    int dataCnt = featMapSizPrev.dataCnt;
    int imgHeiPrev = featMapSizPrev.imgHei;
    int imgWidPrev = featMapSizPrev.imgWid;
    int imgChnPrev = featMapSizPrev.imgChn;
    int padSiz = layerInfo.padSiz;
    int knlSiz = layerInfo.knlSiz;
    int knlCnt = layerInfo.knlCnt;
    int stride = layerInfo.stride;
    double strideDbl = static_cast<double>(stride);
    int nodCnt = layerInfo.nodCnt;

    // compute the feature map size
    switch (layerInfo.type) {
    case ENUM_LyrType::Conv:
      featMapSizCurr.dataCnt = dataCnt;
      featMapSizCurr.imgHei = (imgHeiPrev + 2 * padSiz - knlSiz) / stride + 1;
      featMapSizCurr.imgWid = (imgWidPrev + 2 * padSiz - knlSiz) / stride + 1;
      featMapSizCurr.imgChn = knlCnt;
      break;
    case ENUM_LyrType::Pool:
      featMapSizCurr.dataCnt = dataCnt;
      featMapSizCurr.imgHei =
          ceil((imgHeiPrev + 2 * padSiz - knlSiz) / strideDbl) + 1;
      featMapSizCurr.imgWid =
          ceil((imgWidPrev + 2 * padSiz - knlSiz) / strideDbl) + 1;
      featMapSizCurr.imgChn = imgChnPrev;
      break;
    case ENUM_LyrType::FCnt:
      featMapSizCurr.dataCnt = dataCnt;
      featMapSizCurr.imgHei = 1;
      featMapSizCurr.imgWid = 1;
      featMapSizCurr.imgChn = nodCnt;
      break;
    case ENUM_LyrType::ReLU:
      // fall through
    case ENUM_LyrType::LoRN:
      // fall through
    case ENUM_LyrType::Drpt:
      // fall through
    case ENUM_LyrType::SMax:
      featMapSizCurr = featMapSizPrev;
      break;
    default:
      printf("[ERROR] invalid layer type\n");
      return;
    }  // ENDSWITCH: layerInfo
  }  // ENDFOR: layerInd

  // allocate memory for each feature map
  featMapLst = new Matrix<float>[caffeParaObj.layerCnt + 1];
  for (int layerInd = 0; layerInd <= caffeParaObj.layerCnt; layerInd++) {
    const FeatMapSiz& featMapSiz = featMapSizLst[layerInd];
    featMapLst[layerInd].Create(featMapSiz.dataCnt,
        featMapSiz.imgHei, featMapSiz.imgWid, featMapSiz.imgChn);
  }  // ENDFOR: layerInd

  // display the feature map size
  for (int layerInd = 0; layerInd <= caffeParaObj.layerCnt; layerInd++) {
    const FeatMapSiz& featMapSiz = featMapSizLst[layerInd];
    const Matrix<float>& featMap = featMapLst[layerInd];
    float memUsage = featMap.GetEleCnt() * 4 / 1024.0 / 1024.0;
    printf("layer #%2d: %4d x %4d x %4d x %4d (%6.2f MB)\n",
        layerInd, featMapSiz.dataCnt, featMapSiz.imgHei,
        featMapSiz.imgWid, featMapSiz.imgChn, memUsage);
  }  // ENDFOR: layerInd
}

void CaffeEva::PrepFeatBuf(void) {
  // define a template for <FeatBufStr>
  static FeatBufStr featBufStr;

  // determine the size of each layer's feature buffer
  featBufStrMat.resize(caffeParaObj.layerCnt);
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++) {
    // obtain reference to previous/current feature map size
    const FeatMapSiz& featMapSizCurr = featMapSizLst[layerInd];
    const FeatMapSiz& featMapSizNext = featMapSizLst[layerInd + 1];
    const LayerInfo& layerInfo = caffeParaObj.layerInfoLst[layerInd];
    const LayerPara& layerPara = caffeParaObj.layerParaLst[layerInd];
    FeatBufStrLst& featBufStrLst = featBufStrMat[layerInd];

    // obtain basic variables
    int dataCnt = featMapSizCurr.dataCnt;
    int imgHeiCurr = featMapSizCurr.imgHei;
    int imgWidCurr = featMapSizCurr.imgWid;
    int imgChnCurr = featMapSizCurr.imgChn;
    int imgHeiNext = featMapSizNext.imgHei;
    int imgWidNext = featMapSizNext.imgWid;
    int knlSiz = layerInfo.knlSiz;
    int knlCnt = layerInfo.knlCnt;
    int grpCnt = layerInfo.grpCnt;
    int lrnSiz = layerInfo.lrnSiz;
    int subSpaceCnt = layerPara.ctrdLst.GetDimLen(0);
    int ctrdCntPerSpace = layerPara.ctrdLst.GetDimLen(1);

    // compute the feature buffer size
    featBufStrLst.clear();
    switch (layerInfo.type) {
    case ENUM_LyrType::Conv:
      // feature buffer #0: <featMapSrcPrm>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::PrecComp,
          dataCnt, imgChnCurr, imgHeiCurr, imgWidCurr);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #1: <featMapSrcRsp>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::PrecComp,
          imgChnCurr / grpCnt * knlSiz * knlSiz, imgHeiNext * imgWidNext);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #2: <featMapDstRsp>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::PrecComp,
          knlCnt / grpCnt, imgHeiNext * imgWidNext);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #3: <featMapSrcPerGrp>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::AprxComp,
          dataCnt, imgHeiCurr, imgWidCurr, imgChnCurr / grpCnt);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #4: <inPdMat>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::AprxComp,
          dataCnt * imgHeiCurr * imgWidCurr, subSpaceCnt, ctrdCntPerSpace);
      featBufStrLst.push_back(featBufStr);
      break;
    case ENUM_LyrType::FCnt:
      // feature buffer #0: <featMapSrcRsp>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::AprxComp,
          dataCnt, imgChnCurr * imgHeiCurr * imgWidCurr);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #1: <inPdMat>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::AprxComp,
          dataCnt, subSpaceCnt, ctrdCntPerSpace);
      featBufStrLst.push_back(featBufStr);
      break;
    case ENUM_LyrType::LoRN:
      // feature buffer #0: <featVecSrcExt>
      InitFeatBuf(&featBufStr,
          ENUM_BufUsage::GnrlComp, imgChnCurr + lrnSiz - 1);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #1: <loclSumLst>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::GnrlComp, imgChnCurr);
      featBufStrLst.push_back(featBufStr);
      break;
    case ENUM_LyrType::Pool:
      // fall through
    case ENUM_LyrType::ReLU:
      // fall through
    case ENUM_LyrType::Drpt:
      // fall through
    case ENUM_LyrType::SMax:
      // do nothing
      break;
    default:
      printf("[ERROR] invalid layer type\n");
      return;
    }  // ENDSWITCH: layerInfo
  }  // ENDFOR: layerInd

  // display the feature buffer size
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++) {
    // obtain a constant reference to the feature buffer list
    FeatBufStrLst& featBufStrLst = featBufStrMat[layerInd];

    // display the feature buffer size
    printf("layer #%2d: \n", layerInd + 1);
    for (std::size_t bufInd = 0; bufInd < featBufStrLst.size(); bufInd++) {
      FeatBufStr& featBufStr = featBufStrLst[bufInd];

      // check whether current buffer will be used in the future
      if ((enblAprx && (featBufStr.usage == ENUM_BufUsage::PrecComp)) ||
          (!enblAprx && (featBufStr.usage == ENUM_BufUsage::AprxComp))) {
        continue;
      }  // ENDIF: enblAprx

      // allocate memory space for the current buffer
      featBufStr.pFeatBuf = new Matrix<float>();
      featBufStr.pFeatBuf->Create(featBufStr.dimCnt, featBufStr.dimLenLst);

      // display the memory consumption of the current buffer
      printf("  buffer #%lu: ", bufInd + 1);
      float memUsage = featBufStr.pFeatBuf->GetEleCnt() * 4 / 1024.0 / 1024.0;
      for (int dimInd = 0; dimInd < featBufStr.dimCnt; dimInd++) {
        if (dimInd < featBufStr.dimCnt - 1) {
          printf("%4d x ", featBufStr.dimLenLst[dimInd]);
        } else {
          printf("%4d (%6.2f MB)\n", featBufStr.dimLenLst[dimInd], memUsage);
        }
      }  // ENDFOR: dimInd
    }  // ENDFOR: bufInd
  }  // ENDFOR: layerInd
}

void CaffeEva::PrepCtrdBuf(void) {
  // determine the size of each layer's centroid buffer
  ctrdBufStrLst.resize(caffeParaObj.layerCnt);
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++) {
    // obtain reference to the current layer's parameters
    const LayerInfo& layerInfo = caffeParaObj.layerInfoLst[layerInd];
    const LayerPara& layerPara = caffeParaObj.layerParaLst[layerInd];

    // only allocate centroid buffer for the convolutional layers
    if ((layerInfo.type == ENUM_LyrType::Conv) ||
        (layerInfo.type == ENUM_LyrType::FCnt)) {
      // obtain basic variables
      int subSpaceCnt = layerPara.ctrdLst.GetDimLen(0);
      int ctrdCntPerSpace = layerPara.ctrdLst.GetDimLen(1);
      int featCntPerSpace = layerPara.ctrdLst.GetDimLen(2);

      // create the centroid buffer
      CtrdBufStr& ctrdBufStr = ctrdBufStrLst[layerInd];
      ctrdBufStr.dimCnt = 3;
      ctrdBufStr.dimLenLst[0] = subSpaceCnt;
      ctrdBufStr.dimLenLst[1] = featCntPerSpace;
      ctrdBufStr.dimLenLst[2] = ctrdCntPerSpace;
      ctrdBufStr.pCtrdBuf = new Matrix<float>(layerPara.ctrdLst);
      ctrdBufStr.pCtrdBuf->Permute(0, 2, 1);
    }  // ENDIF: layerInfo
  }  // ENDFOR: layerInd
}

void CaffeEva::PrepAsmtBuf(void) {
  // determine the size of each layer's assignment buffer
  asmtBufStrLst.resize(caffeParaObj.layerCnt);
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++) {
    // obtain reference to the current layer's parameters
    const LayerInfo& layerInfo = caffeParaObj.layerInfoLst[layerInd];
    const LayerPara& layerPara = caffeParaObj.layerParaLst[layerInd];

    // allocate assignment buffer for the convolutional layer
    if (layerInfo.type == ENUM_LyrType::Conv) {
      // obtain basic variables
      int knlCnt = layerPara.asmtLst.GetDimLen(0);
      int knlHei = layerPara.asmtLst.GetDimLen(1);
      int knlWid = layerPara.asmtLst.GetDimLen(2);
      int subSpaceCnt = layerPara.asmtLst.GetDimLen(3);

      // create the assignment buffer
      AsmtBufStr& asmtBufStr = asmtBufStrLst[layerInd];
      asmtBufStr.dimCnt = 4;
      asmtBufStr.dimLenLst[0] = knlHei;
      asmtBufStr.dimLenLst[1] = knlWid;
      asmtBufStr.dimLenLst[2] = subSpaceCnt;
      asmtBufStr.dimLenLst[3] = knlCnt;
      asmtBufStr.pAsmtBuf = new Matrix<uint8_t>(layerPara.asmtLst);
      asmtBufStr.pAsmtBuf->Permute(1, 2, 3, 0);
    }  // ENDIF: layerInfo

    // allocate assignment buffer for the fully-connected layer
    if (layerInfo.type == ENUM_LyrType::FCnt) {
      // obtain basic variables
      int imgChnDst = layerPara.asmtLst.GetDimLen(0);
      int subSpaceCnt = layerPara.asmtLst.GetDimLen(1);

      // create the assignment buffer
      AsmtBufStr& asmtBufStr = asmtBufStrLst[layerInd];
      asmtBufStr.dimCnt = 2;
      asmtBufStr.dimLenLst[0] = subSpaceCnt;
      asmtBufStr.dimLenLst[1] = imgChnDst;
      asmtBufStr.pAsmtBuf = new Matrix<uint8_t>(layerPara.asmtLst);
      asmtBufStr.pAsmtBuf->Permute(1, 0);
    }  // ENDIF: layerInfo
  }  // ENDFOR: layerInd
}

void CaffeEva::CalcFeatMap(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  // determine the corresponding function for the current layer
  swAllLayers.Resume();
  switch (caffeParaObj.layerInfoLst[layerInd].type) {
  case ENUM_LyrType::Conv:
    swConvLayer.Resume();
    CalcFeatMap_Conv(featMapSrc, layerInd, pFeatMapDst);
    swConvLayer.Pause();
    break;
  case ENUM_LyrType::Pool:
    swPoolLayer.Resume();
    CalcFeatMap_Pool(featMapSrc, layerInd, pFeatMapDst);
    swPoolLayer.Pause();
    break;
  case ENUM_LyrType::FCnt:
    swFCntLayer.Resume();
    CalcFeatMap_FCnt(featMapSrc, layerInd, pFeatMapDst);
    swFCntLayer.Pause();
    break;
  case ENUM_LyrType::ReLU:
    swReLuLayer.Resume();
    CalcFeatMap_ReLu(featMapSrc, layerInd, pFeatMapDst);
    swReLuLayer.Pause();
    break;
  case ENUM_LyrType::LoRN:
    swLoRNLayer.Resume();
    CalcFeatMap_LoRN(featMapSrc, layerInd, pFeatMapDst);
    swLoRNLayer.Pause();
    break;
  case ENUM_LyrType::Drpt:
    swDrptLayer.Resume();
    CalcFeatMap_Drpt(featMapSrc, layerInd, pFeatMapDst);
    swDrptLayer.Pause();
    break;
  case ENUM_LyrType::SMax:
    swSMaxLayer.Resume();
    CalcFeatMap_SMax(featMapSrc, layerInd, pFeatMapDst);
    swSMaxLayer.Pause();
    break;
  default:
    printf("[ERROR] invalid layer type\n");
    return;
  }  // ENDSWITCH: caffeParaObj
  swAllLayers.Pause();
}

void CaffeEva::CalcFeatMap_Conv(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  if (enblAprx) {
    CalcFeatMap_ConvAprx(featMapSrc, layerInd, pFeatMapDst);
  } else {
    CalcFeatMap_ConvPrec(featMapSrc, layerInd, pFeatMapDst);
  }  // ENDIF: enblAprx
}

void CaffeEva::CalcFeatMap_ConvPrec(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  // obtain basic variables
  const LayerInfo& layerInfo = caffeParaObj.layerInfoLst[layerInd];
  const LayerPara& layerPara = caffeParaObj.layerParaLst[layerInd];
  int knlCnt = layerPara.convKnlLst.GetDimLen(0);
  int knlSiz = layerPara.convKnlLst.GetDimLen(2);
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHeiSrc = featMapSrc.GetDimLen(1);
  int imgWidSrc = featMapSrc.GetDimLen(2);
  int imgChnSrc = featMapSrc.GetDimLen(3);
  int imgHeiDst = pFeatMapDst->GetDimLen(1);
  int imgWidDst = pFeatMapDst->GetDimLen(2);
  int imgChnDst = pFeatMapDst->GetDimLen(3);
  int knlCntPerGrp = knlCnt / layerInfo.grpCnt;
  int imgChnSrcPerGrp = imgChnSrc / layerInfo.grpCnt;

  // obtain pre-allocated matrices for auxiliary variables
  Matrix<float>& featMapSrcPrm = *(featBufStrMat[layerInd][0].pFeatBuf);
  Matrix<float>& featMapSrcRsp = *(featBufStrMat[layerInd][1].pFeatBuf);
  Matrix<float>& featMapDstRsp = *(featBufStrMat[layerInd][2].pFeatBuf);

  // permute the input feature map dimensions
  featMapSrcPrm.Resize(dataCnt, imgHeiSrc, imgWidSrc, imgChnSrc);
  memcpy(featMapSrcPrm.GetDataPtr(),
      featMapSrc.GetDataPtr(), sizeof(float) * featMapSrc.GetEleCnt());
  featMapSrcPrm.Permute(0, 3, 1, 2);

  // reshape the output feature map
  pFeatMapDst->Resize(dataCnt, imgChnDst, imgHeiDst, imgWidDst);

  // compute the feature map after passing a convolutional layer
  const float* biasVec = layerPara.biasVec.GetDataPtr();
  for (int dataInd = 0; dataInd < dataCnt; dataInd++) {
    for (int grpInd = 0; grpInd < layerInfo.grpCnt; grpInd++) {
      // copy source feature map to feature buffer
      CvtFeatMapToFeatBuf(
          featMapSrcPrm, dataInd, grpInd, layerInfo, &featMapSrcRsp);

      // call CBLAS function to compute the matrix-matrix multiplication
      int knlIndL = grpInd * knlCntPerGrp;
      CBLAS_ORDER order = CblasRowMajor;
      CBLAS_TRANSPOSE transA = CblasNoTrans;
      CBLAS_TRANSPOSE transB = CblasNoTrans;
      CBLAS_INT m = knlCntPerGrp;
      CBLAS_INT n = imgHeiDst * imgWidDst;
      CBLAS_INT k = imgChnSrcPerGrp * knlSiz * knlSiz;
      CBLAS_INT lda = k;
      CBLAS_INT ldb = n;
      CBLAS_INT ldc = n;
      float alpha = 1.0;
      float beta = 0.0;
      float* pa = layerPara.convKnlLst.GetDataPtr(knlIndL, 0, 0, 0);
      float* pb = featMapSrcRsp.GetDataPtr();
      float* pc = featMapDstRsp.GetDataPtr();
      cblas_sgemm(order, transA, transB,
          m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);

      // append the bias term
      int rowCntBuf = featMapDstRsp.GetDimLen(0);
      int colCntBuf = featMapDstRsp.GetDimLen(1);
      for (int rowIndBuf = 0; rowIndBuf < rowCntBuf; rowIndBuf++) {
        const float biasVal = biasVec[rowIndBuf + grpInd * knlCntPerGrp];
        float* pFeatVecDstRsp = featMapDstRsp.GetDataPtr(rowIndBuf, 0);
        for (int colIndBuf = 0; colIndBuf < colCntBuf; colIndBuf++) {
          pFeatVecDstRsp[colIndBuf] += biasVal;
        }  // ENDFOR: colIndBuf
      }  // ENDFOR: rowIndBuf

      // copy feature buffer to target feature map
      CvtFeatBufToFeatMap(
          featMapDstRsp, dataInd, grpInd, layerInfo, pFeatMapDst);
    }  // ENDFOR: grpInd
  }  // ENDFOR: dataInd

  // permute the output feature map dimensions
  pFeatMapDst->Permute(0, 2, 3, 1);
}

void CaffeEva::CalcFeatMap_ConvAprx(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  // obtain basic variables
  const LayerInfo& layerInfo = caffeParaObj.layerInfoLst[layerInd];
  const LayerPara& layerPara = caffeParaObj.layerParaLst[layerInd];
  int knlCnt = layerInfo.knlCnt;
  int knlHei = layerInfo.knlSiz;
  int knlWid = layerInfo.knlSiz;
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHeiSrc = featMapSrc.GetDimLen(1);
  int imgWidSrc = featMapSrc.GetDimLen(2);
  int imgChnSrc = featMapSrc.GetDimLen(3);
  int imgHeiDst = pFeatMapDst->GetDimLen(1);
  int imgWidDst = pFeatMapDst->GetDimLen(2);
  int subSpaceCnt = layerPara.ctrdLst.GetDimLen(0);
  int ctrdCntPerSpace = layerPara.ctrdLst.GetDimLen(1);
  int ctrdCntExt = ctrdCntPerSpace * subSpaceCnt;

  // determine the size of feature map groups
  int knlCntPerGrp = knlCnt / layerInfo.grpCnt;
  int imgChnSrcPerGrp = imgChnSrc / layerInfo.grpCnt;

  // obtain pre-allocated matrices for auxiliary variables
  Matrix<float>& featMapSrcPerGrp = *(featBufStrMat[layerInd][3].pFeatBuf);
  Matrix<float>& inPdMat = *(featBufStrMat[layerInd][4].pFeatBuf);

  // obtain pre-allocated centroid and assignment buffer
  Matrix<float>& ctrdBuf = *(ctrdBufStrLst[layerInd].pCtrdBuf);
  Matrix<uint8_t>& asmtBuf = *(asmtBufStrLst[layerInd].pAsmtBuf);

  // declare <__m256> variables to use AVX instructions
  __m256 vTbl;
  int vRsltCnt = knlCntPerGrp >> 3;
  const int kVRsltCntMax = 64;  // maximal number of channels: 512
  static __m256 vRsltLst[kVRsltCntMax];

  // compute the feature map after passing a convolutional layer
  int sptCntDst = imgHeiDst * imgWidDst;
  const float* biasVec = layerPara.biasVec.GetDataPtr();
  for (int grpInd = 0; grpInd < layerInfo.grpCnt; grpInd++) {
    // obtain basic variables for the current feature map group
    int knlIndL = knlCntPerGrp * grpInd;
    int chnIndSrcL = imgChnSrcPerGrp * grpInd;

    // quantize the source feature map with pre-defined codebook
    swCompLkupTblConv.Resume();
    featMapSrcPerGrp.Resize(dataCnt, imgHeiSrc, imgWidSrc, imgChnSrcPerGrp);
    if (layerInfo.grpCnt == 1) {
      memcpy(featMapSrcPerGrp.GetDataPtr(),
          featMapSrc.GetDataPtr(), sizeof(float) * featMapSrc.GetEleCnt());
    } else {
      featMapSrc.GetSubMat(0, 0, 0, chnIndSrcL, &featMapSrcPerGrp);
    }  // ENDIF: layerInfo
    featMapSrcPerGrp.Resize(dataCnt * imgHeiSrc * imgWidSrc, imgChnSrcPerGrp);
    GetInPdMat(featMapSrcPerGrp, ctrdBuf, &inPdMat);
    inPdMat.Resize(dataCnt, imgHeiSrc, imgWidSrc, ctrdCntExt);
    swCompLkupTblConv.Pause();

    // compute the target response via table look-up operations
    swEstiInPdValConv.Resume();
    for (int sptIndDst = 0; sptIndDst < sptCntDst; sptIndDst++) {
      // determine the corresponding indexes in the target/source feature map
      int heiIndDst = sptIndDst / imgWidDst;
      int widIndDst = sptIndDst % imgWidDst;
      int heiIndSrcL = heiIndDst * layerInfo.stride - layerInfo.padSiz;
      int widIndSrcL = widIndDst * layerInfo.stride - layerInfo.padSiz;

      // determine the lower/upper bound in the convolutional kernels
      int heiIndKnlL = std::max(0, 0 - heiIndSrcL);
      int heiIndKnlU = std::min(knlHei - 1, imgHeiSrc - 1 - heiIndSrcL);
      int widIndKnlL = std::max(0, 0 - widIndSrcL);
      int widIndKnlU = std::min(knlWid - 1, imgWidSrc - 1 - widIndSrcL);

      // compute the target feature map for each instance
      for (int dataInd = 0; dataInd < dataCnt; dataInd++) {
        // initialize <convSumLst> with the bias term
        float* featVecDst =
            pFeatMapDst->GetDataPtr(dataInd, heiIndDst, widIndDst, knlIndL);

        // initialize <vRsltLst> with <biasVec>
        for (int vRsltIdx = 0; vRsltIdx < vRsltCnt; vRsltIdx++) {
          vRsltLst[vRsltIdx] = _mm256_load_ps(&(biasVec[vRsltIdx << 3]));
        }  // ENDFOR: vRsltIdx

        // compute the target response via table look-up operations
        int heiIndKnl;  // to shorten the line length
        int widIndKnl;  // to shorten the line length
        int subSpaceInd;  // to shorten the line length
        for (heiIndKnl = heiIndKnlL; heiIndKnl <= heiIndKnlU; heiIndKnl++) {
          for (widIndKnl = widIndKnlL; widIndKnl <= widIndKnlU; widIndKnl++) {
            int heiIndSrc = heiIndSrcL + heiIndKnl;
            int widIndSrc = widIndSrcL + widIndKnl;
            const float* inPdVec =
                inPdMat.GetDataPtr(dataInd, heiIndSrc, widIndSrc, 0);
            const uint8_t* asmtVec =
                asmtBuf.GetDataPtr(heiIndKnl, widIndKnl, 0, knlIndL);
            for (subSpaceInd = 0; subSpaceInd < subSpaceCnt; subSpaceInd++) {
              for (int knlInd = 0; knlInd < knlCntPerGrp; knlInd += 8) {
                int vRsltIdx = knlInd >> 3;
                SSE_LOOKUP(inPdVec, asmtVec, vTbl, knlInd);
                vRsltLst[vRsltIdx] = _mm256_add_ps(vRsltLst[vRsltIdx], vTbl);
              }  // ENDFOR: knlInd
              inPdVec += ctrdCntPerSpace;
              asmtVec += knlCnt;
            }  // ENDFOR: subSpaceInd
          }  // ENDFOR: widIndKnl
        }  // ENDFOR: heiIndKnl

        // copy results from <vRsltLst> to <featVecDst>
        for (int vRsltIdx = 0; vRsltIdx < vRsltCnt; vRsltIdx++) {
          _mm256_store_ps(&(featVecDst[vRsltIdx << 3]), vRsltLst[vRsltIdx]);
        }  // ENDFOR: vRsltIdx
      }  // ENDFOR: dataInd
    }  // ENDFOR: sptIndDst
    swEstiInPdValConv.Pause();
  }  // ENDFOR: grpInd
}

void CaffeEva::CalcFeatMap_Pool(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  // obtain basic variables
  const LayerInfo& layerInfo = caffeParaObj.layerInfoLst[layerInd];
  int padSiz = layerInfo.padSiz;
  int knlSiz = layerInfo.knlSiz;
  int stride = layerInfo.stride;
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHeiSrc = featMapSrc.GetDimLen(1);
  int imgWidSrc = featMapSrc.GetDimLen(2);
  int imgChn = featMapSrc.GetDimLen(3);
  int imgHeiDst = pFeatMapDst->GetDimLen(1);
  int imgWidDst = pFeatMapDst->GetDimLen(2);

  // compute the feature map after passing a convolutional layer
  for (int heiIndDst = 0; heiIndDst < imgHeiDst; heiIndDst++) {
    // determine the corresponding indexes in the source feature map
    int heiIndSrcL = std::max(0, heiIndDst * stride - padSiz);
    int heiIndSrcU =
        std::min(imgHeiSrc, heiIndDst * stride + knlSiz - padSiz) - 1;
    int heiCntSrcSel = heiIndSrcU - heiIndSrcL + 1;

    for (int widIndDst = 0; widIndDst < imgWidDst; widIndDst++) {
      // determine the corresponding indexes in the source feature map
      int widIndSrcL = std::max(0, widIndDst * stride - padSiz);
      int widIndSrcU =
          std::min(imgWidSrc, widIndDst * stride + knlSiz - padSiz) - 1;
      int widCntSrcSel = widIndSrcU - widIndSrcL + 1;
      int sptCntSrcSel = heiCntSrcSel * widCntSrcSel;

      // perform max-pooling operation
      for (int dataInd = 0; dataInd < dataCnt; dataInd++) {
        float* featVecDst =
            pFeatMapDst->GetDataPtr(dataInd, heiIndDst, widIndDst, 0);
        for (int sptIndSrc = 0; sptIndSrc < sptCntSrcSel; sptIndSrc++) {
          int heiIndSrc = heiIndSrcL + sptIndSrc / widCntSrcSel;
          int widIndSrc = widIndSrcL + sptIndSrc % widCntSrcSel;
          const float* featVecSrc =
              featMapSrc.GetDataPtr(dataInd, heiIndSrc, widIndSrc, 0);
          if (sptIndSrc == 0) {
            memcpy(featVecDst, featVecSrc, sizeof(float) * imgChn);
          } else {
            for (int chnInd = 0; chnInd < imgChn; chnInd++) {
              featVecDst[chnInd] =
                  std::max(featVecSrc[chnInd], featVecDst[chnInd]);
            }  // ENDFOR: chnInd
          }  // ENDIF: sptIndSrc
        }  // ENDFOR: sptIndSrc
      }  // ENDFOR: dataInd
    }  // ENDFOR: widIndDst
  }  // ENDFOR: heiIndDst
}

void CaffeEva::CalcFeatMap_FCnt(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  if (enblAprx) {
    CalcFeatMap_FCntAprx(featMapSrc, layerInd, pFeatMapDst);
  } else {
    CalcFeatMap_FCntPrec(featMapSrc, layerInd, pFeatMapDst);
  }  // ENDIF: enblAprx
}

void CaffeEva::CalcFeatMap_FCntPrec(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  // obtain basic variables
  const LayerPara& layerPara = caffeParaObj.layerParaLst[layerInd];
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgChnSrc = featMapSrc.GetDimStp(0);
  int imgChnDst = pFeatMapDst->GetDimStp(0);

  // call CBLAS function to compute the matrix-matrix multiplication
  CBLAS_ORDER order = CblasRowMajor;
  CBLAS_TRANSPOSE transA = CblasNoTrans;
  CBLAS_TRANSPOSE transB = CblasTrans;
  CBLAS_INT m = dataCnt;
  CBLAS_INT n = imgChnDst;
  CBLAS_INT k = imgChnSrc;
  CBLAS_INT lda = k;
  CBLAS_INT ldb = k;
  CBLAS_INT ldc = n;
  float alpha = 1.0;
  float beta = 0.0;
  float* pa = featMapSrc.GetDataPtr();
  float* pb = layerPara.fcntWeiMat.GetDataPtr();
  float* pc = pFeatMapDst->GetDataPtr();
  cblas_sgemm(order, transA, transB,
      m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);

  // append the bias term
  const float* biasVec = layerPara.biasVec.GetDataPtr();
  for (int dataInd = 0; dataInd < dataCnt; dataInd++) {
    float* pFeatMapVec = pFeatMapDst->GetDataPtr(dataInd, 0);
    for (int chnIndDst = 0; chnIndDst < imgChnDst; chnIndDst++) {
      pFeatMapVec[chnIndDst] += biasVec[chnIndDst];
    }  // ENDFOR: chnIndDst
  }  // ENDFOR: dataInd
}

void CaffeEva::CalcFeatMap_FCntAprx(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  // obtain basic variables
  const LayerPara& layerPara = caffeParaObj.layerParaLst[layerInd];
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgChnDst = pFeatMapDst->GetDimStp(0);
  int subSpaceCnt = layerPara.ctrdLst.GetDimLen(0);
  int ctrdCntPerSpace = layerPara.ctrdLst.GetDimLen(1);

  // obtain pre-allocated matrices for auxiliary variables
  Matrix<float>& featMapSrcRsp = *(featBufStrMat[layerInd][0].pFeatBuf);
  Matrix<float>& inPdMat = *(featBufStrMat[layerInd][1].pFeatBuf);
  inPdMat.Resize(dataCnt, subSpaceCnt, ctrdCntPerSpace);

  // obtain pre-allocated centroid and assignment buffer
  Matrix<float>& ctrdBuf = *(ctrdBufStrLst[layerInd].pCtrdBuf);
  Matrix<uint8_t>& asmtBuf = *(asmtBufStrLst[layerInd].pAsmtBuf);

  // declare <__m256> variables to use AVX instructions
  __m256 vTbl;
  int vRsltCnt = imgChnDst >> 3;
  const int kVRsltCntMax = 512;  // maximal number of outputs: 4096
  static __m256 vRsltLst[kVRsltCntMax];

  // quantize the source feature map with pre-defined codebook
  swCompLkupTblFCnt.Resume();
  memcpy(featMapSrcRsp.GetDataPtr(),
      featMapSrc.GetDataPtr(), sizeof(float) * featMapSrc.GetEleCnt());
  GetInPdMat(featMapSrcRsp, ctrdBuf, &inPdMat);
  inPdMat.Resize(dataCnt, subSpaceCnt * ctrdCntPerSpace);
  swCompLkupTblFCnt.Pause();

  // compute the feature map after passing a fully-connected layer
  swEstiInPdValFCnt.Resume();
  const float* biasVec = layerPara.biasVec.GetDataPtr();
  for (int dataInd = 0; dataInd < dataCnt; dataInd++) {
    // initialize target response with the bias term
    float* featVecDst = pFeatMapDst->GetDataPtr(dataInd, 0, 0, 0);

    // initialize <vRsltLst> with <biasVec>
    for (int vRsltIdx = 0; vRsltIdx < vRsltCnt; vRsltIdx++) {
      vRsltLst[vRsltIdx] = _mm256_load_ps(&(biasVec[vRsltIdx << 3]));
    }  // ENDFOR: vRsltIdx

    // update target response with look-up operations
    const float* inPdVec = inPdMat.GetDataPtr(dataInd, 0);  // index offset
    const uint8_t* asmtVec = asmtBuf.GetDataPtr();
    for (int subSpaceInd = 0; subSpaceInd < subSpaceCnt; subSpaceInd++) {
      // update the target response within the current subspace
      for (int chnIndDst = 0; chnIndDst < imgChnDst; chnIndDst += 8) {
        int vRsltIdx = chnIndDst >> 3;
        SSE_LOOKUP(inPdVec, asmtVec, vTbl, chnIndDst);
        vRsltLst[vRsltIdx] = _mm256_add_ps(vRsltLst[vRsltIdx], vTbl);
      }  // ENDFOR: chnIndDst

      // update pointers to the look-up table and assignment variable
      inPdVec += ctrdCntPerSpace;
      asmtVec += imgChnDst;
    }  // ENDFOR: subSpaceInd

    // copy results from <vRsltLst> to <featVecDst>
    for (int vRsltIdx = 0; vRsltIdx < vRsltCnt; vRsltIdx++) {
      _mm256_store_ps(&(featVecDst[vRsltIdx << 3]), vRsltLst[vRsltIdx]);
    }  // ENDFOR: vRsltIdx
  }  // ENDFOR: dataInd
  swEstiInPdValFCnt.Pause();
}

void CaffeEva::CalcFeatMap_ReLu(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  // compute the feature map after passing a ReLu layer
  int eleCnt = featMapSrc.GetEleCnt();
  const float* featVecSrc = featMapSrc.GetDataPtr();
  float* featVecDst = pFeatMapDst->GetDataPtr();
  for (int eleInd = 0; eleInd < eleCnt; eleInd++) {
    featVecDst[eleInd] = std::max(0.0f, featVecSrc[eleInd]);
  }  // ENDFOR: eleInd
}

void CaffeEva::CalcFeatMap_LoRN(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  // obtain basic variables
  const LayerInfo& layerInfo = caffeParaObj.layerInfoLst[layerInd];
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHei = featMapSrc.GetDimLen(1);
  int imgWid = featMapSrc.GetDimLen(2);
  int imgChn = featMapSrc.GetDimLen(3);
  int lrnRad = (layerInfo.lrnSiz - 1) / 2;
  int sptCnt = imgHei * imgWid;

  // declare auxiliary variable arrays
  int imgChnExt = imgChn + lrnRad * 2;
  float* featVecSrcExt = new float[imgChnExt];
  float* loclSumLst = new float[imgChn];

  // compute the feature map after passing a local response normalization layer
  float coeffVal = layerInfo.lrnAlp / layerInfo.lrnSiz;
  memset(featVecSrcExt, 0, sizeof(float) * imgChnExt);
  for (int dataInd = 0; dataInd < dataCnt; dataInd++) {
    for (int sptInd = 0; sptInd < sptCnt; sptInd++) {
      // determine the height/width indexes
      int heiInd = sptInd / imgWid;
      int widInd = sptInd % imgWid;

      // compute the squared feature vector
      const float* featVecSrc =
          featMapSrc.GetDataPtr(dataInd, heiInd, widInd, 0);
      vsSqr(imgChn, featVecSrc, featVecSrcExt + lrnRad);
      cblas_sscal(imgChn, coeffVal, featVecSrcExt + lrnRad, 1);

      // compute <loclSumLst> with a sliding windows
      for (int chnInd = 0; chnInd < imgChn; chnInd++) {
        loclSumLst[chnInd] = layerInfo.lrnIni;
      }  // ENDFOR: chnInd
      for (int chnInd = 0; chnInd < layerInfo.lrnSiz; chnInd++) {
        vsAdd(imgChn, loclSumLst, featVecSrcExt + chnInd, loclSumLst);
      }  // ENDFOR: chnInd

      // transform local patch sum to normalization factor
      vsPowx_m(imgChn, loclSumLst, -layerInfo.lrnBet, loclSumLst);

      // compute the normalized feature map
      float* featVecDst = pFeatMapDst->GetDataPtr(dataInd, heiInd, widInd, 0);
      vsMul(imgChn, featVecSrc, loclSumLst, featVecDst);
    }  // ENDFOR: sptInd
  }  // ENDFOR: dataInd

  // release auxiliary variable arrays
  delete[] featVecSrcExt;
  delete[] loclSumLst;
}

void CaffeEva::CalcFeatMap_Drpt(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  // compute the feature map after passing a dropout layer
  memcpy(pFeatMapDst->GetDataPtr(),
      featMapSrc.GetDataPtr(), sizeof(float) * featMapSrc.GetEleCnt());
}

void CaffeEva::CalcFeatMap_SMax(const Matrix<float>& featMapSrc,
    const int layerInd, Matrix<float>* pFeatMapDst) {
  // compute the feature map after passing a softmax layer
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgChn = featMapSrc.GetDimStp(0);
  for (int dataInd = 0; dataInd < dataCnt; dataInd++) {
    const float* featVecSrc = featMapSrc.GetDataPtr(dataInd, 0, 0, 0);
    float* featVecDst = pFeatMapDst->GetDataPtr(dataInd, 0, 0, 0);

    float sum = 0.0;
    for (int chnInd = 0; chnInd < imgChn; chnInd++) {
      featVecDst[chnInd] = exp(featVecSrc[chnInd]);
      sum += featVecDst[chnInd];
    }  // ENDFOR: chnInd
    for (int chnInd = 0; chnInd < imgChn; chnInd++) {
      featVecDst[chnInd] /= sum;
    }  // ENDFOR: chnInd
  }  // ENDFOR: dataInd
}

void CaffeEva::InitFeatBuf(
    FeatBufStr* pFeatBufStr, const ENUM_BufUsage us, const int d0) {
  pFeatBufStr->usage = us;
  pFeatBufStr->dimCnt = 1;
  pFeatBufStr->dimLenLst[0] = d0;
}

void CaffeEva::InitFeatBuf(FeatBufStr* pFeatBufStr,
    const ENUM_BufUsage us, const int d0, const int d1) {
  InitFeatBuf(pFeatBufStr, us, d0);
  pFeatBufStr->dimCnt = 2;
  pFeatBufStr->dimLenLst[1] = d1;
}

void CaffeEva::InitFeatBuf(FeatBufStr* pFeatBufStr,
    const ENUM_BufUsage us, const int d0, const int d1, const int d2) {
  InitFeatBuf(pFeatBufStr, us, d0, d1);
  pFeatBufStr->dimCnt = 3;
  pFeatBufStr->dimLenLst[2] = d2;
}

void CaffeEva::InitFeatBuf(FeatBufStr* pFeatBufStr, const ENUM_BufUsage us,
    const int d0, const int d1, const int d2, const int d3) {
  InitFeatBuf(pFeatBufStr, us, d0, d1, d2);
  pFeatBufStr->dimCnt = 4;
  pFeatBufStr->dimLenLst[3] = d3;
}

void CaffeEva::CvtDataLstToFeatMap(const int dataIndL, const int dataIndU,
    const Matrix<float>& dataLst, Matrix<float>* pFeatMap) {
  // obtain basic variables
  int dataVecLen = dataLst.GetDimStp(0);
  int imgChn = dataLst.GetDimLen(1);
  int imgHei = dataLst.GetDimLen(2);
  int imgWid = dataLst.GetDimLen(3);

  // copy each sample's feature vector
  int dataCntSel = dataIndU - dataIndL + 1;
  pFeatMap->Resize(dataCntSel, imgChn, imgHei, imgWid);
  memcpy(pFeatMap->GetDataPtr(), dataLst.GetDataPtr(dataIndL, 0, 0, 0),
      sizeof(float) * dataVecLen * dataCntSel);
  pFeatMap->Permute(0, 2, 3, 1);
}

void CaffeEva::CvtFeatMapToLablVec(const int dataIndL, const int dataIndU,
    const Matrix<float>& featMap, Matrix<uint16_t>* pLablVec) {
  // determine the predicted class label from the output feature map
  int probVecLen = featMap.GetDimStp(0);
  Matrix<float> probLst(probVecLen);
  float* probVec = probLst.GetDataPtr();
  for (int dataInd = dataIndL; dataInd <= dataIndU; dataInd++) {
    // copy category probabilities to a temporary array
    memcpy(probVec, featMap.GetDataPtr(dataInd - dataIndL, 0, 0, 0),
        sizeof(float) * probVecLen);

    // determine the x-th predicted class label
    for (int lablInd = 0; lablInd < kLablCntPerData; lablInd++) {
      // find the maximal probability
      float probValOpt = FLT_MIN;
      uint16_t probValIndOpt = 0;
      for (int probValInd = 0; probValInd < probVecLen; probValInd++) {
        if (probValOpt < probVec[probValInd]) {
          probValOpt = probVec[probValInd];
          probValIndOpt = probValInd;
        }  // ENDIF: probValOpt
      }  // ENDFOR: probValInd

      // record current prediction
      probVec[probValIndOpt] = 0.0;
      pLablVec->SetEleAt(probValIndOpt, dataInd, lablInd, 0, 0);
    }  // ENDFOR: lablInd
  }  // ENDFOR: dataInd
}

// INPUT REQUIREMENTS:
//   featMap: 1 x Cs x Hs x Ws
//   featBuf: (Ht x Wt) x (Ck x Hk x Wk)
void CaffeEva::CvtFeatMapToFeatBuf(
    const Matrix<float>& featMap, const int dataInd, const int grpInd,
    const LayerInfo& layerInfo, Matrix<float>* pFeatBuf) {
  // obtain basic variables
  int imgChnSrc = featMap.GetDimLen(1);
  int imgHeiSrc = featMap.GetDimLen(2);
  int imgWidSrc = featMap.GetDimLen(3);
  int grpCnt = layerInfo.grpCnt;
  int padSiz = layerInfo.padSiz;
  int knlSiz = layerInfo.knlSiz;
  int stride = layerInfo.stride;
  int imgChnSrcPerGrp = imgChnSrc / grpCnt;
  int imgHeiDst = ceil((imgHeiSrc + 2 * padSiz - knlSiz) / stride) + 1;
  int imgWidDst = ceil((imgWidSrc + 2 * padSiz - knlSiz) / stride) + 1;
  int rowCntBuf = pFeatBuf->GetDimLen(0);

  // copy feature data from <featMap> to <featBuf>
  int chnIndSrcL = grpInd * imgChnSrcPerGrp;
  memset(pFeatBuf->GetDataPtr(), 0, sizeof(float) * pFeatBuf->GetEleCnt());
  for (int rowIndBuf = 0; rowIndBuf < rowCntBuf; rowIndBuf++) {
    // obtain basic variables
    int widIndKnl = rowIndBuf % knlSiz;
    int heiIndKnl = rowIndBuf / knlSiz % knlSiz;
    int chnIndKnl = rowIndBuf / knlSiz / knlSiz;
    int heiIndDstL = std::max(0, (padSiz - heiIndKnl - 1) / stride + 1);
    int heiIndDstU =
        std::min(imgHeiDst - 1, (padSiz - heiIndKnl + imgHeiSrc - 1) / stride);
    int heiCntDstSel = heiIndDstU - heiIndDstL + 1;
    int widIndDstL = std::max(0, (padSiz - widIndKnl - 1) / stride + 1);
    int widIndDstU =
        std::min(imgWidDst - 1, (padSiz - widIndKnl + imgWidSrc - 1) / stride);
    int widCntDstSel = widIndDstU - widIndDstL + 1;
    const float* ptrSrc =
        featMap.GetDataPtr(dataInd, chnIndSrcL + chnIndKnl, 0, 0);
    float* ptrDst = pFeatBuf->GetDataPtr(rowIndBuf, 0);

    // copy feature data from <featMap> to <featBuf>
    for (int heiIndDstSel = 0; heiIndDstSel < heiCntDstSel; heiIndDstSel++) {
      for (int widIndDstSel = 0; widIndDstSel < widCntDstSel; widIndDstSel++) {
        int heiIndDst = heiIndDstL + heiIndDstSel;
        int widIndDst = widIndDstL + widIndDstSel;
        int heiIndSrc = heiIndKnl + heiIndDst * stride - padSiz;
        int widIndSrc = widIndKnl + widIndDst * stride - padSiz;
        ptrDst[heiIndDst * imgWidDst + widIndDst] =
            ptrSrc[heiIndSrc * imgWidSrc + widIndSrc];
      }  // ENDFOR: widIndDstSel
    }  // ENDFOR: heiIndDstSel
  }  // ENDFOR: rowIndBuf
}

// INPUT REQUIREMENTS:
//   featBuf: Ct x (Ht x Wt)
//   featMap: 1 x Ct x Ht x Wt
void CaffeEva::CvtFeatBufToFeatMap(
    const Matrix<float>& featBuf, const int dataInd, const int grpInd,
    const LayerInfo& layerInfo, Matrix<float>* pFeatMap) {
  // obtain basic variables
  int imgChnDst = pFeatMap->GetDimLen(1);
  int chnIndDstL = imgChnDst / layerInfo.grpCnt * grpInd;

  // copy feature data from <featBuf> to <featMap>
  const float* ptrSrc = featBuf.GetDataPtr();
  float* ptrDst = pFeatMap->GetDataPtr(dataInd, chnIndDstL, 0, 0);
  memcpy(ptrDst, ptrSrc, sizeof(float) * featBuf.GetEleCnt());
}

void CaffeEva::GetInPdMat(const Matrix<float>& dataLst,
    const Matrix<float>& ctrdLst, Matrix<float>* pInPdMat) {
  // obtain basic variables
  int dataCnt = dataLst.GetDimLen(0);
  int featDimCnt = dataLst.GetDimLen(1);
  int subSpaceCnt = ctrdLst.GetDimLen(0);
  int featCntPerSpace = ctrdLst.GetDimLen(1);
  int ctrdCntPerSpace = ctrdLst.GetDimLen(2);

  // resize the inner-product look-up table
  pInPdMat->Resize(dataCnt, subSpaceCnt, ctrdCntPerSpace);

  // compute the inner-product look-up table in each subspace
  for (int subSpaceInd = 0; subSpaceInd < subSpaceCnt; subSpaceInd++) {
    // determine the selected dimensions
    int featDimIndL = featCntPerSpace * subSpaceInd;
    int featDimCntSel = std::min(featDimCnt - featDimIndL, featCntPerSpace);

    // compute the inner-product look-up table for each instance
    const float* dataVec = dataLst.GetDataPtr(0, featDimIndL);
    float* inPdVec = pInPdMat->GetDataPtr(0, subSpaceInd, 0);
    for (int dataInd = 0; dataInd < dataCnt; dataInd++) {
      const float* ctrdVec = ctrdLst.GetDataPtr(subSpaceInd, 0, 0);
      memset(inPdVec, 0, sizeof(float) * ctrdCntPerSpace);
      for (int featDimInd = 0; featDimInd < featDimCntSel; featDimInd++) {
        cblas_saxpy(
            ctrdCntPerSpace, dataVec[featDimInd], ctrdVec, 1, inPdVec, 1);
        ctrdVec += ctrdCntPerSpace;
      }  // ENDFOR: featDimInd

      // update pointers to the data vector and look-up table
      dataVec += featDimCnt;
      inPdVec += subSpaceCnt * ctrdCntPerSpace;
    }  // ENDFOR: dataInd
  }  // ENDFOR: subSpaceInd
}
