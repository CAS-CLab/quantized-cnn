/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#ifndef INCLUDE_CAFFEEVA_H_
#define INCLUDE_CAFFEEVA_H_

#include <string>
#include <vector>

#include "../include/Common.h"
#include "../include/BlasWrapper.h"
#include "../include/CaffePara.h"
#include "../include/Matrix.h"
#include "../include/StopWatch.h"

// un-comment this for Android debug output
// #define ENBL_ANDROID_DBG

// define structure <FeatMapSiz> and <FeatMapSizLst>
typedef struct {
  int dataCnt;
  int imgHei;
  int imgWid;
  int imgChn;
} FeatMapSiz;
typedef std::vector<FeatMapSiz> FeatMapSizLst;

// define structure <FeatBufStr>, <FeatBufStrLst>, and <FeatBufStrMat>
enum class ENUM_BufUsage {GnrlComp, PrecComp, AprxComp};
typedef struct {
  ENUM_BufUsage usage;
  int dimCnt;
  int dimLenLst[kMatDimCntMax];
  Matrix<float>* pFeatBuf;
} FeatBufStr;
typedef std::vector<FeatBufStr> FeatBufStrLst;
typedef std::vector<FeatBufStrLst> FeatBufStrMat;

// define structure <CtrdBufStr> and <CtrdBufStrLst>
typedef struct {
  int dimCnt;
  int dimLenLst[kMatDimCntMax];
  Matrix<float>* pCtrdBuf;
} CtrdBufStr;
typedef std::vector<CtrdBufStr> CtrdBufStrLst;

// define structure <AsmtBufStr> and <AsmtBufStrLst>
typedef struct {
  int dimCnt;
  int dimLenLst[kMatDimCntMax];
  Matrix<uint8_t>* pAsmtBuf;
  Matrix<CBLAS_INT>* pAsmtBufExt;  // for cblas_sgthr()
} AsmtBufStr;
typedef std::vector<AsmtBufStr> AsmtBufStrLst;

class CaffeEva {
 public:
  // deconstructor function
  ~CaffeEva(void);

 public:
  // initialize essential variables
  void Init(const bool enblAprxSrc);
  // specify the model name
  void SetModelName(const std::string& modelNameSrc);
  // specify the main model directory and the model file name prefix
  void SetModelPath(
      const std::string& dirPathMainSrc, const std::string& fileNamePfxSrc);
  // load dataset
  bool LoadDataset(const std::string& dirPathData);
  // load caffe parameters
  bool LoadCaffePara(void);
  // run forward process for all samples
  void ExecForwardPass(void);
  // run forward process for the input sample
  void ExecForwardPass(
      const Matrix<float>& imgDataIn, Matrix<float>* pProbVecOut);
  // evaluate the classification accuracy
  void CalcPredAccu(void);
  // display the elapsed time of each step
  float DispElpsTime(void);

 private:
  // whether approximate computation is enabled
  bool enblAprx;
  // caffe model name
  std::string modelName;
  // main data directory
  std::string dirPathMain;
  // model parameter file name's prefix
  std::string fileNamePfx;
  // object of <CaffePara> class
  CaffePara caffeParaObj;
  // evaluation samples' feature vectors
  Matrix<float> dataLst;
  // evaluation samples' ground-truth labels
  Matrix<uint16_t> lablVecGrth;
  // evaluation samples' predicted labels
  Matrix<uint16_t> lablVecPred;
  // feature map size after passing through each layer
  FeatMapSizLst featMapSizLst;
  // feature map list for each layer
  Matrix<float>* featMapLst;
  // feature buffer structure for each layer
  FeatBufStrMat featBufStrMat;
  // centroid buffer structure for each layer
  CtrdBufStrLst ctrdBufStrLst;
  // assignment buffer structure for each layer
  AsmtBufStrLst asmtBufStrLst;

 private:
  // objects of <StopWatch> class for accurate time measuring
  StopWatch swAllLayers;
  StopWatch swConvLayer;
  StopWatch swPoolLayer;
  StopWatch swFCntLayer;
  StopWatch swReLuLayer;
  StopWatch swLoRNLayer;
  StopWatch swDrptLayer;
  StopWatch swSMaxLayer;
  StopWatch swCompLkupTblConv;
  StopWatch swEstiInPdValConv;
  StopWatch swCompLkupTblFCnt;
  StopWatch swEstiInPdValFCnt;
  // objects of <StopWatch> class for debug usage
  StopWatch swDebugTimePri;
  StopWatch swDebugTimeSec;
  // objects of <StopWatch> class for each layer in the network
  std::vector<StopWatch> swIndvLayerLst;

 private:
  // prepare feature map for each layer
  void PrepFeatMap(void);
  // prepare feature buffers for each layer
  void PrepFeatBuf(void);
  // prepare centroid buffers for each convolutional/fully-connected layer
  void PrepCtrdBuf(void);
  // prepare assignment buffers for each convolutional layer
  void PrepAsmtBuf(void);
  // compute the feature map after passing a certain layer
  void CalcFeatMap(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_Conv(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_ConvPrec(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_ConvAprx(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_Pool(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_FCnt(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_FCntPrec(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_FCntAprx(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_FCntAprxSp(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_ReLu(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_LoRN(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_Drpt(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  void CalcFeatMap_SMax(const Matrix<float>& featMapSrc,
      const int layerInd, Matrix<float>* pFeatMapDst);
  // initialize the <FeatBufStr> structure
  void InitFeatBuf(
      FeatBufStr* pFeatBufStr, const ENUM_BufUsage us, const int d0);
  void InitFeatBuf(FeatBufStr* pFeatBufStr,
      const ENUM_BufUsage us, const int d0, const int d1);
  void InitFeatBuf(FeatBufStr* pFeatBufStr,
      const ENUM_BufUsage us, const int d0, const int d1, const int d2);
  void InitFeatBuf(FeatBufStr* pFeatBufStr, const ENUM_BufUsage us,
      const int d0, const int d1, const int d2, const int d3);
  // convert samples' feature vectors into the input feature map
  void CvtDataLstToFeatMap(const int dataIndBeg, const int dataIndEnd,
      const Matrix<float>& dataLst, Matrix<float>* pFeatMap);
  // convert the output feature map into samples' predicted labels
  void CvtFeatMapToLablVec(const int dataIndBeg, const int dataIndEnd,
      const Matrix<float>& featMap, Matrix<uint16_t>* pLablVec);
  // convert feature map to feature buffer
  void CvtFeatMapToFeatBuf(const Matrix<float>& featMap, const int dataInd,
      const int grpInd, const LayerInfo& layerInfo, Matrix<float>* pFeatBuf);
  // convert feature buffer to feature map
  void CvtFeatBufToFeatMap(const Matrix<float>& featBuf, const int dataInd,
      const int grpInd, const LayerInfo& layerInfo, Matrix<float>* pFeatMap);
  // compute the look-up table for inner product
  void GetInPdMat(const Matrix<float>& dataLst,
      const Matrix<float>& ctrdLst, Matrix<float>* pInPdMat);
};

#endif  // INCLUDE_CAFFEEVA_H_
