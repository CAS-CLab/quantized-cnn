/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#ifndef INCLUDE_CAFFEPARA_H_
#define INCLUDE_CAFFEPARA_H_

#include <string>
#include <vector>

#include "../include/Common.h"
#include "../include/Matrix.h"

// define ENUM class for file formats
// description:
//   assume <x> is the minimal number of bits to store an index.
//   FileFormat::Raw: each index is stored in <ceil(x / 8)> bytes
//   FileFormat::Compact: each index is stored in <x / 8> bytes
enum class ENUM_AsmtEnc {Raw, Compact};

// define ENUM class for layer types
enum class ENUM_LyrType {Conv, Pool, FCnt, ReLU, LoRN, Drpt, SMax};

// define structure <LayerInfo> and <LayerInfoLst>
typedef struct {
  ENUM_LyrType type;  // layer type
  int padSiz;  // number of padded pixels (on each side)
  int knlSiz;  // width/height of the convolutional kernel
  int knlCnt;  // number of convolutional kernels
  int grpCnt;  // number of source feature map groups
  int stride;  // convolution stride / spatial step
  int nodCnt;  // number of target neurons (for the <fcnt> layer)
  int lrnSiz;  // local response patch size
  float lrnAlp;  // local response normalization - alpha
  float lrnBet;  // local response normalization - beta
  float lrnIni;  // local response normalization - initial value
  float drpRat;  // dropout ratio (how many neurons are preserved)
} LayerInfo;
typedef std::vector<LayerInfo> LayerInfoLst;

// define structure <LayerPara> and <LayerParaLst>
typedef struct {
  Matrix<float> convKnlLst;
  Matrix<float> fcntWeiMat;
  Matrix<float> biasVec;
  Matrix<float> ctrdLst;
  Matrix<uint8_t> asmtLst;
} LayerPara;
typedef std::vector<LayerPara> LayerParaLst;

class CaffePara {
 public:
  // initialize parameters for quantization
  void Init(const std::string& dirPathSrc, const std::string& filePfxSrc);
  // configure all layers according to the <AlexNet> settings
  void ConfigLayer_AlexNet(void);
  // configure all layers according to the <CaffeNet> settings
  void ConfigLayer_CaffeNet(void);
  // configure all layers according to the <VggCnnS> settings
  void ConfigLayer_VggCnnS(void);
  // configure all layers according to the <VGG16> settings
  void ConfigLayer_VGG16(void);
  // configure all layers according to the <CaffeNetFGB> settings
  void ConfigLayer_CaffeNetFGB(void);
  // configure all layers according to the <CaffeNetFGD> settings
  void ConfigLayer_CaffeNetFGD(void);
  // load layer parameters from files
  bool LoadLayerPara(const bool enblAprx, const ENUM_AsmtEnc asmtEnc);
  // convert raw-encoded index files to compact-encoded
  bool CvtAsmtEnc(const ENUM_AsmtEnc asmtEncSrc, const ENUM_AsmtEnc asmtEncDst);

 public:
  // main directory for data import/export
  std::string dirPath;
  // file name prefix
  std::string filePfx;
  // number of layers
  int layerCnt;
  // number of input feature map channels
  int imgChnIn;
  // input feature map height
  int imgHeiIn;
  // input feature map width
  int imgWidIn;
  // all layers' basic information
  LayerInfoLst layerInfoLst;
  // all layers' parameters
  LayerParaLst layerParaLst;

 private:
  // determine the proper value for <bitCntPerEle>
  int CalcBitCntPerEle(const Matrix<uint8_t>& asmtLst);
  // configure each layer type
  void ConfigConvLayer(LayerInfo* pLayerInfo, const int padSiz,
      const int knlSiz, const int knlCnt, const int grpCnt, const int stride);
  void ConfigPoolLayer(LayerInfo* pLayerInfo,
      const int padSiz, const int knlSiz, const int stride);
  void ConfigFCntLayer(LayerInfo* pLayerInfo, const int nodCnt);
  void ConfigReLuLayer(LayerInfo* pLayerInfo);
  void ConfigLoRNLayer(LayerInfo* pLayerInfo, const int lrnSiz,
      const float lrnAlp, const float lrnBet, const float lrnIni);
  void ConfigDrptLayer(LayerInfo* pLayerInfo, const float drpRat);
  void ConfigSMaxLayer(LayerInfo* pLayerInfo);
};

#endif  // INCLUDE_CAFFEPARA_H_
