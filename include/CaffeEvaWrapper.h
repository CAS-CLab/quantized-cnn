/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#ifndef INCLUDE_CAFFEEVAWRAPPER_H_
#define INCLUDE_CAFFEEVAWRAPPER_H_

#include <string>
#include <vector>

#include "../include/Common.h"
#include "../include/BmpImgIO.h"
#include "../include/CaffeEva.h"

enum class ENUM_CaffeModel {
    AlexNet, CaffeNet, VggCnnS, VGG16, CaffeNetFGB, CaffeNetFGD};
enum class ENUM_CompMethod {Prec, Aprx};

typedef struct {
  int clsCntPred;
  float timeTotal;
  bool hasGrthClsName;
  std::string clsNameGrth;
  std::vector<int> clsIdxLst;
  std::vector<float> clsProbLst;
  std::vector<std::string> clsNameLst;
} CaffeEvaRslt;

typedef struct {
  std::string fileName;
  std::string clsNameGrth;
} ClsNameGrthStr;
typedef std::vector<ClsNameGrthStr> ClsNameGrthLst;

class CaffeEvaWrapper {
 public:
  // Constructor Function
  // Description:
  //   this function will initialize basic member variables
  // Input:
  //   none
  // Output:
  //   none
  CaffeEvaWrapper(void);

 public:
  // Path Set-up Function
  // Description:
  //   this function will set-up the main directory path
  // Input:
  //   mainDirPathSrc: main directory path (model parameters and mean images).
  //   clsNameFilePath: file that contains all class names
  //   imgLablFilePath: file that contains all image labels (optional)
  // Output:
  //   none
  bool SetPath(
      const std::string& mainDirPathSrc, const std::string& clsNameFilePath,
      const std::string& imgLablFilePath = "");

 public:
  // Model Set-up Function
  // Description:
  //   this function will load model parameters and pre-allocate
  //   memory space for feature maps and buffers.
  // Input:
  //   caffeModelSrc: caffe model ("AlexNet" or "VggCnnS").
  //   compMethodSrc: computation method ("Prec" or "Aprx").
  // Output:
  //   none
  bool SetModel(const ENUM_CaffeModel& caffeModelSrc,
      const ENUM_CompMethod& compMethodSrc);

 public:
  // Core Process Function
  // Description:
  //   this function will accept a string as the file path to the
  //   input image and generate a fixed number of predicted class
  //   labels with the selected convolutional neural network.
  // Input:
  //   filePathProcImg: file path to the input image.
  //   caffeEvaRslt: top-k classification result.
  // Output:
  //   none
  bool Proc(const std::string& filePathProcImg, CaffeEvaRslt* pCaffeEvaRslt);

 public:
  // Get Error Message Function
  // Description:
  //   this function will return the error message (if any)
  // Input:
  //   none
  // Output:
  //   (std::string) error message
  std::string GetErrorMsg(void);

 public:
  // Clear Error Message Function
  // Description:
  //   this function will clear the error message
  // Input:
  //   none
  // Output:
  //   none
  void ClrErrorMsg(void);

 private:
  // main directory path
  std::string mainDirPath;
  // caffe model
  ENUM_CaffeModel caffeModel;
  // computation method
  ENUM_CompMethod compMethod;
  // basic parameters for <BmpImgIO> class
  BmpImgIOPara bmpImgIOPara;
  // caffe model name
  std::string caffeModelName;
  // directory path to model parameters
  std::string dirPathPara;
  // model file name's prefix
  std::string fileNamePfx;
  // object of <BmpImgIO> class
  BmpImgIO bmpImgIOObj;
  // object of <CaffeEva> class
  CaffeEva caffeEvaObj;
  // full list of class names
  std::vector<std::string> clsNameLst;
  // ground-truth class names
  ClsNameGrthLst clsNameGrthLst;
  // error message
  std::string errorMsg;

 private:
  // load the full list of class names
  bool LoadClsName(const std::string& filePath);
  // load the full list of image lables
  bool LoadImgLabl(const std::string& filePath);
  // extract file name from the given file path
  std::string ExtrFileName(const std::string& filePath);
};

#endif  // INCLUDE_CAFFEEVAWRAPPER_H_
