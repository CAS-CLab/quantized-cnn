/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#include "../include/CaffeEvaWrapper.h"

CaffeEvaWrapper::CaffeEvaWrapper(void) {
  // clear error message
  ClrErrorMsg();
}

bool CaffeEvaWrapper::SetPath(const std::string& mainDirPathSrc,
    const std::string& clsNameFilePath, const std::string& imgLablFilePath) {
  // declare auxiliary variables
  bool rtnFlg;

  // initialize basic variables
  mainDirPath = mainDirPathSrc;

  // load the full list of class names
  rtnFlg = LoadClsName(clsNameFilePath);
  if (!rtnFlg) {  // failed
    errorMsg =
        "[CaffeEvaWrapper::SetPath] could not open file: " + clsNameFilePath;
    return false;
  }  // ENDIF: rtnFlg

  // load the full list of ground-truth class names
  if (imgLablFilePath != "") {
    rtnFlg = LoadImgLabl(imgLablFilePath);
    if (!rtnFlg) {  // failed
      errorMsg =
          "[CaffeEvaWrapper::SetPath] could not open file: " + imgLablFilePath;
      return false;
    }  // ENDIF: rtnFlg
  }  // ENDIF: imgLablFileName

  return true;
}

bool CaffeEvaWrapper::SetModel(const ENUM_CaffeModel& caffeModelSrc,
    const ENUM_CompMethod& compMethodSrc) {
  // declare auxiliary variables
  bool rtnFlg;

  // initialize basic variables
  caffeModel = caffeModelSrc;
  compMethod = compMethodSrc;

  // initialize remaining variables according to the selected caffe model
  switch (caffeModel) {
  case ENUM_CaffeModel::AlexNet:
    // fall through
  case ENUM_CaffeModel::CaffeNet:
    // fall through
  case ENUM_CaffeModel::CaffeNetFGB:
    // fall through
  case ENUM_CaffeModel::CaffeNetFGD:
    bmpImgIOPara.reszType = ENUM_ReszType::Strict;
    bmpImgIOPara.meanType = ENUM_MeanType::Full;
    bmpImgIOPara.imgHeiFull = 256;
    bmpImgIOPara.imgWidFull = 256;
    bmpImgIOPara.imgHeiCrop = 227;
    bmpImgIOPara.imgWidCrop = 227;
    break;
  case ENUM_CaffeModel::VggCnnS:
    bmpImgIOPara.reszType = ENUM_ReszType::Relaxed;
    bmpImgIOPara.meanType = ENUM_MeanType::Crop;
    bmpImgIOPara.imgHeiFull = 256;
    bmpImgIOPara.imgWidFull = 256;
    bmpImgIOPara.imgHeiCrop = 224;
    bmpImgIOPara.imgWidCrop = 224;
    break;
  case ENUM_CaffeModel::VGG16:
    printf("[FATAL ERROR] VGG-16 is not supported (for now)\n");
    errorMsg = "[CaffeEvaWrapper::SetModel] unsupported caffe model name";
    return false;
  default:
    printf("[FATAL ERROR] unrecognized <ENUM_CaffeModel> value\n");
    errorMsg = "[CaffeEvaWrapper::SetModel] unrecognized caffe model name";
    return false;
  }  // ENDSWITCH: caffeModel

  switch (caffeModel) {
  case ENUM_CaffeModel::AlexNet:
    bmpImgIOPara.filePathMean =
        mainDirPath + "/AlexNet/imagenet_mean.single.bin";
    caffeModelName = "AlexNet";
    dirPathPara = mainDirPath + "/AlexNet/Bin.Files";
    fileNamePfx = "bvlc_alexnet_aCaF";
    break;
  case ENUM_CaffeModel::CaffeNet:
    bmpImgIOPara.filePathMean =
        mainDirPath + "/CaffeNet/imagenet_mean.single.bin";
    caffeModelName = "CaffeNet";
    dirPathPara = mainDirPath + "/CaffeNet/Bin.Files";
    fileNamePfx = "bvlc_caffenet_aCaF";
    break;
  case ENUM_CaffeModel::VggCnnS:
    bmpImgIOPara.filePathMean =
        mainDirPath + "/VggCnnS/imagenet_mean.single.bin";
    caffeModelName = "VggCnnS";
    dirPathPara = mainDirPath + "/VggCnnS/Bin.Files";
    fileNamePfx = "vgg_cnn_s_aCaF";
    break;
  case ENUM_CaffeModel::VGG16:
    printf("[FATAL ERROR] VGG-16 is not supported (for now)\n");
    errorMsg = "[CaffeEvaWrapper::SetModel] unsupported caffe model name";
    return false;
  case ENUM_CaffeModel::CaffeNetFGB:
    bmpImgIOPara.filePathMean =
        mainDirPath + "/CaffeNetFGB/imagenet_mean.single.bin";
    caffeModelName = "CaffeNetFGB";
    dirPathPara = mainDirPath + "/CaffeNetFGB/Bin.Files";
    fileNamePfx = "bvlc_caffenetfgb_aCaF";
    break;
  case ENUM_CaffeModel::CaffeNetFGD:
    bmpImgIOPara.filePathMean =
        mainDirPath + "/CaffeNetFGD/imagenet_mean.single.bin";
    caffeModelName = "CaffeNetFGD";
    dirPathPara = mainDirPath + "/CaffeNetFGD/Bin.Files";
    fileNamePfx = "bvlc_caffenetfgd_aCaF";
    break;
  default:
    printf("[FATAL ERROR] unrecognized <ENUM_CaffeModel> value\n");
    errorMsg = "[CaffeEvaWrapper::SetModel] unrecognized caffe model name";
    return false;
  }  // ENDSWITCH: caffeModel

  // initialize essential variables in <BmpImgIO>
  rtnFlg = bmpImgIOObj.Init(bmpImgIOPara);
  if (!rtnFlg) {  // failed
    errorMsg = "[CaffeEvaWrapper::SetModel] could not open the mean image file";
    return false;
  }  // ENDIF: rtnFlg

  // initialize essential variables in <CaffeEva>
  caffeEvaObj.Init(compMethodSrc == ENUM_CompMethod::Aprx);
  caffeEvaObj.SetModelName(caffeModelName);
  caffeEvaObj.SetModelPath(dirPathPara, fileNamePfx);
  rtnFlg = caffeEvaObj.LoadCaffePara();
  if (!rtnFlg) {  // failed
    errorMsg = "[CaffeEvaWrapper::SetModel] could not load model files";
    return false;
  }  // ENDIF: rtnFlg

  return true;
}

bool CaffeEvaWrapper::Proc(
    const std::string& filePathProcImg, CaffeEvaRslt* pCaffeEvaRslt) {
  // declare auxiliary variables
  bool rtnFlg;

  // obtain the image file path for classification
  Matrix<float> imgData;
  rtnFlg = bmpImgIOObj.Load(filePathProcImg, &imgData);
  if (!rtnFlg) {  // failed
    errorMsg = "[CaffeEvaWrapper::Proc] could open the BMP file";
    return false;
  }  // ENDIF: rtnFlg

  // execute the forward-passing process of caffe model
  Matrix<float> probVec;
  caffeEvaObj.ExecForwardPass(imgData, &probVec);
  pCaffeEvaRslt->timeTotal = caffeEvaObj.DispElpsTime();

  // find the ground-truth class name (if exists)
  std::string fileNameProcImg = ExtrFileName(filePathProcImg);
  pCaffeEvaRslt->hasGrthClsName = false;
  for (std::size_t idx = 0; idx < clsNameGrthLst.size(); idx++) {
    if (fileNameProcImg == clsNameGrthLst[idx].fileName) {
      pCaffeEvaRslt->hasGrthClsName = true;
      pCaffeEvaRslt->clsNameGrth = clsNameGrthLst[idx].clsNameGrth;
      break;
    }  // ENDIF: fileNameProcImg
  }  // ENDFOR: idx

  // find the top-ranked categories and pack them into <caffeEvaRslt>
  int clsCnt = probVec.GetEleCnt();
  float* pProbVec = probVec.GetDataPtr();
  pCaffeEvaRslt->clsIdxLst.clear();
  pCaffeEvaRslt->clsProbLst.clear();
  pCaffeEvaRslt->clsNameLst.clear();
  for (int rankInd = 0; rankInd < pCaffeEvaRslt->clsCntPred; rankInd++) {
    // fine the k-th top-ranked class label
    int clsIndOpt = 0;
    float probValOpt = pProbVec[clsIndOpt];
    for (int clsInd = 1; clsInd < clsCnt; clsInd++) {
      if (probValOpt < pProbVec[clsInd]) {
        clsIndOpt = clsInd;
        probValOpt = pProbVec[clsInd];
      }  // ENDIF: probValOpt
    }  // ENDFOR: clsInd

    // record current predicted class label
    pCaffeEvaRslt->clsIdxLst.push_back(clsIndOpt);
    pCaffeEvaRslt->clsProbLst.push_back(probValOpt);
    pCaffeEvaRslt->clsNameLst.push_back(clsNameLst[clsIndOpt]);

    // modify <probVec> for the next round
    pProbVec[clsIndOpt] = 0.0;
  }  // ENDFOR: rankInd

  return true;
}

std::string CaffeEvaWrapper::GetErrorMsg(void) {
  return errorMsg;
}

void CaffeEvaWrapper::ClrErrorMsg(void) {
  errorMsg = "";
}

bool CaffeEvaWrapper::LoadClsName(const std::string& filePath) {
  // define auxiliary variables
  char* rtnPtr = nullptr;
  const int kStrBufLen = 1024;
  char strBuf[kStrBufLen];
  FILE* inFile = fopen(filePath.c_str(), "r");

  // check whether the file has been successfully opened
  if (inFile == nullptr) {
    return false;
  }  // ENDIF: inFile

  // scan through each line
  clsNameLst.clear();
  while (true) {
    // read in a new line
    rtnPtr = fgets(strBuf, kStrBufLen, inFile);

    // check whether EOF is reached
    if (rtnPtr == nullptr) {
      break;
    }  // ENDIF: rtnPtr

    // add the current class name to <clsNameLst>
    strBuf[strlen(strBuf) - 1] = '\0';  // remove the EOL character
    clsNameLst.push_back(strBuf);
  }  // ENDWHILE: true
  fclose(inFile);

  return true;
}

bool CaffeEvaWrapper::LoadImgLabl(const std::string& filePath) {
  // define auxiliary variables
  int rtnVal;
  const int kStrBufLen = 1024;
  char strBuf[kStrBufLen];
  int clsNameInd;
  ClsNameGrthStr clsNameGrthStr;
  FILE* inFile = fopen(filePath.c_str(), "r");

  // check whether the file has been successfully opened
  if (inFile == nullptr) {
    return false;
  }  // ENDIF: inFile

  // scan through each line
  clsNameGrthLst.clear();
  while (true) {
    // read in a new line
    rtnVal = fscanf(inFile, "%s%d", strBuf, &clsNameInd);

    // check whether EOF is reached
    if (rtnVal != 2) {
      break;
    }  // ENDIF: rtnVal

    // add the current grouth-truth class name to <clsNameGrthLst>
    clsNameGrthStr.fileName = ExtrFileName(std::string(strBuf));
    clsNameGrthStr.clsNameGrth = clsNameLst[clsNameInd];
    clsNameGrthLst.push_back(clsNameGrthStr);
  }  // ENDWHILE: true
  fclose(inFile);

  return true;
}

std::string CaffeEvaWrapper::ExtrFileName(const std::string& filePath) {
  // define auxiliary variables
  const int kStrBufLen = 1024;
  char strBuf[kStrBufLen];
  std::string fileName;
  int charIndBeg = 0;
  bool fileNameFound = false;

  // copy the full file path to <strBuf>
  snprintf(strBuf, kStrBufLen, "%s", filePath.c_str());

  // find the first slash symbol
  for (int charInd = strlen(strBuf) - 1; (charInd >= 0); charInd--) {
    switch (strBuf[charInd]) {
    case '.':
      strBuf[charInd] = '\0';
      break;
    case '/':
      charIndBeg = charInd + 1;
      fileNameFound = true;
      break;
    }  // ENDSWITCH: strBuf

    // check whether the file name has been found
    if (fileNameFound) {
      break;
    }  // ENDIF: fileNameFound
  }  // ENDFOR: charInd

  // extract the file name
  fileName = std::string(strBuf + charIndBeg);

  return fileName;
}
