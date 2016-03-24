/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#include "../include/UnitTest.h"

#include "../include/CaffeEva.h"
#include "../include/CaffeEvaWrapper.h"
#include "../include/CaffePara.h"
#include "../include/StopWatch.h"

void UnitTest::UT_CaffePara(void) {
  // create class objects for unit-test
  CaffePara caffeParaObj;

  // load parameters for the caffe model
  caffeParaObj.Init("./AlexNet/Bin.Files", "bvlc_alexnet_aCaF");
  caffeParaObj.ConfigLayer_AlexNet();
  caffeParaObj.LoadLayerPara(true, ENUM_AsmtEnc::Raw);
  caffeParaObj.CvtAsmtEnc(ENUM_AsmtEnc::Raw, ENUM_AsmtEnc::Compact);
  caffeParaObj.LoadLayerPara(true, ENUM_AsmtEnc::Compact);
}

void UnitTest::UT_CaffeEva(void) {
  // create class objects for unit-test
  StopWatch stopWatchObj;
  CaffeEva caffeEvaObj;

  // choose a caffe model
  bool kEnblAprxComp = true;
  const std::string kCaffeModelName = "AlexNet";

  // evaluate the caffe model's classification accuracy
  stopWatchObj.Reset();
  stopWatchObj.Resume();
  caffeEvaObj.Init(kEnblAprxComp);  // enable approximate but fast computation
  if (kCaffeModelName == "AlexNet") {
    caffeEvaObj.SetModelName("AlexNet");
    caffeEvaObj.SetModelPath("./AlexNet/Bin.Files", "bvlc_alexnet_aCaF");
    caffeEvaObj.LoadDataset("./ILSVRC12.227x227.IMG");
  } else if (kCaffeModelName == "CaffeNet") {
    caffeEvaObj.SetModelName("CaffeNet");
    caffeEvaObj.SetModelPath("./CaffeNet/Bin.Files", "bvlc_caffenet_aCaF");
    caffeEvaObj.LoadDataset("./ILSVRC12.227x227.IMG");
  } else if (kCaffeModelName == "VggCnnS") {
    caffeEvaObj.SetModelName("VggCnnS");
    caffeEvaObj.SetModelPath("./VggCnnS/Bin.Files", "vgg_cnn_s_aCaF");
    caffeEvaObj.LoadDataset("./ILSVRC12.224x224.IMG");
  } else if (kCaffeModelName == "VGG16") {
    caffeEvaObj.SetModelName("VGG16");
    caffeEvaObj.SetModelPath("./VGG16/Bin.Files", "vgg16_aCaF");
    caffeEvaObj.LoadDataset("./ILSVRC12.224x224.PXL");
  } else {
    printf("[ERROR] unrecognized caffe model: %s\n", kCaffeModelName.c_str());
  }  // ENDIF: kCaffeModelName
  caffeEvaObj.LoadCaffePara();
  caffeEvaObj.ExecForwardPass();
  caffeEvaObj.CalcPredAccu();
  caffeEvaObj.DispElpsTime();
  stopWatchObj.Pause();
  printf("elapsed time: %.4f (s)\n", stopWatchObj.GetTime());
}

void UnitTest::UT_CaffeEvaWrapper(void) {
  // declare auxiliary variables
  bool rtnFlg;

  // create class objects for unit-test
  StopWatch stopWatchObj;
  CaffeEvaWrapper caffeEvaWrapperObj;

  // initialize constant variables
  const std::string kMainDirPath = "./";
  const ENUM_CaffeModel kCaffeModel = ENUM_CaffeModel::AlexNet;
  const ENUM_CompMethod kCompMethod = ENUM_CompMethod::Aprx;
  const std::string kClsNameFilePath = "./Cls.Names/class_names.txt";
  const std::string kImgLablFilePath = "./Cls.Names/image_labels.txt";
  const std::string kBmpFilePath = "./Bmp.Files/ILSVRC2012_val_00000002.BMP";

  // initialize the structure for result storing
  CaffeEvaRslt caffeEvaRslt;
  caffeEvaRslt.clsCntPred = 5;

  // predict class labels for a single BMP image
  stopWatchObj.Reset();
  stopWatchObj.Resume();
  rtnFlg = caffeEvaWrapperObj.SetPath(
      kMainDirPath, kClsNameFilePath, kImgLablFilePath);
  if (!rtnFlg) {  // failed
    printf("[ERROR] CaffeEvaWrapper::SetPath() return FALSE\n");
    printf("[ERROR] call CaffeEvaWrapper::GetErrorMsg() to details\n");
    return;
  }  // ENDIF: rtnFlg
  rtnFlg = caffeEvaWrapperObj.SetModel(kCaffeModel, kCompMethod);
  if (!rtnFlg) {  // failed
    printf("[ERROR] CaffeEvaWrapper::SetModel() returns FALSE\n");
    printf("[ERROR] call CaffeEvaWrapper::GetErrorMsg() to details\n");
    return;
  }  // ENDIF: rtnFlg
  rtnFlg = caffeEvaWrapperObj.Proc(kBmpFilePath, &caffeEvaRslt);
  if (!rtnFlg) {  // failed
    printf("[ERROR] CaffeEvaWrapper::Proc() returns FALSE\n");
    printf("[ERROR] call CaffeEvaWrapper::GetErrorMsg() to details\n");
    return;
  }  // ENDIF: rtnFlg
  stopWatchObj.Pause();
  printf("elapsed time: %.4f (s)\n", stopWatchObj.GetTime());

  // display the classification result
  printf("[INFO] input image: %s\n", kBmpFilePath.c_str());
  if (caffeEvaRslt.hasGrthClsName) {
    printf("[INFO] Ground-truth class name: %s\n",
        caffeEvaRslt.clsNameGrth.c_str());
  }  // ENDIF: caffeEvaRslt
  for (int clsIdxPred = 0; clsIdxPred < caffeEvaRslt.clsCntPred; clsIdxPred++) {
    printf("[INFO] No. %d: %s (%d / %.4f)\n", clsIdxPred + 1,
        caffeEvaRslt.clsNameLst[clsIdxPred].c_str(),
        caffeEvaRslt.clsIdxLst[clsIdxPred],
        caffeEvaRslt.clsProbLst[clsIdxPred]);
  }  // ENDFOR: clsIdxPred
}
