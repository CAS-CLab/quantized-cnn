/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#include "../include/BmpImgIO.h"

#include "../include/FileIO.h"
#include "../include/bitmap_image.hpp"

const int kImgChn = 3;
const double kEpsilon = 0.0000001;

bool BmpImgIO::Init(const BmpImgIOPara& bmpImgIOPara) {
  // initialize basic variables
  reszType = bmpImgIOPara.reszType;
  meanType = bmpImgIOPara.meanType;
  imgHeiFull = bmpImgIOPara.imgHeiFull;
  imgWidFull = bmpImgIOPara.imgWidFull;
  imgHeiCrop = bmpImgIOPara.imgHeiCrop;
  imgWidCrop = bmpImgIOPara.imgWidCrop;
  filePathMean = bmpImgIOPara.filePathMean;

  // load mean image
  // NOTE: <imgDataMean> is <CxHxW>, using BGR format
  bool rtnFlg = FileIO::ReadBinFile(filePathMean, &imgDataMean);
  if (!rtnFlg) {  // failed
    return false;
  }  // ENDIF: rtnFlg

  // obtain the height/width of the mean image
  imgHeiMean = imgDataMean.GetDimLen(1);
  imgWidMean = imgDataMean.GetDimLen(2);

  return true;
}

bool BmpImgIO::Load(const std::string& filePath, Matrix<float>* pImgDataFnal) {
  // define auxiliary variables
  Matrix<float> imgDataOrgn;
  Matrix<float> imgDataFull;

  // load a BMP image from file
  // NOTE: <imgDataOrgn> is <1xCxHxW>, using BGR format
  bool rtnFlg = LoadBmpImg(filePath, &imgDataOrgn);
  if (!rtnFlg) {  // failed
    return false;
  }  // ENDIF: rtnFlg

  // resize the original into to full image for cropping
  ReszImg(imgDataOrgn, &imgDataFull, reszType, imgHeiFull, imgWidFull);

  // remove the mean image and crop the central patch (or vice versa)
  switch (meanType) {
  case ENUM_MeanType::Full:
    RmMeanImg(imgDataMean, &imgDataFull);
    CropImg(imgDataFull, pImgDataFnal, imgHeiCrop, imgWidCrop);
    break;
  case ENUM_MeanType::Crop:
    CropImg(imgDataFull, pImgDataFnal, imgHeiCrop, imgWidCrop);
    RmMeanImg(imgDataMean, pImgDataFnal);
    break;
  default:
    printf("[ERROR] unrecognized <ENUM_ResizeType> value\n");
    return false;
  }  // ENDSWITCH: meanType

  return true;
}

bool BmpImgIO::LoadBmpImg(
    const std::string& filePath, Matrix<float>* pImgData) {
  // read initial image data from file
  bitmap_image image(filePath);

  // check whether the image has been successfully opened
  if (!image) {
    printf("[ERROR] cannot open the BMP image at %s\n", filePath.c_str());
    return false;
  }  // ENDIF: image

  // allocate memory for <pImgData>
  int imgHei = image.height();
  int imgWid = image.width();
  pImgData->Create(1, kImgChn, imgHei, imgWid);

  // copy image data to <pImgData>
  uint8_t valR;
  uint8_t valG;
  uint8_t valB;
  for (int heiIdx = 0; heiIdx < imgHei; heiIdx++) {
    for (int widIdx = 0; widIdx < imgWid; widIdx++) {
      image.get_pixel(widIdx, heiIdx, valR, valG, valB);
      pImgData->SetEleAt(valB, 0, 0, heiIdx, widIdx);
      pImgData->SetEleAt(valG, 0, 1, heiIdx, widIdx);
      pImgData->SetEleAt(valR, 0, 2, heiIdx, widIdx);
    }  // ENDFOR: widIdx
  }  // ENDFOR: heiIdx

  return true;
}

void BmpImgIO::ReszImg(const Matrix<float>& imgDataSrc,
    Matrix<float>* pImgDataDst, const ENUM_ReszType type,
    const int imgHeiDstPst, const int imgWidDstPst) {
  // obtain basic variables
  int imgHeiSrc = imgDataSrc.GetDimLen(2);
  int imgWidSrc = imgDataSrc.GetDimLen(3);

  // determine the target image size
  int imgHeiDst;
  int imgWidDst;
  float scalFctrHei;
  float scalFctrWid;
  switch (type) {
  case ENUM_ReszType::Strict:
    scalFctrHei = static_cast<float>(imgHeiSrc - 1) / (imgHeiDstPst - 1);
    scalFctrWid = static_cast<float>(imgWidSrc - 1) / (imgWidDstPst - 1);
    imgHeiDst = imgHeiDstPst;
    imgWidDst = imgWidDstPst;
    break;
  case ENUM_ReszType::Relaxed:
    scalFctrHei = static_cast<float>(imgHeiSrc - 1) / (imgHeiDstPst - 1);
    scalFctrWid = static_cast<float>(imgWidSrc - 1) / (imgWidDstPst - 1);
    scalFctrHei = std::min(scalFctrHei, scalFctrWid);
    scalFctrWid = std::min(scalFctrHei, scalFctrWid);
    imgHeiDst = static_cast<int>((imgHeiSrc - 1) / scalFctrHei + kEpsilon) + 1;
    imgWidDst = static_cast<int>((imgWidSrc - 1) / scalFctrWid + kEpsilon) + 1;
    break;
  default:
    printf("[ERROR] unrecognized <ENUM_ResizeType> value\n");
    return;
  }  // ENDSWITCH: type
  printf("[INFO] resizing image from %d x %d to %d x %d\n",
      imgHeiSrc, imgWidSrc, imgHeiDst, imgWidDst);

  // resize the target image
  pImgDataDst->Resize(1, kImgChn, imgHeiDst, imgWidDst);

  // determine the image data after resizing
  for (int heiIdxDst = 0; heiIdxDst < imgHeiDst; heiIdxDst++) {
    // determine the corresponding indexes in the source image
    float heiIdxSrcC = scalFctrHei * heiIdxDst;
    int heiIdxSrcL = std::max(0, static_cast<int>(heiIdxSrcC));
    int heiIdxSrcH = std::min(imgHeiSrc - 1, heiIdxSrcL + 1);
    float weiHeiL = 1.0 - (heiIdxSrcC - heiIdxSrcL);
    float weiHeiH = 1.0 - (heiIdxSrcH - heiIdxSrcC);

    // scan through all indices in the target image
    for (int widIdxDst = 0; widIdxDst < imgWidDst; widIdxDst++) {
      // determine the corresponding indices in the source image
      float widIdxSrcC = scalFctrWid * widIdxDst;
      int widIdxSrcL = std::max(0, static_cast<int>(widIdxSrcC));
      int widIdxSrcH = std::min(imgWidSrc - 1, widIdxSrcL + 1);
      float weiWidL = 1.0 - (widIdxSrcC - widIdxSrcL);
      float weiWidH = 1.0 - (widIdxSrcH - widIdxSrcC);

      // fetch the LT/RT/LB/RB pixel values in the source image
      for (int chnIdx = 0; chnIdx < kImgChn; chnIdx++) {
        float valLT = imgDataSrc.GetEleAt(0, chnIdx, heiIdxSrcL, widIdxSrcL);
        float weiLT = weiHeiL * weiWidL;
        float valRT = imgDataSrc.GetEleAt(0, chnIdx, heiIdxSrcL, widIdxSrcH);
        float weiRT = weiHeiL * weiWidH;
        float valLB = imgDataSrc.GetEleAt(0, chnIdx, heiIdxSrcH, widIdxSrcL);
        float weiLB = weiHeiH * weiWidL;
        float valRB = imgDataSrc.GetEleAt(0, chnIdx, heiIdxSrcH, widIdxSrcH);
        float weiRB = weiHeiH * weiWidH;

        float valSum = valLT * weiLT +
            valRT * weiRT + valLB * weiLB + valRB * weiRB;
        float weiSum = weiLT + weiRT + weiLB + weiRB;
        pImgDataDst->SetEleAt(valSum / weiSum, 0, chnIdx, heiIdxDst, widIdxDst);
      }  // ENDFOR: chnIdx
    }  // ENDFOR: widIdxDst
  }  // ENDFOR: heiIdxDst
}

void BmpImgIO::CropImg(const Matrix<float>& imgDataSrc,
    Matrix<float>* pImgDataDst, const int imgHeiDst, const int imgWidDst) {
  // obtain basic variables
  int imgHeiSrc = imgDataSrc.GetDimLen(2);
  int imgWidSrc = imgDataSrc.GetDimLen(3);
  int heiOffset = (imgHeiSrc - imgHeiDst) / 2;
  int widOffset = (imgWidSrc - imgWidDst) / 2;

  // resize the target image
  pImgDataDst->Resize(1, kImgChn, imgHeiDst, imgWidDst);

  // crop the central patch from the source image
  for (int chnIdx = 0; chnIdx < kImgChn; chnIdx++) {
    for (int heiIdxDst = 0; heiIdxDst < imgHeiDst; heiIdxDst++) {
      int heiIdxSrc = heiIdxDst + heiOffset;
      const float* pImgSrc =
          imgDataSrc.GetDataPtr(0, chnIdx, heiIdxSrc, widOffset);
      float* pImgDst = pImgDataDst->GetDataPtr(0, chnIdx, heiIdxDst, 0);
      memcpy(pImgDst, pImgSrc, sizeof(float) * imgWidDst);
    }  // ENDFOR: heiIdxDst
  }  // ENDFOR: chnIdx
}

void BmpImgIO::RmMeanImg(
    const Matrix<float>& imgDataMean, Matrix<float>* pImgDataProc) {
  // obtain basic variables
  int imgHeiProc = pImgDataProc->GetDimLen(2);
  int imgWidProc = pImgDataProc->GetDimLen(3);

  // verify the image size
  if ((imgHeiProc != imgHeiMean) || (imgWidProc != imgWidMean)) {
    printf("[ERROR] mismatch in the image size\n");
    printf("imgHeiProc/Mean = %d/%d\n", imgHeiProc, imgHeiMean);
    printf("imgWidProc/Mean = %d/%d\n", imgWidProc, imgWidMean);
    return;
  }  // ENDIF: imgHeiProc

  // modify each pixel's value
  int pxlCnt = pImgDataProc->GetEleCnt();
  const float* pImgMean = imgDataMean.GetDataPtr();
  float* pImgProc = pImgDataProc->GetDataPtr();
  for (int pxlIdx = 0; pxlIdx < pxlCnt; pxlIdx++) {
    *(pImgProc++) -= *(pImgMean++);
  }  // ENDFOR: pixelIdx
}
