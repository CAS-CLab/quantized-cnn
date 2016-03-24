/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#ifndef INCLUDE_BMPIMGIO_H_
#define INCLUDE_BMPIMGIO_H_

#include <string>

#include "../include/Common.h"
#include "../include/Matrix.h"

/*
 *Resizing Type:
 *  Strict: image is strictly resized to HxW (aspect may change)
 *  Relaxed: image is resized to H'xW' (ascpect will not change)
 *           s.t. H' >= H, W' >= W, (H' - H) * (W' - W) = 0
 */
enum class ENUM_ReszType {Strict, Relaxed};

/*
 * Mean Image Type:
 *   Full: mean image is of equal size to the full image
 *   Crop: mean image is of equal size to the cropped image
 */
enum class ENUM_MeanType {Full, Crop};

typedef struct {
  ENUM_ReszType reszType;  // resizing type
  ENUM_MeanType meanType;  // mean image type
  int imgHeiFull;  // full image height
  int imgWidFull;  // full image width
  int imgHeiCrop;  // cropped image height
  int imgWidCrop;  // cropped image width
  std::string filePathMean;  // file path of the mean image
} BmpImgIOPara;

class BmpImgIO {
 public:
  // initialize parameters
  bool Init(const BmpImgIOPara& bmpImgIOPara);
  // load a BMP image from file and prepare it for CNN input
  bool Load(const std::string& filePath, Matrix<float>* pImgDataFnal);

 private:
  // resizing type
  ENUM_ReszType reszType;
  // mean image type
  ENUM_MeanType meanType;
  // full image height
  int imgHeiFull;
  // full image width
  int imgWidFull;
  // cropped image height
  int imgHeiCrop;
  // cropped image width
  int imgWidCrop;
  // file path to the mean image
  std::string filePathMean;
  // mean image
  Matrix<float> imgDataMean;
  // height of the mean image
  int imgHeiMean;
  // width of the mean image
  int imgWidMean;

 private:
  // load a BMP image from file
  bool LoadBmpImg(const std::string& filePath, Matrix<float>* pImgData);
  // resize image to the specified size
  void ReszImg(const Matrix<float>& imgDataSrc, Matrix<float>* pImgDataDst,
      const ENUM_ReszType type, const int imgHeiDstPst, const int imgWidDstPst);
  // crop the central patch from the source image
  void CropImg(const Matrix<float>& imgDataSrc, Matrix<float>* pImgDataDst,
      const int imgHeiDst, const int imgWidDst);
  // remove the mean image from the source image
  void RmMeanImg(const Matrix<float>& imgDataMean, Matrix<float>* pImgDataProc);
};

#endif  // INCLUDE_BMPIMGIO_H_
