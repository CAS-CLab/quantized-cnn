/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#ifndef INCLUDE_UNITTEST_H_
#define INCLUDE_UNITTEST_H_

#include "../include/Common.h"

class UnitTest {
 public:
  // unit-test for the <CaffePara> class
  static void UT_CaffePara(void);
  // unit-test for the <CaffeEva> class
  static void UT_CaffeEva(void);
  // unit-test for the <CaffeEvaWrapper> class
  static void UT_CaffeEvaWrapper(void);
};

#endif  // INCLUDE_UNITTEST_H_
