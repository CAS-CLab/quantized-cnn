/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#include "../include/UnitTest.h"

int main(int argc, char* argv[]) {
  // run unit-test for the <CaffePara> class
  // UnitTest::UT_CaffePara();

  // MODE 1: SPEED TEST
  // run unit-test for the <CaffeEva> class
  UnitTest::UT_CaffeEva();

  // MODE 2: SINGLE IMAGE CLASSIFICATION
  // run unit-test for the <CaffeEvaWrapper> class
  // UnitTest::UT_CaffeEvaWrapper();

  return 0;
}
