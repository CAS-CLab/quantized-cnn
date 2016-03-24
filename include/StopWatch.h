/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#ifndef INCLUDE_STOPWATCH_H_
#define INCLUDE_STOPWATCH_H_

#include <ctime>

class StopWatch {
 public:
  // reset the stop-watch
  inline void Reset(void);
  // resume the stop-watch
  inline void Resume(void);
  // pause the stop-watch
  inline void Pause(void);
  // return the elapsed time
  inline float GetTime(void);

 private:
  // indicator for whether the stop-watch is running
  bool isRun;
  // timestamp of the latest Resume() operation
  time_t timeBeg;
  // timestamp of the latest Pause() operation
  time_t timeEnd;
  // elapsed time (in seconds)
  float timeElapsed;
};

// implementation of member functions

inline void StopWatch::Reset(void) {
  isRun = false;
  timeElapsed = 0.0;
}

inline void StopWatch::Resume(void) {
  if (!isRun) {
    isRun = true;
    timeBeg = clock();
  }  // ENDIF: isRun
}

inline void StopWatch::Pause(void) {
  if (isRun) {
    isRun = false;
    timeEnd = clock();
    timeElapsed += static_cast<float>(timeEnd - timeBeg) / CLOCKS_PER_SEC;
  }  // ENDIF: isRun
}

inline float StopWatch::GetTime(void) {
  return timeElapsed;
}

#endif  // INCLUDE_STOPWATCH_H_
