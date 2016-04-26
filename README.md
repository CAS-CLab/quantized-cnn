# Quantized-CNN for Mobile Devices

Quantized-CNN is a novel framework of convolutional neural network (CNN) with simultaneous computation acceleration and model compression in the test-phase. Mobile devices can perform efficient on-site image classification via our Quantized-CNN, with only negligible loss in accuracy.

## Installation

We have prepared a file (500+MB) containing 1k images drawn from the ILSVRC-12 validation set for a more accurate speed-test. You can download it from [here](https://onedrive.live.com/redir?resid=D968C5EC99B231C!647138&authkey=!AFq00tB5N71t1Cw&ithint=file%2cbin), and put it under the "ILSVRC12.227x227.IMG" directory.

For the original AlexNet model, you can download the corresponding model files from [here](https://onedrive.live.com/redir?resid=D968C5EC99B231C!742681&authkey=!AMiIg1D39Bdxumo&ithint=file%2cgzl), and put them under the "AlexNet/Bin.Files" directory.

Prior to compilation, you need to install [ATLAS](http://math-atlas.sourceforge.net) and [OpenVML](https://github.com/xianyi/OpenVML), and modify Makefile if needed. After that, use "make" to generate the executable file and "make run" to perform the speed-test with the above 1k images.

You can also use our code for single image classification (BMP format). Please refer to "src/Main.cc" for details.

## Speed-test

The experiment is carried out on a single desktop PC, equipped with an Intel&reg; Core&trade; i7-4790K CPU and 32GB RAM. All programs are executed in the single-thread mode, without GPU acceleration. **Note that the run-time speed comparison result may vary under different hardware conditions.**

We compare the run-time speed of [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks), for which Quantized-CNN's theoretical speed-up is 4.15&times;. For the baseline method, we use the [Caffe](http://caffe.berkeleyvision.org/) implementation, compiled with ATLAS (default BLAS choice). We measure the forward-passing time per image, based on the average of 100 batches. Each batch contains a single image, since in practice, users usually take one photo with their cellphones and then fed it into the ConvNet for classification. The experiment is repeated five times and here are the results:

| Time (ms) |     CNN | Quantized-CNN |    Speed-up |
|:---------:|:-------:|:-------------:|:-----------:|
|         1 | 167.431 |        55.346 |           - |
|         2 | 168.578 |        55.382 |           - |
|         3 | 166.120 |        55.372 |           - |
|         4 | 172.792 |        55.389 |           - |
|         5 | 164.008 |        55.250 |           - |
|       Ave.| 167.786 |        55.348 | 3.03&times; |

Quantized-CNN achieves 3.03&times; speed-up against the Caffe implementation, slightly lower than the theoretical one but still quite acceptable. Meanwhile, our method requires much less memory and storage space, which is critical for mobile applications.

## Citation

Please cite our paper if it helps your research:

    @inproceedings{wu2016quantized,
      author = {Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng},
      title = {Quantized Convolutional Neural Networks for Mobile Devices},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2016},
    }
