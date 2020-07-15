# libtorch yolov4

yolov4 implement by libtorch in c++

### LibTorch version
LibTorch 1.5.1+cu101


### usage
can only inference by now 

```
git clone
mkdir build
cmake -DCMAKE_PREFIX_PATH=<libtorch abs path> ..
./yolov4 <yolov4.cfg> <yolov4.weights> <image_path>
```
the result write to det_result.png

[libtorch-yolov3](https://github.com/walktree/libtorch-yolov3)
[pytorch-YOLOV4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
