# libtorch yolov4

yolov4 implement by libtorch in c++

### usage
can only inference by now 

```
git clone
mkdir build
cmake -DCMAKE_PREFIX_PATH=<libtorch abs path> ..
./yolov4 <yolov4.cfg> <yolov4.weights> <image_path>
```
the result write to det_result.png
