# DanoDet-Plus TensorRT


## 1.生成wts文件
+ 将gen_wts.py文件copy到nandeodet[原始仓库](https://github.com/RangiLyu/nanodet)
+ 选取合适参数(config文件和pth模型),运行脚本得到wts文件

## 2.生成engine文件运行
```
mkdir build && cd build
make -j4
./nanodet -s
./nanodet -d
```

## 3.Reference resources
* 1.https://github.com/wang-xinyu/tensorrtx</br>
* 2.https://github.com/RangiLyu/nanodet</br>
* 3.https://docs.nvidia.com/deeplearning/tensorrt/api/c_api
