# YOLOV8_Deploy
**YOLOV8's TRT deploy**
```bash
pip install ultralytics
```
导出ONNX（这里ONNX和原版做了修改）
```bash
Python ultralytics/infer.py
Python ultralytics/onnx_to_smi.py
```
**C++ Cfg**
![avatar](/cfg.png)
![avatar](/cfg2.png)

链接器->输入
```bash
cublas.lib
cublasLt.lib
cuda.lib
cudadevrt.lib
cudart.lib
cudart_static.lib
cufft.lib
cufftw.lib
curand.lib
cusolver.lib
cusolverMg.lib
cusparse.lib
nppc.lib
nppial.lib
nppicc.lib
nppidei.lib
nppif.lib
nppig.lib
nppim.lib
nppist.lib
nppisu.lib
nppitc.lib
npps.lib
nvblas.lib
nvjpeg.lib
nvml.lib
nvrtc.lib
OpenCL.lib
nvparsers.lib
nvonnxparser.lib
nvinfer_plugin.lib
nvinfer.lib
opencv_world480.lib
```
