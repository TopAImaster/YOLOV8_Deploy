import onnx
import onnxsim

f=r"yolov8m_smi3.onnx"
model=onnx.load("yolov8m.onnx")

print(model.ir_version)

onnx_mdel,check=onnxsim.simplify(model)

assert check,"assert check fail"
onnx.save(onnx_mdel,f)

print("end")