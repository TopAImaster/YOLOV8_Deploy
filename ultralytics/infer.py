from ultralytics import YOLO
import torch
import torch.nn as nn


# 加载模型
model = YOLO("yolov7m.yaml")  # 从头开始构建新模型
model = YOLO("yolov7m.pt")  # 加载预训练模型（建议用于训练）
#metrics = model.val()  # 在验证集上评估模型性能
#results = model("dog2.jpeg") 
success = model.export(format="onnx")




print("end")