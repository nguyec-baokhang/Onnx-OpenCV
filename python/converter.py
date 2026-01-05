from ultralytics import YOLO 
import os 
import shutil

model = YOLO('yolov8n.pt')
model.eval()
model.export(format='onnx',simplify=True)

modelPath = '../models'

if os.path.exists(modelPath):
  shutil.move('yolov8n.onnx','../models/yolov8n.onnx')
  os.remove('yolov8n.pt')

