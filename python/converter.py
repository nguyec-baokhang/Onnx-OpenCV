from ultralytics import YOLO 
import os 
import shutil

model = YOLO('yolo11n.pt')
model.eval()
model.export(format='onnx',simplify=True)

modelPath = '../models'

if os.path.exists(modelPath):
  shutil.move('yolo11n.onnx','../models/yolo11n.onnx')
  os.remove('yolo11n.pt')

