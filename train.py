from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO("yolov8s.yaml")  
model =YOLO("last.pt")
if __name__ == '__main__':
    model.train(data="data.yaml", epochs=400,batch =8)  
