from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(data = "data.yaml",    
                epochs = 200,           
                imgsz = 1920,          
                batch = 1,             
                device = "cuda")

    
    
    