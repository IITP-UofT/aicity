from ultralytics import YOLO

def train():
    model = YOLO('yolov8x.pt')
    # default imgsz(nHD): 640*260, original imgsz(FHD) = 1920 * 1080, input imgsz(HD) = 1280 * 720
    results = model.train(data='./cfg/yolov8.yaml', 
                          epochs=1, 
                          imgsz=(1280, 720), 
                          device=[0, 1],
                          save=True,
                          save_period=1,
                          seed=0,
                          project="practices",
                          name="practice_1"
                          )
    
if __name__=="__main__":
    train()
