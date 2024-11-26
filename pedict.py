from ultralytics import YOLO

model = YOLO("yolo11m-seg-custom.pt")

#model.predict(source="3.jpg", show = True , save = True,
 #   conf = 0.7, line_width = 2, save_crop= True, save_txt = True,
  #   show_labels = True, show_conf = True, classes = [0,1] )


model.predict(source="brain_tumor.mp4", show = True , save = True,
    conf = 0.7, line_width = 2, save_crop= False, save_txt = False,
     show_labels = True, show_conf = True, classes = [0,1] )

