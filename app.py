from flask import Flask, request
import cv2
from ultralytics import YOLO


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    file.save("img1.jpg")
    image_path = 'img1.jpg' 
    model_path = "best.pt"
    objl = []
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    #H, W, _ = image.shape
    results = model(image, verbose=False)[0]
    threshold = 0.1         
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            if results.names[int(class_id)] == "book":
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                obj={}
                obj["paper_book"] = [int(x1),int(y1),int(x2),int(y2)]
                objl.append(obj)
                cv2.putText(image, "PAPER_BOOK", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
                
            if results.names[int(class_id)] =="laptop":
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                obj={}
                obj["e_laptop"] = [int(x1),int(y1),int(x2),int(y2)]
                objl.append(obj)
                cv2.putText(image, "E_LAPTOP", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
                
            if results.names[int(class_id)] =="cell phone":
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                obj={}
                obj["e_phone"] = [int(x1),int(y1),int(x2),int(y2)]
                objl.append(obj)
                cv2.putText(image, "E_PHONE", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
                
            if results.names[int(class_id)] =="bottle":
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                obj={}
                obj["plastic_bottle"] = [int(x1),int(y1),int(x2),int(y2)]
                objl.append(obj)
                cv2.putText(image, "PLASTIC_BOTTLE", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)


            
    print(objl) 
    return objl

