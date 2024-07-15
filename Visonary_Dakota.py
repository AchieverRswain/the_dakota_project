import spacy
import cv2
import cvzone
import math
import random
from gtts import gTTS
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Initialize YOLO for object detection (assuming you have already configured YOLO)
from ultralytics import YOLO
model = YOLO('yolov8m.pt')

# COCO classes for YOLO
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
    "hair drier", "toothbrush"
]

prev_objects = []
# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera.")
        continue

    # Perform object detection with YOLO
    results = model(img, stream=True)

    current_objects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])
            if conf>0.8:
                current_objects.append(coco_classes[cls])

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))


            if current_objects != prev_objects:
    # Generate a descriptive sentence using spaCy
                if current_objects:
        # Construct a general prompt describing the detected objects
                    object_descriptions = ', '.join(current_objects)

                    template =[ 
                                f"Oh, I see a {object_descriptions}.",
                                f"Looks like there's a {object_descriptions} over there.",
                                f"Hey, check out the {object_descriptions}!",
                                f"I'm seeing a {object_descriptions}.", 
                                 f"Hmm, I see a {object_descriptions}.",
                                f"Interesting! There's a {object_descriptions} right there.",
                                f"Ah, that's a {object_descriptions}.",
                                f"Looks like we have {object_descriptions}."
                 
                                        ]

                    prompt = random.choice(template)
# Generate descriptive sentence using spaCy
                    doc = nlp(prompt)
                    descriptive_sentence = ""
                    for sent in doc.sents:
                      descriptive_sentence = sent.text

                    tts = gTTS(text=descriptive_sentence, lang='en')
                    tts.save("output.mp3")
                    os.system("afplay output.mp3")        

                    prev_objects = current_objects
        
        
               
        
            
                  

        # Display the descriptive sentence on the image
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            cvzone.putTextRect(img,f'{coco_classes[cls]} {conf}',(max(0,x1),max(0,y1)))
            cv2.putText(img, descriptive_sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

     

    cv2.imshow("snapshot", img)
    
    key = cv2.waitKey(1)
    if key == ord('c'):
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
