# the_dakota_project
Real-Time Object Detection and Description with YOLOv8, spaCy, and gTTS



Here's a description of this GitHub repository:

Real-Time Object Detection and Description with YOLOv8, spaCy, and gTTS
This project leverages computer vision and natural language processing to detect objects in real-time using YOLOv8 and provide descriptive audio feedback using spaCy and gTTS. The application captures video from a webcam, identifies objects, and generates descriptive sentences that are spoken out loud, making it useful for applications such as aiding the visually impaired or enhancing user experience in various computer vision tasks.

Features 👀
> Real-Time Object Detection: Utilizes YOLOv8 for efficient and accurate object detection in real-time.
> Natural Language Descriptions: Uses spaCy to generate human-like descriptive sentences for detected objects.
> Text-to-Speech: Converts the generated descriptions to speech using gTTS, providing audible feedback.
> Dynamic Updates: Only generates and speaks descriptions when new objects are detected or when the set of detected objects changes.
> User-Friendly Display: Visualizes detected objects and corresponding descriptions on the video feed.

Dependencies
opencv-python: For capturing video from the webcam and image processing.
cvzone: For drawing utility functions.
spacy: For natural language processing.
gTTS: For text-to-speech conversion.
ultralytics: For YOLOv8 object detection.


Future Improvements ✅✨
1) Implement support for multiple languages.
2) Add more customization options for text and speech output.
3) Optimize performance for lower-end hardware.
