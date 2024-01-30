# Automatic License Plate Recognition Project	
Final project for subject PRP501.5 / MSE FPT

Model:
- YOLOv8 for car detection and license plate detection
- EasyOCR for OCR license plate 

Programming Language: 
- Python

This project aims to train and integrate three deep learning models to create a simple model for automatic license plate recognition from an input image or video.

## Usage
- Clone this repo github, running this in a virtual environment
- Install required dependency
```
pip install -r requirements.txt
```
***NOTICE:***

To ensure optimal performance, it is necessary to make sure that the GPU of your personal computer is utilized during the project execution. In case your computer has a GPU but it is not being utilized, you may consider installing PyTorch cuda. More detailed information can be found here [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
### For car detection and license plate detection on single image
- running [`detect_recognize.py`](detect_recognize.py) after modifying the end of this file as follow. Result will be saved in the directory with name 'Result.png'
```
model = license_plate_recognition()
model.predict_detection('sample.png')
```
### For car detection and license plate detection on video 
- running [`detect_recognize.py`](detect_recognize.py) after modifying the end of this file as follow. Result will be showed directly
```
model = license_plate_recognition()
model.predict_detection_video('sample.mp4')
```
### For car detection, license plate detection and recognition on single image 
- running [`detect_recognize.py`](detect_recognize.py) after modifying the end of this file as follow. Result will be saved in the directory with name 'Result.png'
```
model = license_plate_recognition()
model.predict_detection_ocr('sample.png')
```
### For car detection, license plate detection and recognition on video 
- running [`detect_recognize.py`](detect_recognize.py) after modifying the end of this file as follow. Result of all frames in video will be saved in the folder name 'result_frame'
```
model = license_plate_recognition()
model.predict_detection_ocr('sample.png')
```
- running [`write_video.py`](write_video.py). Remember to specify path to original video and new video. Video result will be saved in the path you have specified
```
cap = cv2.VideoCapture('original_video.mp4')
```
```
out = cv2.VideoWriter('./new_video.mp4', fourcc, fps, (width, height))
``` 

<img src="https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/GUI.jpg">

## Sample results

### Input images
<img src="https://github.com/AnhDuy26/license_plate_recognition/blob/main/images/frame24.png" >

### Result detection
![detection](https://github.com/AnhDuy26/license_plate_recognition/blob/main/images/Result_detect.png)

### Result detection and OCR
![detect and OCR](https://github.com/AnhDuy26/license_plate_recognition/blob/main/images/Result.png)
