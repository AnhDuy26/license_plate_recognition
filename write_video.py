import cv2
import os

# Read original video to get information
cap = cv2.VideoCapture('sample_20p.mp4')
# Get the width of the frames in the video stream
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# Get the height of the frames in the video stream
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Specify the codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Get frame per second of video
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# Initialize VideoWriter object with information from original video, specify name of output video
out = cv2.VideoWriter('./out_sample_20p.mp4', fourcc, fps, (width, height))

# Specify the path to the folder containing images
folder_path = './result_frame'

# List contain all frame name to sort
img_name = []

for filename in os.listdir(folder_path):
    img_name.append(filename)

# Sort image in foler result follow the order to write video
img_name.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# Loop through all image frame to write video
for item in img_name:
    img = cv2.imread("./result_frame/"+str(item))
    out.write(img)

cv2.destroyAllWindows()
out.release()