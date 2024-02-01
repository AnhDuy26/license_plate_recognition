import cv2
from ultralytics import YOLO
import easyocr
import string

class license_plate_recognition:
    def __init__(self):
        self.model_coco = YOLO('./yolov8n.pt')
        self.model_plate = YOLO('./plate_detector.pt')
        self.reader = easyocr.Reader(['en'], gpu=True)
        # Mapping dictionaries for character conversion
        self.dict_char_to_int = {'O': '0',
                                'I': '1',
                                'J': '3',
                                'A': '4',
                                'G': '6',
                                'S': '5'}

        self.dict_int_to_char = {'0': 'O',
                                '1': 'I',
                                '3': 'J',
                                '4': 'A',
                                '6': 'G',
                                '5': 'S'}

    def predict_detection(self,file):
        """
        Read the input image and return image with car detection, license plate detection

        Args:
            file (image): input image containing car and license plate

        Returns:
            tuple: Saving result image with name Result.png
        """
        results = self.model_coco.predict(file, show_labels=False, classes=[2,3,5,7])
        annotated_frame = results[0].plot(labels=False)
        results_detect = self.model_plate.predict(file, show_labels=False)
        # Visualize the results on the frame
        annotated_frame_detect = results_detect[0].plot(labels=False)
        # Concatenate the annotated frames horizontally
        alpha = 0.5
        beta = 1-alpha
        combined_frame = cv2.addWeighted(annotated_frame, alpha, annotated_frame_detect, beta, 0.0)

        cv2.imwrite("Result_detect.png", combined_frame)

    def predict_detection_video(self,file):
        """
        Read the input video and show video result with car detection, license plate detection

        Args:
            file (video): input video containing car and license plate

        Returns:
            tuple: Show video result with car detection and license plate detection
        """
        cap = cv2.VideoCapture(file)
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                # Model license plate
                results = self.model_plate(frame)

                # Visualize the results on the frame
                annotated_frame = results[0].plot(labels=False)
                # Model coco
                results_coco = self.model_coco(frame)

                # Visualize the results on the frame
                annotated_frame_coco = results_coco[0].plot(labels=False)

                # Concatenate the annotated frames horizontally
                alpha = 0.5
                beta = 1 - alpha
                combined_frame = cv2.addWeighted(annotated_frame, alpha, annotated_frame_coco, beta, 0.0)
                rs = cv2.resize(combined_frame, (960, 540))
                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", rs)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break
        cap.release()
        cv2.destroyAllWindows()

    def license_complies_format(self, text):
        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """
        if len(text) != 7:
            return False

        if (text[0] in string.ascii_uppercase or text[0] in self.dict_int_to_char.keys()) and \
                (text[1] in string.ascii_uppercase or text[1] in self.dict_int_to_char.keys()) and \
                (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in self.dict_char_to_int.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in self.dict_char_to_int.keys()) and \
                (text[4] in string.ascii_uppercase or text[4] in self.dict_int_to_char.keys()) and \
                (text[5] in string.ascii_uppercase or text[5] in self.dict_int_to_char.keys()) and \
                (text[6] in string.ascii_uppercase or text[6] in self.dict_int_to_char.keys()):
            return True
        else:
            return False

    def format_license(self, text):
        """
        Format the license plate text by converting characters using the mapping dictionaries.

        Args:
            text (str): License plate text.

        Returns:
            str: Formatted license plate text.
        """
        license_plate_ = ''
        mapping = {0: self.dict_int_to_char, 1: self.dict_int_to_char, 4: self.dict_int_to_char, 5: self.dict_int_to_char,
                   6: self.dict_int_to_char,
                   2: self.dict_char_to_int, 3: self.dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

        return license_plate_

    def read_license_plate(self, license_plate_crop):
        """
        Read the license plate text from the given cropped image.

        Args:
            license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

        Returns:
            tuple: Tuple containing the formatted license plate text and its confidence score.
        """

        detections = self.reader.readtext(license_plate_crop)
        # When easyOCR cannot detect
        try:
            text = detections[0][1]
            text = text.upper().replace(' ', '')
            if self.license_complies_format(text):
                text = self.format_license(text)
                return text
            return None
        except:
            return None

    def predict_detection_ocr(self,file):
        """
        Read the input image and return image with car detection, license plate detection and recognition

        Args:
            file (image): input image containing car and license plate

        Returns:
            tuple: Saving result image with name Result.png
        """
        results = self.model_coco.predict(file, show_labels=False, classes=[2,3,5,7])
        annotated_frame = results[0].plot(labels=False)
        results_detect = self.model_plate.predict(file, show_labels=False)
        # Visualize the results on the frame
        annotated_frame_detect = results_detect[0].plot(labels=False)
        # Concatenate the annotated frames horizontally
        alpha = 0.5
        beta = 1-alpha
        combined_frame = cv2.addWeighted(annotated_frame, alpha, annotated_frame_detect, beta, 0.0)

        # OCR license plates
        license_plates = self.model_plate(file)[0]
        im = cv2.imread(file)
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            license_plate_crop = im[int(y1):int(y2), int(x1): int(x2), :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                         cv2.THRESH_BINARY_INV)
            # results_ocr = self.read_license_plate(license_plate_crop_thresh)
            results_ocr = self.read_license_plate(license_plate_crop)
            if results_ocr is not None:
                # Position of text
                top_left = tuple([int(x1),int(y1-10)])
                text = results_ocr
                font = cv2.FONT_HERSHEY_SIMPLEX
                combined_frame = cv2.putText(combined_frame, text, top_left, font, 2, (255, 0, 0), 3, cv2.LINE_AA)

        # Save the annotated
        cv2.imwrite("Result.png", combined_frame)

    def predict_detection_ocr_video(self,file):
        """
        Read the input video and return all frames of the input video with car detection, license plate detection and recognition

        Args:
            file (video): input video containing car and license plate

        Returns:
            tuple: Saving results of all frames of input video in folder result_frame
        """
        frame_nmr = -1
        cap = cv2.VideoCapture(file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret = True
        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if ret:
                results = self.model_coco.predict(frame, show_labels=False, classes=[2,3,5,7])
                annotated_frame = results[0].plot(labels=False)
                results_detect = self.model_plate.predict(frame, show_labels=False)
                # Visualize the results on the frame
                annotated_frame_detect = results_detect[0].plot(labels=False)
                # Concatenate the annotated frames horizontally
                alpha = 0.5
                beta = 1-alpha
                combined_frame = cv2.addWeighted(annotated_frame, alpha, annotated_frame_detect, beta, 0.0)
                # OCR license plates
                license_plates = self.model_plate(frame)[0]
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                                 cv2.THRESH_BINARY_INV)
                    results_ocr = self.read_license_plate(license_plate_crop_thresh)
                    if results_ocr is not None:
                        # Position of text
                        top_left = tuple([int(x1),int(y1-10)])
                        text = results_ocr
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        combined_frame = cv2.putText(combined_frame, text, top_left, font, 2, (255, 0, 0), 3, cv2.LINE_AA)
                # cv2.imwrite("result_frame/result"+str(frame_nmr)+".png", combined_frame)
                cv2.imwrite("test_pil/result" + str(frame_nmr) + ".png", combined_frame)
        cap.release()


model = license_plate_recognition()
# model.predict_detection_ocr_video('sample_20p.mp4')
model.predict_detection_video('sample_20p.mp4')
# model.predict_detection('frame24.png')