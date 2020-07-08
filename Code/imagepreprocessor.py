from constants import *
from readcameramodel import readCameraModel
import cv2
from undistortimage import undistortImage
import os

class ImagePreprocessor:
    def __init__(self):
        fx, fy, cx, cy, G_camera_image, LUT = readCameraModel(MODEL_DIR)
        self.intrinsic_camrera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        self.LUT = LUT

    def process_image(self, image):
        '''
        1. demosaicing
        2. undistorting
        3. convert to grayscale
        4. cropping the image to omit the bonnet portion of the car
        '''

        image = cv2.cvtColor(image, cv2.COLOR_BayerGR2BGR)
        undistorted_img = undistortImage(image, self.LUT)
        gray_image = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
        cropped_image = gray_image[:800, :]
        return cropped_image




if __name__ == "__main__":
    image_processor = ImagePreprocessor()
    image_paths = os.listdir(DATA_DIR)
    image_paths.sort()
    frame_count = 1
    for path in image_paths:
        image = cv2.imread((DATA_DIR + path), 0)
        processed_img = image_processor.process_image(image)
        filename = f'{PROCESSED_DATA_DIR}{str(frame_count).zfill(ZERO_PAD)}.png'
        cv2.imwrite(filename, processed_img)
        frame_count += 1
    print('Done!!')

