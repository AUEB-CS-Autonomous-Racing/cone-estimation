from keypoint_regression.keypoint_regression_model import KeypointRegression
import cv2
from ultralytics import YOLO
from pnp_algorithm import PnP
import matplotlib.pyplot as plt
import time
import numpy as np
import json

COLORS_CV2 = {
    0: (255, 0, 0),    # Blue  - #0000FF
    1: (0, 255, 255),  # Yellow - #FFFF00
    2: (0, 165, 255),  # Orange - #FFA500
    3: (0, 165, 255),  # Orange - #FFA500
}

COLORS_HEX = {
    0: '#0000FF',
    1: '#FFFF00',
    2: '#FFA500',
    3: '#FFA500',
}

def cone_estimation(image_path, demo=True):
    """"
    Demo includes image and plot visualization of keypoints and cone estimates.

    > Full Img 
    > Bounding Box Detection 
    > Crop Cone Imgs to feed into keypoint regression model
    > Keypoint Regression on Cropped Images (x,y output on cropped coordinates) 
    > Translation to full image coordinates
    > PnP
    """

    total_time_start = time.time()
    image = cv2.imread(image_path)

    model_src = 'models/yolo-nano.pt'
    cone_detection_model = YOLO(model_src)
    print("Model loaded.")

    image = cv2.imread(image_path)

    results = cone_detection_model.predict(image)
    results = results[0]

    if demo:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.numpy()[0]
            conf = box.conf.item()
            class_id = int(box.cls.item())
            print([x1, y1, x2, y2], conf, class_id)
            cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), COLORS_CV2[class_id], 1)
        for cone in results.keypoints.data:
            for (x,y) in cone:
                cv2.circle(image, (int(x),int(y)), 1, (0, 0, 255), 1)
        cv2.imshow("img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cones = []
    for cone_points in results.keypoints.data:
        rvec, tvec = PnP(cone_points)
        
        # magic
        tvec /= 1000
        tvec[2] /= 100

        x, y = tvec[0][0], tvec[2][0]
        cones.append((x, y))

    if demo:
        x_coords, y_coords = zip(*cones)
        plt.figure(figsize=(8, 6))
        # Create a scatter plot
        plt.scatter(0, 0, color='red', marker='x', s=100, edgecolor='black')
        plt.scatter(x_coords, y_coords, color='blue', marker='o', s=100, edgecolor='black')

        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Scatter Plot of Points')

        # Optionally, add grid
        plt.grid(True)

        # Display the plot
        plt.show()

    total_time_end = time.time()
    print(f"\nTotal Pipeline Time: {total_time_end-total_time_start:.4}")  

    return cones

if __name__ == '__main__':
    cone_estimation('full_images/00.jpg')
