from keypoint_regression_model import KeypointRegression
import cv2
from ultralytics import YOLO
from pnp_algorithm import PnP
import matplotlib.pyplot as plt
import time
import numpy as np
import json

colors = {
    0: '#0000FF',
    1: '#FFFF00',
    2: '#FFA500',
    3: '#FFA500',
}

def cone_estimation(demo=True):
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

    image_path = 'full_images/amz_00045.jpg'
    image = cv2.imread(image_path)

    cone_detection_src = 'models/yolov8n.pt'
    cone_detection_model = YOLO(cone_detection_src)
    keypoint_model = KeypointRegression('models/E10-AVL9.9054.pth')

    cone_det_start = time.time()
    result = cone_detection_model.predict(image)
    cone_det_end = time.time()
    bounding_boxes = result[0].boxes

    full_image = cv2.imread(image_path)

    cone_estimates = {}
    cropped_cones = []
    id = 0

    # Gather info for each cone and store in cone_estimates
    # Crop cone images
    for box in bounding_boxes:
        label = box.cls.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cone_estimates[id] = {"id": id, "label": label}
        cropped_img = image[y1:y2, x1:x2]
        cropped_cones.append(cropped_img)
        id += 1

    keypoint_reg_start = time.time()
    keypoints = keypoint_model.eval(cropped_cones)
    keypoint_reg_end = time.time()
    print("Keypoint regression finished")

    keypoints_2d = {}
    # Map 7 keypoints for each cone
    for cone in cone_estimates.values():
        # Map keypoint model  output from cropped image to full image coordinates
        # The model output is 14 (x,y) coordinates => 7 keypoints

        id = cone["id"]
        x1, y1, x2, y2 = map(int, bounding_boxes[id].xyxy[0]) # bounding box coordinates in full image
        for point in range(0, len(keypoints[0]), 2):
            cropped_height = y2 - y1
            cropped_width = x2 - x1

            cropped_x = int(keypoints[cone["id"]][point] / 100.0 * cropped_width)
            cropped_y = int(keypoints[cone["id"]][point+1] / 100.0 * cropped_height)
            
            full_x = int(cropped_x + x1)
            full_y = int(cropped_y + y1)

            # Add to 2D keypoints map for PnP
            keypoints_2d[point//2] = [full_x, full_y]

            if demo:
                cv2.putText(full_image, str(cone["id"]), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
                cv2.circle(full_image, (full_x, full_y), 2, (0, 0, 255), -1)

        # Estimate cone position with PnP using all the 2d keypoints for this cone
        rvec, tvec = PnP(keypoints_2d)
        tvec = tvec / 10000
        tvec[2] /= 100

        # 2D Coordinates for cone position in map
        cone["X"] = tvec[0][0]
        cone["Y"] = tvec[2][0]
    

    total_time_end = time.time()
    print(f"\nTotal Pipeline Time: {total_time_end-total_time_start:.4}")  
    print(f"Cone Detection Time: {cone_det_end-cone_det_start:.4}")
    print(f"Total Keypoint Regr. Time: {keypoint_reg_end-keypoint_reg_start:.4}")

    if demo:
        cv2.imshow("Keypoints", full_image)
        cv2.waitKey(0)
        plt.figure()
        plt.title("Estimated Cone Position Relative to Camera")
        plt.scatter(0, 0, color='r', label='Camera')
        plt.xlabel("X-axis")
        plt.ylabel("Depth-axis")
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
    
        for cone in cone_estimates.values():   
            x = cone["X"]
            y = cone["Y"]
            plt.scatter(x, y, color=colors[cone['label']])
            plt.annotate(f'{cone["id"]}', (x, y))

        plt.show()

    return cone_estimates

if __name__ == '__main__':
    cone_estimation()
