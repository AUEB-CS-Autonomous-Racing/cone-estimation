from cnn import cnn
from keypoint_regression_model import KeypointRegression
import torch
import cv2
from torchvision import transforms
from ultralytics import YOLO
from pnp_algorithm import pnp
import numpy as np
import matplotlib.pyplot as plt
import time

colors = {
    0: '#0000FF',
    1: '#FFFF00',
    2: '#FFA500',
    3: '#FFA500',
}

def main():

    total_time_start = time.time()

    image_path = 'full_images/amz_00000.jpg'
    image = cv2.imread(image_path)

    cone_detection_src = 'models/yolov8s700.pt'
    cone_detection_model = YOLO(cone_detection_src)

    cone_det_start = time.time()
    result = cone_detection_model.predict(image)
    cone_det_end = time.time()
    bounding_boxes = result[0].boxes

    full_image = cv2.imread(image_path)

    cone_positions = []
    id = 0
    
    keypoint_reg_time = []

    keypoint_model = KeypointRegression('models/E10-AVL9.9054.pth')

    for box in bounding_boxes:
        id += 1
        conf = box.conf.item()
        label = box.cls.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_img = image[y1:y2, x1:x2]

        cropped_height, cropped_width, _ = cropped_img.shape
        conf = box.conf.item()
        
        if conf > 0.2:
            print("Cropped (width x height):", cropped_width, cropped_height, "\n")

            keypoint_reg_start = time.time()
            keypoints = keypoint_model.eval(cropped_img)
            keypoint_reg_end = time.time()
            keypoint_reg_time.append(keypoint_reg_end-keypoint_reg_start)

            # Draw points on the image to mark the keypoints
            keypoints_2d = {}
            cv2.putText(full_image, str(id), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
            
            for i in range(0, len(keypoints), 2):
                
                cropped_x = int(keypoints[i] / 100.0 * cropped_width)
                cropped_y = int(keypoints[i+1] / 100.0 * cropped_height)

                full_x = int(cropped_x + x1)
                full_y = int(cropped_y + y1)

                # Add to 2D keypoints map
                keypoints_2d[i//2] = [full_x, full_y]

                cv2.circle(full_image, (full_x, full_y), 2, (0, 0, 255), -1)

            # Estimate cone position
            print(f"id: {id}")
            rvec, tvec = pnp(keypoints_2d)
            tvec = tvec / 10000
            tvec[2] /= 100
            print(f"Translation Vector:\n{tvec}")

            cone_positions.append({"id": id, "label": label, "pos": tvec})  

    total_time_end = time.time()
    print()
    print(f"Total Pipeline Time: {total_time_end-total_time_start:.4}")  
    print(f"Cone Detection Time: {cone_det_end-cone_det_start:.4}")
    print(f"Average Keypoint Regr.Time Per Box: {np.mean(keypoint_reg_time):.4}")
    print(f"Total Keypoint Regr. Time: {sum(keypoint_reg_time):.4}")

    cv2.imshow("Keypoints", full_image)
    cv2.waitKey(0)

    # Create a new figure and axis for the 2D plot
    plt.figure()
    plt.title("Estimated Cone Position Relative to Camera")

    # camera viewpoint
    plt.scatter(0, 0, color='r', label='Camera')

    # # Plot estimated cone position
    for cone in cone_positions:
        
        x = cone["pos"][0]
        z = cone["pos"][2]

        plt.scatter(x, z, color=colors[cone['label']])
        plt.annotate(f'{cone["id"]}', (x, z))

    # # Set axis labels and legend
    plt.xlabel("X-axis")
    plt.ylabel("Z-axis")
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    main()
