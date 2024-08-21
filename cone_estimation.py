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

    cones = []
    for box, keypoints in zip(results.boxes, results.keypoints.data):
        
        bounding_box_left = box.xyxy.numpy()[0]

        if demo:
            x1, y1, x2, y2 = bounding_box_left
            conf = box.conf.item()
            class_id = int(box.cls.item())
            cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), COLORS_CV2[class_id], 1)
            for (x,y) in keypoints:
                cv2.circle(image, (int(x),int(y)), 1, (0, 0, 255), 1)

        rvec, tvec = PnP(keypoints)
        image_points_right, bounding_box_right = bounding_box_propagation(rvec, tvec)
        
        if demo:
            for point in image_points_right:
                    x, y = point[0][0], point[0][1]
                    cv2.circle(image, (int(x),int(y)), 2, (255, 0, 0), 1)
                    cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (0, 255, 0), 1, lineType=1)

        # Extract SIFT features for triangulation
        image_left = image # temporarily for demo purposes
        image_right = image
        keypoints_left, descriptors_left = extract_sift_features(image_left, bounding_box_left)
        keypoints_right, descriptors_right = extract_sift_features(image_right, bounding_box_right)

        good_matches = match_features(descriptors_left, descriptors_right)
        if good_matches is None:
            continue
        # matched points
        pts1 = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches])

        # Camera calibration data
        K1 = np.array([[895.45613853, 0, 682.3924525], [0, 667.03818821, 360.49268128], [0, 0, 1]])
        K2 = np.array([[914.0586327, 0, 679.06288554], [0, 680.80769983, 361.59960622], [0, 0, 1]])

        # Stereo camera parameters (Example values; replace with your own)
        R = np.eye(3)  # Rotation matrix between left and right cameras
        T = np.array([0.1, 0, 0])  # Translation vector between left and right cameras
    
        # Triangulate points
        points_3d = triangulate_points(pts1, pts2, K1, K2, R, T)
    
        # Apply median filtering to 3D points
        points_3d_filtered = np.median(points_3d, axis=0)

        cones.append(points_3d_filtered)

        print(f"3D Points: {points_3d_filtered}")

        # magic
        # tvec /= 1000
        # tvec[2] /= 100

        # x, y = tvec[0][0], tvec[2][0]
        # cones.append((x, y))

    cv2.destroyAllWindows()

    if demo:
        X, Y, Z = zip(*cones)
        print(X)
        X /= 1000
        Z /= 1000
        Z /= 100
        # tvec /= 1000
        # tvec[2] /= 100
        plt.figure(figsize=(8, 6))
        # Create a scatter plot
        plt.scatter(0, 0, color='red', marker='x', s=100)
        plt.scatter(X, Z, color='blue', marker='o', s=100, edgecolor='black')

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

def bounding_box_propagation(pnp_rvec, pnp_tvec):

    # Project the 3D points to the right camera's image plane
    cone_3d_points = np.array([
        [-114, 0, 0], 
        [-69, 157, 0],
        [-39, 265, 0],
        [0, 325, 0],
        [114, 0, 0],
        [69, 157, 0],
        [39, 265, 0]
    ], dtype=np.float32)

    # Use with real camera.
    # camera_matrix_right = np.array([[895.45613853, 0, 682.3924525],
    #                                 [0, 667.03818821, 360.49268128],
    #                                 [0, 0, 1]])
    
    K = get_camera_matrix()
    distortion_coeffs_right = np.array([-0.01005559, 0.0777699, 0.00040632, 0.00196239, -0.10160105])
    image_points_right, _ = cv2.projectPoints(cone_3d_points, pnp_rvec, pnp_tvec, K, distortion_coeffs_right)

    x_left = min(image_points_right[:, 0, 0])
    x_right = max(image_points_right[:, 0, 0])
    y_bottom = max(image_points_right[:, 0, 1])
    y_top = min(image_points_right[:, 0, 1])

    # Scale box
    SCALING_FACTOR = 1.5

    center_x = (x_left + x_right) / 2
    center_y = (y_bottom + y_top) / 2
    width = x_right - x_left
    height = y_bottom - y_top

    scaled_width = width * SCALING_FACTOR
    scaled_height = height * SCALING_FACTOR

    # scaled coordinates
    x_left = center_x - scaled_width / 2
    x_right = center_x + scaled_width / 2
    y_bottom = center_y + scaled_height / 2
    y_top = center_y - scaled_height / 2

    bounding_box = (x_left, y_top, x_right, y_bottom)
    return image_points_right, bounding_box


def get_camera_matrix():

    # IMX219-83 intrinsics
    f_mm = 2.6  # Focal length in mm
    W = 3280    # Image width in pixels
    H = 2464    # Image height in pixels
    CMOS_width = 1/4  # CMOS size in inches

    # ETH intrinsics
    f_mm = 10
    W = 1600
    H = 1200
    CMOS_width = 1/4


    # Convert focal length from mm to pixels
    f_px = f_mm * W / CMOS_width

    # Calculate principal point
    cx = W / 2
    cy = H / 2

    K = np.array([[f_px, 0, cx],
              [0, f_px, cy],
              [0, 0, 1]])
              
    
    return K

def extract_sift_features(image, bounding_box):
    """Extract SIFT features within a given bounding box."""
    x1, y1, x2, y2 = map(int, bounding_box)
    cropped_image = image[y1:y2, x1:x2]
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(cropped_image, None)
    
    keypoints = [cv2.KeyPoint(kp.pt[0] + x1, kp.pt[1] + y1, kp.size) for kp in keypoints]
    
    return keypoints, descriptors

def match_features(desc1, desc2):
    """Match features using KNN."""
    if desc1 is None or desc2 is None:
        # Return an empty list if any of the descriptors are empty
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) != 2: # need at least 2 elements
            return None
        m, n = match_pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    return good_matches

def triangulate_points(pts1, pts2, K1, K2, R, T):
    """Triangulate points from two images."""
    P1 = np.hstack((K1, np.zeros((3, 1))))
    P2 = np.hstack((K2 @ np.hstack((R, T.reshape(-1, 1))), np.zeros((3, 1))))


    RT = np.hstack((R, T.reshape(-1, 1))) 
    P2 = K2 @ RT

    pts1 = pts1.T  # Transpose to (2, N)
    pts2 = pts2.T
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    
    return points_3d.T

cone_estimation('full_images/00.jpg')
