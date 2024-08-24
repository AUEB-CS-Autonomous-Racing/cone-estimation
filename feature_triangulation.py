import cv2
import numpy as np

""""
> L, R Frames of cone bounding box.
> SIFT Features on both frames.
> KNN Feature matching.
> Triangulation on matched features.
"""


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