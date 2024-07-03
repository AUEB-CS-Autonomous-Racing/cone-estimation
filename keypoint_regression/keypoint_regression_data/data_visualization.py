import os
import cv2
from PIL import Image
import json
import random

# Define the path to the directory containing images
image_dir = "images/"
annotations = "ann.json"

# Get a list of image files in the directory
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg')]

# Initialize index to track the current image
current_image_index = 0

# Initialize OpenCV window
cv2.namedWindow("Image with Keypoints", cv2.WINDOW_NORMAL)

# Load annotations from JSON file
with open(annotations, 'r') as f:
    annotations_data = json.load(f)

print("Press SPACE for next image.")
print("Press q to quit.")


while True:
    
    # Load an image for inference
    image_path = image_files[current_image_index]
    image = Image.open(image_path).convert('RGB')  # Open the image and convert it to RGB

    # Extract keypoints from annotations JSON for the current image
    current_image_annotations = [ann for ann in annotations_data if ann['img'] == os.path.basename(image_path)]
    keypoints = current_image_annotations[0]['kp-1'] if current_image_annotations else []

    print(f"\n{image_path}\nResolution: {image.size}\n{keypoints}\n")

    # Open the image using OpenCV
    image_cv2 = cv2.imread(image_path)

    # Draw circles or points on the image to mark the keypoints
    for kp in keypoints:
        x = int(kp['x'] / 100.0 * kp['original_width'])
        y = int(kp['y'] / 100.0 * kp['original_height'])
        cv2.circle(image_cv2, (x, y), 3, (0, 0, 255), -1)  # Draw a green circle at each keypoint

    # Display the image with the marked keypoints
    cv2.imshow("Image with Keypoints", image_cv2)

    # Wait for key press
    key = cv2.waitKey(0) & 0xFF

    # If the spacebar is pressed, proceed to the next image
    if key == ord(' '):
        current_image_index = random.randrange(len(image_files))
    # If the 'q' key is pressed, exit the loop
    elif key == ord('q'):
        break

# Close the OpenCV window
cv2.destroyAllWindows()
