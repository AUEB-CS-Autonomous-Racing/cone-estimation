from cnn import cnn
import torch
import cv2
from torchvision import transforms
from ultralytics import YOLO
from pnp_algorithm import pnp
import numpy as np
import matplotlib.pyplot as plt

def keypoint_regression(image):
    keypoint_model_src = 'models/E10-AVL9.9054.pth'
    

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

    # Load the saved model
    model = cnn()  # Assuming 'cnn' is your model class
    model.load_state_dict(torch.load(keypoint_model_src, map_location=torch.device(device)))
    model.eval()  # Set model to evaluation mode


    # Define the transform for preprocessing the images
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((80, 80)),  # Resize the image to match the input size of your model
        transforms.ToTensor(),         # Convert the image to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image
    ])

    # Apply the transform to preprocess the image
    input_image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Move the input image to the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image = input_image.to(device)

    # Perform inference
    with torch.no_grad():  # Disable gradient calculation during inference
        model.eval()  # Set model to evaluation mode
        output = model(input_image)

    # Extract keypoints from the output tensor
    keypoints = output.squeeze().cpu().numpy()  # Assuming output is a tensor

    return keypoints


def main():
    image_path = 'full_images/amz_00000.jpg'
    image = cv2.imread(image_path)

    cone_detection_src = 'models/yolov8s700.pt'
    cone_detection_model = YOLO(cone_detection_src)

    result = cone_detection_model.predict(image)
    bounding_boxes = result[0].boxes

    full_image = cv2.imread(image_path)

    cone_positions = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_img = image[y1:y2, x1:x2]

        cropped_height, cropped_width, _ = cropped_img.shape
        conf = box.conf.item()
        if conf > 0.5 and cropped_height > 10 and cropped_width > 10:
            
            print("Cropped (width x height):", cropped_width, cropped_height)

            keypoints = keypoint_regression(cropped_img)


            # Draw points on the image to mark the keypoints
            keypoints_2d = {}
            print()

            for i in range(0, len(keypoints), 2):
                
                cropped_x = int(keypoints[i] / 100.0 * cropped_width)
                cropped_y = int(keypoints[i+1] / 100.0 * cropped_height)

                full_x = int(cropped_x + x1)
                full_y = int(cropped_y + y1)

                # Add to 2D keypoints map
                keypoints_2d[i//2] = [full_x, full_y]

                cv2.circle(full_image, (full_x, full_y), 2, (0, 0, 255), -1)

            # Estimate cone position
            R, t = pnp(keypoints_2d)
            
            cone_positions.append(t)
    
    cv2.imshow("Keypoints", full_image)
    cv2.waitKey(0)

    # Create a new figure and axis for the 2D plot
    plt.figure()
    plt.title("Estimated Cone Position Relative to Camera")

    # # Plot camera viewpoint (optional)
    plt.scatter(0, 0, color='r', label='Camera')

    # # Plot estimated cone position
    for pos in cone_positions:
        plt.scatter(pos[0], pos[1], color='g')

    # # Set axis labels and legend
    plt.xlabel("X-axis (m)")
    plt.ylabel("Y-axis (m)")
    plt.legend()

    # # Show the plot
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == '__main__':
    main()