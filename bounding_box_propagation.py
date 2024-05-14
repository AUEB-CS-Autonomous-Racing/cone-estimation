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

colors = {
    0: '#0000FF',
    1: '#FFFF00',
    2: '#FFA500',
    3: '#FFA500',
}

def main():
    image_path = 'full_images/amz_00001.jpg'
    image = cv2.imread(image_path)

    cone_detection_src = 'models/yolov8s700.pt'
    cone_detection_model = YOLO(cone_detection_src)

    result = cone_detection_model.predict(image)
    bounding_boxes = result[0].boxes

    full_image = cv2.imread(image_path)

    cone_positions = []
    id = 0
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
            keypoints = keypoint_regression(cropped_img)

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
            R, t = pnp(keypoints_2d)
            
            cone_positions.append({"id": id, "label": label, "pos": t})    

    cv2.imshow("Keypoints", full_image)
    cv2.waitKey(0)

    # Create a new figure and axis for the 2D plot
    plt.figure()
    plt.title("Estimated Cone Position Relative to Camera")

    # camera viewpoint
    plt.scatter(0, 0, color='r', label='Camera')

    # # Plot estimated cone position
    for cone in cone_positions:
        plt.scatter(cone["pos"][0], cone["pos"][1], color=colors[cone['label']])
        plt.annotate(f"{cone["id"]}", (cone["pos"][0], cone["pos"][1]+3))

    # # Set axis labels and legend
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == '__main__':
    main()