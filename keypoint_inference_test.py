from cnn import cnn
import torch
import cv2
from PIL import Image
from torchvision import transforms
import os
import random

from pnp_algorithm import pnp

model_src = 'models/E10-AVL9.9054.pth'
device = (
"cuda"
if torch.cuda.is_available()
else "mps"
if torch.backends.mps.is_available()
else "cpu"
)

# Load the saved model
model = cnn()  # Assuming 'cnn' is your model class
model.load_state_dict(torch.load(model_src, map_location=torch.device(device)))
model.eval()  # Set model to evaluation mode


# Define the transform for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((80, 80)),  # Resize the image to match the input size of your model
    transforms.ToTensor(),         # Convert the image to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image
])

# Define the path to the directory containing images
image_dir = "data/images/"

# Get a list of image files in the directory
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg')]

# Initialize index to track the current image
current_image_index = 0

# Initialize OpenCV window
cv2.namedWindow("Image with Keypoints", cv2.WINDOW_NORMAL)

print("Press SPACE for next image.")
print("Press q to quit.")


while True:
    # Load an image for inference
    image_path = image_files[current_image_index]
    image = Image.open(image_path).convert('RGB')  # Open the image and convert it to RGB

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

    # Open the image using OpenCV
    image_cv2 = cv2.imread(image_path)
    height, width, channels = image_cv2.shape

    # Draw circles or points on the image to mark the keypoints
    keypoints_2d = {}
    print()

    for i in range(0, len(keypoints), 2):
        x = int(keypoints[i] / 100.0 * width)
        y = int(keypoints[i+1] / 100.0 * height)
        cv2.putText(image_cv2, str(i//2), (x, y+10), 0, 1, (0,0,255), 2)

        # Add to 2D keypoints map
        keypoints_2d[i//2] = [x, y]

        cv2.circle(image_cv2, (x, y), 3, (0, 0, 255), -1)  # Draw a green circle at each keypoint

    # Test PnP Algorithm
    pnp(keypoints_2d)

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