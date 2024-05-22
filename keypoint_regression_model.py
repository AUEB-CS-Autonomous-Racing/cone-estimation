from torchvision import transforms
import torch
from cnn import cnn



class KeypointRegression:

    def __init__(self, model_src):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )

        # Load the saved model
        self.model = cnn()  # Assuming 'cnn' is your model class
        self.model.load_state_dict(torch.load(model_src, map_location=torch.device(device)))
        self.model.eval()  # Set model to evaluation mode


        # Define the transform for preprocessing the images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((80, 80)),  # Resize the image to match the input size of your model
            transforms.ToTensor(),         # Convert the image to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image
        ])

    def eval(self, image):
        
        input_image = self.transform(image).unsqueeze(0)  # Add a batch dimension

        # Move the input image to the device (GPU or CPU)
        input_image = input_image.to(self.device)

        with torch.no_grad():  # Disable gradient calculation during inference
            self.model.eval()  # Set model to evaluation mode
            output = self.model(input_image)

        # Extract keypoints from the output tensor
        keypoints = output.squeeze().cpu().numpy() 

        return keypoints