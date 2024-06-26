import torch.optim as optim
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from cnn import cnn
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from multiprocessing import freeze_support
from torch.utils.data import random_split

# Assuming your data is in data_dir that contains an ann.json file and an images directory
data_dir = 'data'

class CustomDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None):
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.annotations = self.load_annotations()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations[idx]['img']
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        annotation = self.annotations[idx]['kp-1']

        # Extract keypoints' coordinates and convert them into a tensor
        keypoints = []
        for kp in annotation:
            x_coord = kp['x']
            y_coord = kp['y']
            keypoints.extend([x_coord, y_coord])
        labels = torch.tensor(keypoints, dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, labels

    def load_annotations(self):
        with open(self.annotations_file, 'r') as f:
            annotations = json.load(f)
        return annotations
    
if __name__ == '__main__':
    freeze_support()

        
    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(80),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(80),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }



    # Load the dataset
    custom_dataset = CustomDataset(data_dir=os.path.join(data_dir, 'images'),
                                   annotations_file=os.path.join(data_dir, 'ann.json'),
                                   transform=data_transforms['train'])

    # Split the dataset into train and validation
    dataset_size = len(custom_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    # Create data loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    }

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

    # Initialize the CNN
    model = cnn()
    model = model.to(device)

    # Define Hyperparameters
    EPOCHS = 1
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9

    output_shape = model(torch.randn(1, 3, 80, 80)).shape
    print("Model Output shape:", output_shape)

    # Define a Loss function and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


    # Train the network
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloaders['train'], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)
            labels = data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        # Print loss statistics
        print('[%d] epoch loss: %.3f' % (epoch + 1, running_loss / len(dataloaders['train'])))
        running_loss = 0.0

    print('Finished Training')

    # Evaluate and save model
    model.eval()  # Set model to evaluation mode
    val_losses = []

    with torch.no_grad():
        for i, data in enumerate(dataloaders['val'], 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss (if needed)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            
    # Calculate average validation loss
    avg_val_loss = sum(val_losses) / len(val_losses)
    print('Average validation loss:', avg_val_loss)
    torch.save(model.state_dict(), 'models/' \
            f'E{EPOCHS}-AVL{avg_val_loss:.4f}.pth')
    print("Model saved.")