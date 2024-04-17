from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
from PIL import Image
import os

class MonetDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(MonetDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = []  # List to store paths of images
        self.labels = []       # List to store labels (optional)

        # Populate image_paths and labels here
        self._load_data()

    def _load_data(self):
        # Implement logic to load image paths and labels
        # Example:
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    self.image_paths.append(os.path.join(root, file))
                    # Assuming labels are encoded in directory names or file names
                    # You can modify this according to your dataset structure
                    self.labels.append(self._extract_label(root))

    def _extract_label(self, path):
        # Implement logic to extract label from path
        # Example:
        label = os.path.basename(path)  # Extracting label from directory name
        return label

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index] if self.labels else None
        
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.image_paths)

# Example usage:
# Define transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Create custom dataset instance
# dataset = CustomDataset(root='path/to/your/dataset', transform=transform)

# # Accessing data
# image, label = dataset[0]  # Accessing first data sample
