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
        self.image_paths = []  
        self.labels = []     
        self._load_data()

    def _load_data(self):
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    self.image_paths.append(os.path.join(root, file))
                    self.labels.append(self._extract_label(root))

    def _extract_label(self, path):
        label = os.path.basename(path) 
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
