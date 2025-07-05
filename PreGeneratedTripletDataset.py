import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch

class PreGeneratedTripletDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        self.person_ids = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        
        # Create a mapping from string IDs to numeric IDs
        self.id_to_numeric = {pid: idx for idx, pid in enumerate(sorted(self.person_ids))}
        
        self.person_images = {pid: [] for pid in self.person_ids}
        
        for person_id in self.person_ids:
            person_dir = os.path.join(data_path, person_id)
            self.person_images[person_id] = [
                os.path.join(person_dir, img) 
                for img in os.listdir(person_dir) 
                if img.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        
        # Keep only person IDs that have at least 2 images (to form triplets)
        self.valid_person_ids = [pid for pid in self.person_ids if len(self.person_images[pid]) >= 2]
        
        # Precompute triplets once instead of dynamically creating them each epoch
        self.triplets = self._generate_triplets()

    def _generate_triplets(self):
        triplets = []
        
        for anchor_id in self.valid_person_ids:
            anchor_imgs = self.person_images[anchor_id]
            
            for anchor_img in anchor_imgs:
                positive_candidates = [img for img in anchor_imgs if img != anchor_img]
                if not positive_candidates:
                    continue
                positive_img = random.choice(positive_candidates)
                
                negative_id = random.choice([pid for pid in self.valid_person_ids if pid != anchor_id])
                negative_img = random.choice(self.person_images[negative_id])
                
                triplets.append((anchor_img, positive_img, negative_img))
        
        return triplets

    def load_image(self, image_path):
        """Load and return PIL Image from path"""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), (0, 0, 0))

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]
        
        # Load images and convert to tensors (CPU)
        anchor = self.load_image(anchor_path)
        positive = self.load_image(positive_path)
        negative = self.load_image(negative_path)
        
        # Basic CPU transforms (resize, to tensor)
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        # Extract person IDs from the file paths
        anchor_id = os.path.basename(os.path.dirname(anchor_path))
        negative_id = os.path.basename(os.path.dirname(negative_path))
        
        # Convert string IDs to integers (Market 1501 has numeric IDs like "0001")
        anchor_id_int = int(anchor_id)
        negative_id_int = int(negative_id)
        
        # Return tensors and labels (GPU augmentation will be applied in training loop)
        return anchor, positive, negative, torch.tensor(anchor_id_int), torch.tensor(negative_id_int)
