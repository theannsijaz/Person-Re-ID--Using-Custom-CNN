import random
import os
from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import Sampler

class EHTMSampler(Sampler):
    def __init__(self, dataset, batch_size, num_instances=4):
        """
        Args:
            dataset: Your PreGeneratedTripletDataset
            batch_size: Total batch size (must be divisible by num_instances)
            num_instances: Number of instances per person in each batch (K)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_classes = batch_size // num_instances
        
        # Group triplets by person IDs
        self.person_to_triplets = defaultdict(list)
        self._build_person_groups()
        
        # Filter persons with enough triplets
        self.valid_persons = [
            person_id for person_id, triplets in self.person_to_triplets.items()
            if len(triplets) >= self.num_instances
        ]
        
        print(f"EHTMSampler initialized:")
        print(f"  - Total persons: {len(self.person_to_triplets)}")
        print(f"  - Valid persons (>={num_instances} triplets): {len(self.valid_persons)}")
        print(f"  - Batch composition: {self.num_classes} persons × {num_instances} instances")
    
    def _build_person_groups(self):
        """Group triplets by anchor person ID"""
        for idx, (anchor_path, _, _) in enumerate(self.dataset.triplets):
            # Extract person ID from anchor path
            person_id = os.path.basename(os.path.dirname(anchor_path))
            self.person_to_triplets[person_id].append(idx)
    
    def __iter__(self):
        """Generate batches with P persons and K instances each"""
        # Calculate number of batches
        num_batches = len(self.dataset) // self.batch_size
        
        for _ in range(num_batches):
            batch_indices = []
            
            # Randomly select P persons for this batch
            selected_persons = random.sample(self.valid_persons, self.num_classes)
            
            for person_id in selected_persons:
                # Get all triplet indices for this person
                person_triplets = self.person_to_triplets[person_id]
                
                # Randomly sample K instances from this person
                if len(person_triplets) >= self.num_instances:
                    sampled_triplets = random.sample(person_triplets, self.num_instances)
                else:
                    # If not enough triplets, sample with replacement
                    sampled_triplets = random.choices(person_triplets, k=self.num_instances)
                
                batch_indices.extend(sampled_triplets)
            
            # Shuffle the batch to mix different persons
            random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
