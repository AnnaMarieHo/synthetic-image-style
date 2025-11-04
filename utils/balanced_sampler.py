import torch
from torch.utils.data import Sampler
import numpy as np

class BalancedBatchSampler(Sampler):
    
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        
        # Get indices for each class
        self.real_indices = np.where(labels == 1)[0]
        self.fake_indices = np.where(labels == 0)[0]
        
        # Calculate number of batches
        self.n_batches = min(len(self.real_indices), len(self.fake_indices)) * 2 // batch_size
        
    def __iter__(self):
        # Shuffle indices for each class
        real_shuffled = self.real_indices.copy()
        fake_shuffled = self.fake_indices.copy()
        np.random.shuffle(real_shuffled)
        np.random.shuffle(fake_shuffled)
        
        # Create balanced batches
        half_batch = self.batch_size // 2
        for i in range(self.n_batches):
            # Get half real, half fake
            real_batch = real_shuffled[i*half_batch:(i+1)*half_batch]
            fake_batch = fake_shuffled[i*half_batch:(i+1)*half_batch]
            
            # Combine and shuffle within batch
            batch_indices = np.concatenate([real_batch, fake_batch])
            np.random.shuffle(batch_indices)
            
            # Yield the batch as a list
            yield batch_indices.tolist()
    
    def __len__(self):
        return self.n_batches * self.batch_size

