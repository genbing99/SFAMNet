
from torch.utils.data import Dataset
import random
random.seed(1)

# Convert dataset into tensor for training and testing
class OFFSTRDataset(Dataset):
    def __init__(self, tensors, transform, train):
        self.tensors = tensors
        self.train = train
        self.transform = transform
        
    def __len__(self):
        return self.tensors[0].shape[0]
    
    def __getitem__(self, index):
        if self.train:
            val = random.randint(90, 110) / 100
            x1 = self.tensors[0][index] * val
            x2 = self.tensors[1][index] * val
            x3 = self.tensors[2][index] * val
        else:
            x1 = self.tensors[0][index]
            x2 = self.tensors[1][index]
            x3 = self.tensors[2][index]
        y  = self.tensors[3][index]
        y1 = self.tensors[4][index]
        return x1, x2, x3, y, y1
    