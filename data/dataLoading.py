from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import numpy as np
import random

class collocation:
    def __init__(self, coord, point_type, label):
        self.coord = coord # (x, y)
        self.point_type = point_type # interior, or boundary
        self.label = label # e.g. (u, v, p)

# Intake point_cloud should be a 
class pointCloudDataset(Dataset):
    def __init__(self, coords, known_values, point_types):
        # Store data on CPU for DataLoader compatibility
        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.known_values = torch.tensor(known_values, dtype=torch.float32)
        self.point_types = point_types
    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return self.coords[idx], self.known_values[idx], self.point_types[idx]

class pointCloudCollection:
    def __init__(self, point_cloud, device):
        coords, point_types, known_values, ground_truth = point_cloud
        
        # Convert to numpy if pandas
        if hasattr(coords, 'values'):
            coords = coords.values
        if hasattr(known_values, 'values'):
            known_values = known_values.values
        if hasattr(ground_truth, 'values'):
            ground_truth = ground_truth.values
            
        # Dataset stores data on CPU, GPU transfer happens in DataLoader
        self.init_dataset = pointCloudDataset(coords, known_values, point_types)
        self.ground_truth = torch.tensor(ground_truth, dtype=torch.float32, device=device)
    

    
if __name__ == '__main__':
    from .plaque_flow import preprocessing
    point_cloud_path = 'data/point_cloud.xlsx'
    cfd_results_path = 'data/cfd_results.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    point_cloud_dict, point_cloud, cfd_results = preprocessing(point_cloud_path, cfd_results_path)
    pass