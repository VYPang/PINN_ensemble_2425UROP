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
        
        # Convert point_types to numpy array for proper integer indexing
        if hasattr(point_types, 'values'):
            self.point_types = point_types.values
        elif hasattr(point_types, 'reset_index'):
            self.point_types = point_types.reset_index(drop=True).values
        else:
            self.point_types = np.array(point_types)
            
        self.idx = np.arange(len(self.coords))
        self.set_interior_dataset()
    
    def set_interior_dataset(self):
        interior_mask = self.point_types == 'interior'
        interior_coords = self.coords[interior_mask]
        interior_known_values = self.known_values[interior_mask]
        interior_point_types = self.point_types[interior_mask]
        self.interior_idx = self.idx[interior_mask]
        
        # Convert filtered data to proper format for interiorDataset
        if len(interior_coords) > 0:
            self.interior_dataset = interiorDataset(
                interior_coords.cpu().numpy(), 
                interior_known_values.cpu().numpy(), 
                interior_point_types
            )
        else:
            # Create empty dataset if no interior points found
            self.interior_dataset = interiorDataset(
                np.empty((0, 2)), 
                np.empty((0, 3)), 
                np.array([])
            )
    
    def get_interior_dataset(self):
        return self.interior_dataset, self.interior_idx
    
    def save_pseudo_labels(self, pseudo_labels, idx):
        # Save pseudo labels to the known_values tensor at the specified indices
        self.known_values[idx] = torch.tensor(pseudo_labels, dtype=torch.float32)
        self.set_interior_dataset()  # Update interior dataset after saving pseudo labels

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return self.coords[idx], self.known_values[idx], self.point_types[idx]

class interiorDataset(pointCloudDataset):
    def __init__(self, coords, known_values, point_types):
        super().__init__(coords, known_values, point_types)
    
    def set_interior_dataset(self):
        return

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

        self.init_coords = coords
        self.init_known_values = known_values
        self.init_point_types = point_types
            
        # Dataset stores data on CPU, GPU transfer happens in DataLoader
        self.init_dataset = pointCloudDataset(coords, known_values, point_types)
        self.branch_dataset = []
        self.ground_truth = torch.tensor(ground_truth, dtype=torch.float32, device=device)

    def add_branch_dataset(self, adaptive_method_list): # create new branches according to no. of adaptive methods
        for method in adaptive_method_list:
            branch_dataset = pointCloudDataset(
                self.init_coords, 
                self.init_known_values, 
                self.init_point_types
            )
            self.branch_dataset.append(branch_dataset)
    
if __name__ == '__main__':
    from .plaque_flow import preprocessing
    point_cloud_path = 'data/point_cloud.xlsx'
    cfd_results_path = 'data/cfd_results.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    point_cloud_dict, point_cloud, cfd_results = preprocessing(point_cloud_path, cfd_results_path)