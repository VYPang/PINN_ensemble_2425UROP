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
    def __init__(self, point_cloud, device, conf):
        self.point_cloud = point_cloud
        self.device = device
        self.conf = conf
        self.data = self.load_data()
    
    def __len__(self):
        return len(self.data)
    

    
if __name__ == '__main__':
    from .plaque_flow import preprocessing
    point_cloud_path = 'data/point_cloud.xlsx'
    cfd_results_path = 'data/cfd_results.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    point_cloud_dict, point_cloud, cfd_results = preprocessing(point_cloud_path, cfd_results_path)
    pass