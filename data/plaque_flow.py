import pandas as pd
import numpy as np
import torch.nn as nn
import torch

def preprocessing(point_cloud_path, cfd_results_path, char_length):
    point_cloud_dict = pd.read_excel(point_cloud_path, sheet_name=None)
    cfd_results = pd.read_csv(cfd_results_path)

    # preprocess cfd results
    cfd_results = cfd_results.rename(columns={'X [ m ]': 'x', ' Y [ m ]': 'y', ' Pressure [ Pa ]': 'p', ' Velocity [ m s^-1 ]': 'velocity [m/s]', ' Velocity u [ m s^-1 ]': 'u', ' Velocity v [ m s^-1 ]': 'v'})
    group = cfd_results.groupby(by=[' Z [ m ]'])
    z, group = next(iter(group))
    cfd_results = group.reset_index(drop=True)
    cfd_results.drop(columns=[' Z [ m ]'], inplace=True)
    cfd_results = cfd_results.iloc[:, :-6]

    # preprocess point_cloud_dict
    # poiseuille flow with inlet max velocity = 1 m/s
    inlet = point_cloud_dict['inlet_line_points']
    min_x = inlet['x'].min()
    max_x = inlet['x'].max()
    inlet['u'] = 0
    inlet['v'] = 4*(inlet['x'] - min_x)*(max_x - inlet['x'])/(max_x - min_x)**2
    inlet['p'] = np.nan
    inlet['point_type'] = 'inlet'
    inlet['point_label'] = 'inlet'
    point_cloud_dict['inlet_line_points'] = inlet
    # outlet pressure = 0
    outlet = point_cloud_dict['outlet_line_points']
    outlet['u'] = np.nan
    outlet['v'] = np.nan
    outlet['p'] = 0
    outlet['point_type'] = 'outlet'
    outlet['point_label'] = 'outlet'
    point_cloud_dict['outlet_line_points'] = outlet
    # bd1 and bd2 are the upper and lower boundaries with u = v = 0
    bd1 = point_cloud_dict['bd1']
    bd1['u'] = 0
    bd1['v'] = 0
    bd1['p'] = np.nan
    bd1['point_label'] = 'bd1'
    bd1['point_type'] = 'boundary'
    point_cloud_dict['bd1'] = bd1
    bd2 = point_cloud_dict['bd2']
    bd2['u'] = 0
    bd2['v'] = 0
    bd2['p'] = np.nan
    bd2['point_label'] = 'bd2'
    bd2['point_type'] = 'boundary'
    point_cloud_dict['bd2'] = bd2
    # interior points have no boundary conditions
    encrypted_points = point_cloud_dict['encrypted_points']
    encrypted_points['u'] = np.nan
    encrypted_points['v'] = np.nan
    encrypted_points['p'] = np.nan
    encrypted_points['type'] = 'interior'
    encrypted_points['point_type'] = 'interior'
    point_cloud_dict['encrypted_points'] = encrypted_points
    # combine all points
    point_cloud = pd.concat(point_cloud_dict.values(), ignore_index=True)
    point_cloud = [point_cloud[['x', 'y']] / char_length, # coordinates normalized by characteristic length 
                   point_cloud['point_type'], 
                   point_cloud[['u', 'v', 'p']], # known values
                   cfd_results[['u', 'v', 'p']]] # ground truth values


    return point_cloud_dict, point_cloud, cfd_results

class flow_loss:
    def __init__(self, loss_fn=nn.MSELoss()):
        self.loss_fn = loss_fn

    def compute(self, pred, known):
        # u
        tensor = torch.stack([pred[:, 0], known[:, 0]], dim=1)
        mask = ~torch.isnan(tensor[:, 1])
        filtered_tensor = tensor[mask]
        u_loss = self.loss_fn(filtered_tensor[:, 0], filtered_tensor[:, 1])
        # v
        tensor = torch.stack([pred[:, 1], known[:, 1]], dim=1)
        mask = ~torch.isnan(tensor[:, 1])
        filtered_tensor = tensor[mask]
        v_loss = self.loss_fn(filtered_tensor[:, 0], filtered_tensor[:, 1])
        # p
        tensor = torch.stack([pred[:, 2], known[:, 2]], dim=1)
        mask = ~torch.isnan(tensor[:, 1])
        filtered_tensor = tensor[mask]
        p_loss = self.loss_fn(filtered_tensor[:, 0], filtered_tensor[:, 1])
        # total loss
        loss = u_loss + v_loss + p_loss
        message_dict = {
            'u_loss': u_loss.item(),
            'v_loss': v_loss.item(),
            'p_loss': p_loss.item(),
        }
        return loss, message_dict