import torch
import torch.nn as nn
from tqdm import tqdm
from data.dataLoading import pointCloudCollection
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime
from omegaconf import OmegaConf
import numpy as np
import os

def train(savePath, device, loss_fn, conf, trainLoader, model, optimizer, valLoader=None, saveModel=True):
    epochs = conf.epochs
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    for epoch in range(epochs):
        train_tqdm = tqdm(trainLoader, total=len(trainLoader))
        for batch_idx, batch in enumerate(train_tqdm):
            train_tqdm.set_description(f'Epoch {epoch+1}/{epochs}')
            model.train()
            optimizer.zero_grad()

            coords, known, point_type = batch
            coords = coords.to(device, non_blocking=True)
            known = known.to(device, non_blocking=True)
            coords.requires_grad_(True)
            
            # Use mixed precision for faster training
            with torch.cuda.amp.autocast():
                pred = model(coords)
                total_residual, residual_message_dict = model.residual(coords, pred)
                total_loss, loss_message_dict = loss_fn.compute(pred, known)
                loss = total_loss + total_residual
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch % 50 == 0 and batch_idx == 0:  # print loss message once per epoch
                message = f'Epoch {epoch}: '
                if residual_message_dict:
                    for key, value in residual_message_dict.items():
                        message += f'{key}: {value:.4f} '
                if loss_message_dict:
                    for key, value in loss_message_dict.items():
                        message += f'{key}: {value:.4f} '
                if len(message) > 0:
                    print(message)
                    
    if saveModel:
        torch.save(model.state_dict(), savePath + f'/final.pt')
    else:
        return model

def init_training(conf, model, optimizer, loss_fn, point_cloud_collection, savePath, device):
    print('Initial training')
    init_dataset = point_cloud_collection.init_dataset
    batch_size = conf.batch_size if conf.batch_size != 'all' else len(init_dataset)
    
    # Get configuration values with defaults
    num_workers = getattr(conf, 'num_workers', 0)
    pin_memory = getattr(conf, 'pin_memory', False)
    
    # Configure DataLoader based on num_workers
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    
    # Only add multiprocessing-specific options if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = getattr(conf, 'prefetch_factor', 2)
        dataloader_kwargs['persistent_workers'] = True
    
    init_loader = DataLoader(init_dataset, **dataloader_kwargs)
    
    init_model = train(savePath, device, loss_fn, conf, init_loader, model, optimizer, saveModel=False)
    return init_model

def infer(model, infer_loader, device):
    model.eval()
    infer_tqdm = tqdm(infer_loader, total=len(infer_loader))
    all_preds = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(infer_tqdm):
            coords, _, _ = batch
            coords = coords.to(device, non_blocking=True)
            coords.requires_grad_(True)
            
            pred = model(coords)
            all_preds.append(pred.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds

def votedSemiSupervisedLearning(conf, model, optimizer, loss_fn, point_cloud_collection, savePath, device, adaptive_method_list):
    # Initial training: provide pseudo-labels & initial model
    init_model = init_training(conf, model, optimizer, loss_fn, point_cloud_collection, savePath, device)

    # Pseudo-label of interior points
    init_interior_dataset, init_interior_idx = point_cloud_collection.init_dataset.get_interior_dataset()
    init_infer_loader = DataLoader(init_interior_dataset)
    pseudo_labels = infer(init_model, init_infer_loader, device)
    point_cloud_collection.init_dataset.save_pseudo_labels(pseudo_labels, init_interior_idx)

    # Adaptive sampling loop
    point_cloud_collection.add_branch_dataset(adaptive_method_list)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ######### USER INPUT #########
    # Load point cloud & loss function (point_cloud & loss_fn)
    flow_prop = OmegaConf.load('data/plaque_flow_1/flow.yaml')
    re = flow_prop.density * flow_prop.inlet_avg_velocity * flow_prop.char_length / flow_prop.viscosity
    point_cloud_path = 'data/plaque_flow_1/point_cloud.xlsx'
    cfd_results_path = 'data/plaque_flow_1/cfd_results.csv'
    from data.plaque_flow import preprocessing, flow_loss
    point_cloud_dict, point_cloud, cfd_results = preprocessing(point_cloud_path, cfd_results_path, flow_prop.char_length)
    loss_fn = flow_loss(loss_fn=nn.MSELoss())

    # Select model (model)
    from model.flow_model import flow_model
    model = flow_model(reynolds_number=re)

    # Read configuration (conf)
    config_path = 'config.yaml'
    conf = OmegaConf.load(config_path)

    # Define adaptive sampling methods (adaptive_method_list)
    adaptive_method_list = ['random', 'uncertainty', 'gradient', 'residual']
    ######### USER INPUT END #########

    point_cloud_collection = pointCloudCollection(point_cloud, device)

    # create checkpoint folder
    current_time = datetime.datetime.now()
    file_name = current_time.strftime("%Y-%m-%d_%H%M%S")
    savePath = f'ckpt/{file_name}'
    os.makedirs(savePath)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True  # Optimize CUDNN for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matrix operations
    torch.backends.cudnn.allow_tf32 = True
    
    votedSemiSupervisedLearning(conf, model, optimizer, loss_fn, point_cloud_collection, savePath, device, adaptive_method_list)