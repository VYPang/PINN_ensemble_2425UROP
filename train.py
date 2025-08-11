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
import copy

def train(savePath, device, loss_fn, conf, dataset, model, optimizer, valLoader=None, saveModel=True, adaptive_sampling_method=None):
    epochs = conf.epochs
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    batch_size = conf.batch_size if conf.batch_size != 'all' else len(dataset)

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
    
    trainLoader = DataLoader(dataset, **dataloader_kwargs)
    for epoch in range(epochs):
        train_tqdm = tqdm(trainLoader, total=len(trainLoader))
        if adaptive_sampling_method and (epoch + 1) % conf.adaptive_sampling.sampling_interval == 0:
            residual_record = []
            idx_record = []
        else:
            residual_record = None
            idx_record = None

        for batch_idx, batch in enumerate(train_tqdm):
            train_tqdm.set_description(f'Epoch {epoch+1}/{epochs}')
            model.train()
            optimizer.zero_grad()

            coords, known, point_type, idx = batch
            coords = coords.to(device, non_blocking=True)
            known = known.to(device, non_blocking=True)
            coords.requires_grad_(True)
            
            # Use mixed precision for faster training
            with torch.cuda.amp.autocast():
                pred = model(coords)
                total_residual, collocation_residual, residual_message_dict = model.residual(coords, pred)
                total_loss, loss_message_dict = loss_fn.compute(pred, known)
                loss = total_loss + total_residual
            if residual_record is not None:
                residual_record.append(collocation_residual.detach().cpu().numpy())
                idx_record.append(idx.detach().cpu().numpy())

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
        
        # Adaptive sampling step
        if residual_record is not None:
            adaptive_sampling_method.print()
            # concatenate residuals and indices
            ordered_residuals = np.concatenate(residual_record, axis=0)
            ordered_idx = np.concatenate(idx_record, axis=0)
            # rearrange residuals based on indices
            ordered_residuals = ordered_residuals[np.argsort(ordered_idx)]
            ordered_idx = np.sort(ordered_idx)
            # preserve interior points
            _, interior_idx = dataset.get_interior_dataset()
            ordered_residuals = ordered_residuals[interior_idx]
            ordered_idx = ordered_idx[interior_idx]

            dataset = adaptive_sampling_method.sample(residuals=ordered_residuals, collocation_dataset=dataset)
            dataset.add_random_from_replay_buffer() # add collocation from replay buffer
            trainLoader = DataLoader(dataset, **dataloader_kwargs) # set trainLoader with updated dataset

    if saveModel:
        torch.save(model.state_dict(), savePath + f'/final.pt')
    else:
        return model

def infer(model, infer_loader, device):
    model.eval()
    infer_tqdm = tqdm(infer_loader, total=len(infer_loader))
    all_preds = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(infer_tqdm):
            coords, _, _, _ = batch
            coords = coords.to(device, non_blocking=True)
            coords.requires_grad_(True)
            
            pred = model(coords)
            all_preds.append(pred.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds

def votedSemiSupervisedLearning(conf, model, optimizer, loss_fn, point_cloud_collection, savePath, device, adaptive_method_list):
    # Create an independent copy of the original model
    og_model = copy.deepcopy(model)
    
    # Initial training: provide pseudo-labels & initial model
    print('Initial training')
    init_dataset = point_cloud_collection.init_dataset
    init_model = train(savePath, device, loss_fn, conf, init_dataset, model, optimizer, saveModel=False)

    # Pseudo-label of interior points
    if conf.psuedo_labeling.use_pseudo_labels:
        print('Infer & Pseudo-labeling initial interior points')
        init_interior_dataset, init_interior_idx = point_cloud_collection.init_dataset.get_interior_dataset()
        init_infer_loader = DataLoader(init_interior_dataset)
        pseudo_labels = infer(init_model, init_infer_loader, device)
        point_cloud_collection.init_dataset.save_pseudo_labels(pseudo_labels, init_interior_idx)

    # Adaptive sampling loop
    point_cloud_collection.add_branch_dataset(adaptive_method_list)
    point_cloud_collection.add_random_interior_collocation_to_branch_datasets(conf.adaptive_sampling.add_random_interior_collocation)
    point_cloud_collection.add_branch_model(og_model, init_model)
    for i in range(len(adaptive_method_list)):
        adaptive_method_list[i].print()
        branch_dataset = point_cloud_collection.branch_dataset[i]
        branch_model = point_cloud_collection.branch_model[i]
        branch_optimizer = optim.Adam(branch_model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        branch_model = train(savePath, device, loss_fn, conf, branch_dataset, branch_model, branch_optimizer, saveModel=False, adaptive_sampling_method=adaptive_method_list[i])

def semiSupervisedLearning(conf, model, optimizer, loss_fn, point_cloud_collection, savePath, device, adaptive_method):
    # Create an independent copy of the original model
    og_model = copy.deepcopy(model)
    
    # Initial training: provide pseudo-labels & initial model
    print('Initial training')
    init_dataset = point_cloud_collection.init_dataset
    init_model = train(savePath, device, loss_fn, conf, init_dataset, model, optimizer, saveModel=False)

    # Pseudo-label of interior points
    if conf.psuedo_labeling.use_pseudo_labels:
        print('Infer & Pseudo-labeling initial interior points')
        init_interior_dataset, init_interior_idx = point_cloud_collection.init_dataset.get_interior_dataset()
        init_infer_loader = DataLoader(init_interior_dataset)
        pseudo_labels = infer(init_model, init_infer_loader, device)
        point_cloud_collection.init_dataset.save_pseudo_labels(pseudo_labels, init_interior_idx)

    # Semi-supervised learning loop
    adaptive_method = adaptive_method if isinstance(adaptive_method, list) else [adaptive_method]
    point_cloud_collection.add_branch_dataset(adaptive_method)
    point_cloud_collection.add_random_interior_collocation_to_branch_datasets(conf.adaptive_sampling.add_random_interior_collocation)
    point_cloud_collection.add_branch_model(og_model, init_model)

    # Adaptive sampling loop
    adaptive_method = adaptive_method[0]  # Use the first method for semi-supervised learning
    dataset = point_cloud_collection.branch_dataset[0]
    model = point_cloud_collection.branch_model[0]
    optimizer = optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    train(savePath, device, loss_fn, conf, dataset, model, optimizer, saveModel=True, adaptive_sampling_method=adaptive_method)

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
    from util.evo_sampling import evolutionary_sampling
    adaptive_method_list = [evolutionary_sampling(), 'uncertainty', 'gradient', 'residual']
    ######### USER INPUT END #########

    point_cloud_collection = pointCloudCollection(point_cloud, conf, device)

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
    
    semiSupervisedLearning(conf, model, optimizer, loss_fn, point_cloud_collection, savePath, device, evolutionary_sampling())
    # votedSemiSupervisedLearning(conf, model, optimizer, loss_fn, point_cloud_collection, savePath, device, adaptive_method_list)