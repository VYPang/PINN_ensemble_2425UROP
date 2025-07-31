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

def train(savePath, device, loss_fn, conf, trainLoader, model, valLoader=None, saveModel=True):
    epochs = conf.epochs
    for epoch in range(epochs):
        train_tqdm = tqdm(trainLoader, total=len(trainLoader))
        for batch_idx, batch in enumerate(train_tqdm):
            train_tqdm.set_description(f'Epoch {epoch+1}/{epochs}')
            model.train()
            optimizer.zero_grad()

            coords, known, point_type = batch
            coords.requires_grad_(True)
            pred = model(coords)

            total_residual, residual_message_dict = model.residual(coords, pred) # residual loss
            total_loss, loss_message_dict = loss_fn.compute(pred, known) # known loss
            loss = total_loss + total_residual
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0: # print loss message
                message = ''
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

def votedSemiSupervisedLearning(conf, model, optimizer, loss_fn, point_cloud_collection, savePath):
    print('Initial training')
    init_dataset = point_cloud_collection.init_dataset
    batch_size = conf.batch_size if conf.batch_size != 'all' else len(init_dataset.coords)
    init_loader = DataLoader(init_dataset, batch_size=batch_size, shuffle=True)
    init_model = train(savePath, device, loss_fn, conf, init_loader, model, saveModel=False)

if __name__ == '__main__':
    flow_prop = OmegaConf.load('data/plaque_flow_1/flow.yaml')
    re = flow_prop.density * flow_prop.inlet_avg_velocity * flow_prop.char_length / flow_prop.viscosity
    point_cloud_path = 'data/plaque_flow_1/point_cloud.xlsx'
    cfd_results_path = 'data/plaque_flow_1/cfd_results.csv'
    config_path = 'config.yaml'
    conf = OmegaConf.load(config_path)

    from model.flow_model import flow_model
    from data.plaque_flow import preprocessing, flow_loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    point_cloud_dict, point_cloud, cfd_results = preprocessing(point_cloud_path, cfd_results_path, flow_prop.char_length)
    point_cloud_collection = pointCloudCollection(point_cloud, device)

    # create checkpoint folder
    current_time = datetime.datetime.now()
    file_name = current_time.strftime("%Y-%m-%d_%H%M%S")
    savePath = f'ckpt/{file_name}'
    os.makedirs(savePath)

    model = flow_model(reynolds_number=re).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    loss_fn = flow_loss(loss_fn=nn.MSELoss())
    votedSemiSupervisedLearning(conf, model, optimizer, loss_fn, point_cloud_collection, savePath)