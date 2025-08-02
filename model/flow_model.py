import torch.nn as nn
from torch.autograd import grad
import torch

class MLP(nn.Module):
    def __init__(self, input_size=2, output_size=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 160)
        self.fc3 = nn.Linear(160, 160)
        self.fc4 = nn.Linear(160, 160)
        self.fc5 = nn.Linear(160, 128)
        self.fc6 = nn.Linear(128, output_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x
    
    def residual(self, inn_var, out_var): # should be defined in child class
        return 0

class flow_model(MLP):
    def __init__(self, reynolds_number, input_size=2, output_size=3):
        super(flow_model, self).__init__(input_size, output_size)
        self.re = reynolds_number
        
    def residual(self, inn_var, out_var):
        u, v, p = out_var[:, 0], out_var[:, 1], out_var[:, 2]

        # first order derivatives
        duda = grad(u, inn_var, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        dudx, dudy = duda[:, 0], duda[:, 1]
        dvda = grad(v, inn_var, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        dvdx, dvdy = dvda[:, 0], dvda[:, 1]
        dpda = grad(p, inn_var, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        dpdx, dpdy = dpda[:, 0], dpda[:, 1]

        # second order derivatives
        d2uda2 = grad(dudx, inn_var, grad_outputs=torch.ones_like(dudx), create_graph=True, retain_graph=True)[0]
        d2udx2, d2udy2 = d2uda2[:, 0], d2uda2[:, 1]
        d2vda2 = grad(dvdx, inn_var, grad_outputs=torch.ones_like(dvdx), create_graph=True, retain_graph=True)[0]
        d2vdx2, d2vdy2 = d2vda2[:, 0], d2vda2[:, 1]

        mass_residual = dudx + dvdy # conservation of mass
        momentum_residual_x = u*dudx + v*dudy + dpdx - (1/self.re)*(d2udx2 + d2udy2)
        momentum_residual_y = u*dvdx + v*dvdy + dpdy - (1/self.re)*(d2vdx2 + d2vdy2)
        
        mass_residual = torch.mean(mass_residual**2)
        momentum_residual_x = torch.mean(momentum_residual_x**2)
        momentum_residual_y = torch.mean(momentum_residual_y**2)

        total_residual = mass_residual + momentum_residual_x + momentum_residual_y
        message_dict = {
            'mass_residual': mass_residual.item(),
            'momentum_residual_x': momentum_residual_x.item(),
            'momentum_residual_y': momentum_residual_y.item()
        }
        return total_residual, message_dict

if __name__ == '__main__':
    model = flow_model(reynolds_number=1000)
    # sample batch of data
    coord = torch.randn(100, 2)
    coord.requires_grad_(True)
    y = torch.randn(100, 3) # u, v, p
    y.requires_grad_(True)
    # forward pass
    output = model(coord)
    # calculate residual
    residual = model.residual(coord, output)