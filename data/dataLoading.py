from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.path as mplpath
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
import copy

class collocation:
    def __init__(self, coord, point_type, label):
        self.coord = coord # (x, y)
        self.point_type = point_type # interior, or boundary
        self.label = label # e.g. (u, v, p)

# Intake point_cloud should be a 
class pointCloudDataset(Dataset):
    def __init__(self, conf, coords, known_values, point_types, is_pseudo_label=None, dataset_name=None):
        # Store data on CPU for DataLoader compatibility
        self.conf = conf
        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.known_values = torch.tensor(known_values, dtype=torch.float32)
        self.is_pseudo_label = torch.full((len(self.coords),), False, dtype=torch.bool) if is_pseudo_label is None else torch.tensor(is_pseudo_label, dtype=torch.bool)
        self.dataset_name = dataset_name
        
        # Convert point_types to numpy array for proper integer indexing
        if hasattr(point_types, 'values'):
            self.point_types = point_types.values
        elif hasattr(point_types, 'reset_index'):
            self.point_types = point_types.reset_index(drop=True).values
        else:
            self.point_types = np.array(point_types)
            
        self.idex = np.arange(len(self.coords))
        self._create_domain_boundary()
        self.set_interior_dataset()
    
    def add_random_from_replay_buffer(self):
        """
        Warning: USING THIS METHOD RESETS INDEXES.
        """
        if self.conf.adaptive_sampling.replay_buffer.use_replay_buffer:
            num_points = self.conf.adaptive_sampling.replay_buffer.add_num_points
            if hasattr(self, 'replay_buffer') and len(self.replay_buffer) > 0:
                # Randomly sample points from replay buffer
                indices = torch.randperm(len(self.replay_buffer))[:num_points]
                sampled_points = self.replay_buffer[indices]
                sampled_is_pseudo_label = self.replay_buffer_is_pseudo_label[indices]
                sampled_known_value = self.replay_buffer_known_value[indices]

                # Update dataset with sampled points
                self.coords = torch.cat((self.coords, sampled_points), dim=0)
                self.known_values = torch.cat((self.known_values, sampled_known_value), dim=0)
                self.is_pseudo_label = torch.cat((self.is_pseudo_label, sampled_is_pseudo_label), dim=0)
                self.point_types = np.concatenate((self.point_types, np.full(num_points, 'interior')))
                self.idex = np.arange(len(self.coords))
                self.set_interior_dataset()

    def update_replay_buffer(self, removed_points, removed_is_pseudo_label, removed_known_value):
        if removed_points is None: # adaptive sampling method does not remove points
            return
        if self.conf.adaptive_sampling.replay_buffer.use_replay_buffer and len(removed_points) > 0:
            # Ensure removed_points is a torch tensor
            if not isinstance(removed_points, torch.Tensor):
                removed_points = torch.tensor(removed_points, dtype=torch.float32)
            if not isinstance(removed_is_pseudo_label, torch.Tensor):
                removed_is_pseudo_label = torch.tensor(removed_is_pseudo_label, dtype=torch.bool)
            if not isinstance(removed_known_value, torch.Tensor):
                removed_known_value = torch.tensor(removed_known_value, dtype=torch.float32)

            # Add removed points to replay buffer
            if not hasattr(self, 'replay_buffer'):
                self.replay_buffer = removed_points
            else:
                self.replay_buffer = torch.cat((self.replay_buffer, removed_points), dim=0)
            if not hasattr(self, 'replay_buffer_is_pseudo_label'):
                self.replay_buffer_is_pseudo_label = removed_is_pseudo_label
            else:
                self.replay_buffer_is_pseudo_label = torch.cat((self.replay_buffer_is_pseudo_label, removed_is_pseudo_label), dim=0)
            if not hasattr(self, 'replay_buffer_known_value'):
                self.replay_buffer_known_value = removed_known_value
            else:
                self.replay_buffer_known_value = torch.cat((self.replay_buffer_known_value, removed_known_value), dim=0)
            
            # Limit replay buffer size
            if len(self.replay_buffer) > self.conf.adaptive_sampling.replay_buffer.replay_buffer_size:
                # randomly sample from replay buffer to maintain size
                indices = torch.randperm(len(self.replay_buffer))[:self.conf.adaptive_sampling.replay_buffer.replay_buffer_size]
                self.replay_buffer = self.replay_buffer[indices]
                self.replay_buffer_is_pseudo_label = self.replay_buffer_is_pseudo_label[indices]
                self.replay_buffer_known_value = self.replay_buffer_known_value[indices]

    def _create_domain_boundary(self):
        """Create a polygon representing the domain boundary from boundary, inlet, and outlet points."""
        # Get boundary points (including boundary, inlet, outlet)
        boundary_types = ['boundary', 'inlet', 'outlet']
        boundary_mask = np.isin(self.point_types, boundary_types)
        boundary_coords = self.coords[boundary_mask].numpy()
        
        if len(boundary_coords) < 3:
            raise ValueError("Need at least 3 boundary points to define a domain")
        
        try:
            # Method 1: Use alpha shape for irregular boundaries
            from scipy.spatial import distance_matrix
            
            # Calculate pairwise distances
            distances = distance_matrix(boundary_coords, boundary_coords)
            
            # Find ordering by connecting nearest neighbors
            ordered_indices = []
            current_idx = 0  # Start with first point
            remaining_indices = set(range(1, len(boundary_coords)))
            ordered_indices.append(current_idx)
            
            while remaining_indices:
                # Find nearest unvisited point
                current_distances = distances[current_idx]
                nearest_idx = min(remaining_indices, key=lambda i: current_distances[i])
                ordered_indices.append(nearest_idx)
                remaining_indices.remove(nearest_idx)
                current_idx = nearest_idx
            
            self.boundary_polygon = boundary_coords[ordered_indices]
            self.domain_path = mplpath.Path(self.boundary_polygon)
            
        except Exception as e:
            print(f"Warning: Could not create proper boundary ordering: {e}")
            # Fallback: try to use the points as-is if they're already ordered
            self.boundary_polygon = boundary_coords
            self.domain_path = mplpath.Path(self.boundary_polygon)
    
    def visualize_domain_and_points(self):
        """Visualize the domain boundary and points for debugging."""        
        plt.figure(figsize=(10, 8))
        
        # Plot boundary
        boundary_coords = np.vstack([self.boundary_polygon, self.boundary_polygon[0]])
        plt.plot(boundary_coords[:, 0], boundary_coords[:, 1], 'k-', linewidth=2, label='Domain Boundary')
        
        # Plot different point types
        for point_type in np.unique(self.point_types):
            mask = self.point_types == point_type
            coords = self.coords[mask].numpy()
            if point_type == 'interior':
                plt.scatter(coords[:, 0], coords[:, 1], label=point_type, alpha=0.5, s=2)
            else:
                plt.scatter(coords[:, 0], coords[:, 1], label=point_type, alpha=0.7)
        
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('Domain and Collocation Points')
        plt.savefig(f'{self.dataset_name}_domain_and_points.png')

    def add_random_interior_collocation(self, num_points):
        """
        Warning: USING THIS METHOD RESETS INDEXES.
        Add random interior collocation points using rejection sampling with adaptive bounds.
        """
        if not hasattr(self, 'domain_path') or self.domain_path is None:
            raise ValueError("Domain boundary not defined. Cannot add interior points.")
        
        # Use Shapely for more robust point-in-polygon testing
        boundary_polygon_shapely = Polygon(self.boundary_polygon)
        
        # Get tighter bounds using actual polygon bounds
        bounds = boundary_polygon_shapely.bounds  # (minx, miny, maxx, maxy)
        x_min, y_min, x_max, y_max = bounds
        
        new_interior_coords = []
        batch_size = min(1000, num_points * 10)  # Sample in batches for efficiency
        
        while len(new_interior_coords) < num_points:
            # Generate batch of random points
            x_batch = np.random.uniform(x_min, x_max, batch_size)
            y_batch = np.random.uniform(y_min, y_max, batch_size)
            
            # Check all points in batch
            for x, y in zip(x_batch, y_batch):
                if len(new_interior_coords) >= num_points:
                    break
                    
                if self.check_inside_boundary(x, y):
                    new_interior_coords.append([x, y])
        
        # Add the points to the dataset (same as previous method)
        if len(new_interior_coords) > 0:
            new_coords = torch.tensor(new_interior_coords[:num_points], dtype=torch.float32)
            new_values = torch.full((len(new_coords), self.known_values.shape[1]), float('nan'))
            new_point_types = np.array(['interior'] * len(new_coords))
            new_psuedo_label = torch.full((len(new_coords),), False, dtype=torch.bool)
            
            self.coords = torch.cat([self.coords, new_coords], dim=0)
            self.known_values = torch.cat([self.known_values, new_values], dim=0)
            self.point_types = np.concatenate([self.point_types, new_point_types])
            self.is_pseudo_label = torch.cat([self.is_pseudo_label, new_psuedo_label])
            self.idex = np.arange(len(self.coords))
            self.set_interior_dataset()
            
            print(f"Added {len(new_coords)} random interior collocation points")
        
        return len(new_interior_coords)
    
    def set_interior_dataset(self):
        interior_mask = self.point_types == 'interior'
        interior_coords = self.coords[interior_mask]
        interior_known_values = self.known_values[interior_mask]
        interior_point_types = self.point_types[interior_mask]
        interior_pseudo_label = self.is_pseudo_label[interior_mask]
        self.interior_idx = self.idex[interior_mask]
        
        # Convert filtered data to proper format for interiorDataset
        if len(interior_coords) > 0:
            self.interior_dataset = interiorDataset(
                self.conf,
                interior_coords.cpu().numpy(), 
                interior_known_values.cpu().numpy(), 
                interior_point_types,
                is_pseudo_label=interior_pseudo_label.cpu().numpy()
            )
        else:
            # Create empty dataset if no interior points found
            self.interior_dataset = interiorDataset(
                self.conf,
                np.empty((0, 2)), 
                np.empty((0, 3)), 
                np.array([]),
                is_pseudo_label=np.empty((0,), dtype=bool)
            )
    def remove_interior_dataset(self, removed_idx):
        """
        Warning: USING THIS METHOD RESETS INDEXES.
        """
        self.coords = self.coords[~np.isin(self.idex, removed_idx)]
        self.known_values = self.known_values[~np.isin(self.idex, removed_idx)]
        self.point_types = self.point_types[~np.isin(self.idex, removed_idx)]
        self.is_pseudo_label = self.is_pseudo_label[~np.isin(self.idex, removed_idx)]
        self.idex = np.arange(len(self.coords))
        self.set_interior_dataset()  # Update interior dataset after removing points
    
    def get_interior_dataset(self):
        return self.interior_dataset, self.interior_idx
    
    def save_pseudo_labels(self, pseudo_labels, idx):
        # Save pseudo labels to the known_values tensor at the specified indices
        self.known_values[idx] = torch.tensor(pseudo_labels, dtype=torch.float32)
        self.is_pseudo_label[idx] = True
        self.set_interior_dataset()  # Update interior dataset after saving pseudo labels

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return self.coords[idx], self.known_values[idx], self.point_types[idx], self.idex[idx]

    def check_inside_boundary(self, x, y): # Check if a point (x, y) is inside the domain boundary.
        if not hasattr(self, 'domain_path') or self.domain_path is None:
            raise ValueError("Domain boundary not defined. Cannot check point location.")
        
        # Use Shapely for more robust point-in-polygon testing
        boundary_polygon_shapely = Polygon(self.boundary_polygon)
        point_shapely = Point(x, y)
        return boundary_polygon_shapely.contains(point_shapely)

class interiorDataset(pointCloudDataset):
    def __init__(self, conf, coords, known_values, point_types, is_pseudo_label=None):
        super().__init__(conf, coords, known_values, point_types, is_pseudo_label=is_pseudo_label)
    
    def set_interior_dataset(self): # Override to prevent setting interior dataset
        return
    
    def _create_domain_boundary(self): # Override to prevent boundary creation for interior points
        return

class pointCloudCollection:
    def __init__(self, point_cloud, conf, device):
        coords, point_types, known_values, ground_truth = point_cloud
        self.conf = conf
        
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
        self.init_dataset = pointCloudDataset(conf, coords, known_values, point_types, dataset_name='initial')
        self.branch_dataset = []
        self.ground_truth = torch.tensor(ground_truth, dtype=torch.float32, device=device)

    def add_branch_dataset(self, adaptive_method_list): # create new branches according to no. of adaptive methods
        for method in adaptive_method_list:
            branch_dataset = pointCloudDataset(
                self.conf,
                self.init_coords, 
                self.init_known_values, 
                self.init_point_types,
                self.init_dataset.is_pseudo_label,
                dataset_name=f'branch_{method}'
            )
            self.branch_dataset.append(branch_dataset)

    def add_random_interior_collocation_to_branch_datasets(self, num_points):
        for branch in self.branch_dataset:
            branch.add_random_interior_collocation(num_points)
    
    def add_branch_model(self, original_model, initial_model):
        self.branch_model = []
        for method in self.branch_dataset:
            if self.conf.adaptive_sampling.initiate_new_model:
                # Create an independent copy of the original model
                self.branch_model.append(copy.deepcopy(original_model))
            else:
                # Create an independent copy of the initial model
                self.branch_model.append(copy.deepcopy(initial_model))

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from plaque_flow import preprocessing
    point_cloud_path = 'data/plaque_flow_1/point_cloud.xlsx'
    cfd_results_path = 'data/plaque_flow_1/cfd_results.csv'
    flow_prop = OmegaConf.load('data/plaque_flow_1/flow.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    point_cloud_dict, point_cloud, cfd_results = preprocessing(point_cloud_path, cfd_results_path, flow_prop.char_length)
    point_cloud_collection = pointCloudCollection(point_cloud, cfd_results, device)
    
    point_cloud_collection.add_branch_dataset(['PINN', 'PINN_ensemble']) # testing methods