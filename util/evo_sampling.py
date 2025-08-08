import numpy as np
from omegaconf import OmegaConf

class evolutionary_sampling:
    def __init__(self):
        self.conf =  OmegaConf.load('evo_sampling_conf.yaml')

    def __print__(self):
        print("Adaptive sampling method: evolutionary_sampling.")

    def sample(self, residuals, collocation_dataset):
        """
        Perform one step of Evolutionary Sampling (Evo) as per Algorithm 1.
        
        Args:
        - residuals: 1D NumPy array of residuals |R(x_r)| for each point in current_points.
        - current_points: NxD NumPy array of current collocation points (population P_i).
        
        Returns:
        - new_points: Updated population P_{i+1} of size N_r (identical to original number of points).
        """
        N_r = len(collocation_dataset.interior_dataset) # current interior points
        current_points = collocation_dataset.interior_dataset.coords
        
        # Step 3: Compute fitness F(x_r) = |R(x_r)| for each x_r in P_i
        residuals = np.abs(residuals)
        
        # Step 4: Compute threshold tau_i = mean F
        tau_i = np.mean(residuals)
        
        # Step 5: Select retained population P_r^i where F > tau_i
        retain_mask = residuals > tau_i
        retained_points = current_points[retain_mask]
        
        # Minimum retain proportion
        min_retain_proportion = self.conf.min_retain_proportion
        if min_retain_proportion is not None:
            min_retain_count = int(min_retain_proportion * N_r)
            if len(retained_points) < min_retain_count:
                # If not enough points retained, keep at least min_retain_count
                sorted_idx = np.argsort(-residuals[retain_mask])[:min_retain_count]
                retained_points = retained_points[sorted_idx]
        
        # Compute number of new points to sample
        num_new = N_r - len(retained_points)
        
        # Step 6: Generate re-sampled population P_s^i ~ U(Ω), size num_new
        if num_new > 0:
            collocation_dataset.add_random_interior_collocation(num_new)
        else:
            new_points = np.empty((0, current_points.shape[1]))
        
        # Step 7: Merge P_{i+1} = P_r^i ∪ P_s^i
        updated_points = np.vstack([retained_points, new_points])
        removed_points = current_points[~retain_mask]
        return updated_points, removed_points