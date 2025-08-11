import numpy as np
from omegaconf import OmegaConf

class evolutionary_sampling:
    def __init__(self):
        self.conf =  OmegaConf.load('util/evo_sampling_conf.yaml')

    def print(self):
        print("Adaptive sampling method: evolutionary_sampling.")

    def sample(self, residuals, collocation_dataset):
        """
        Perform one step of Evolutionary Sampling (Evo) as per Algorithm 1.
        
        Args:
        - residuals: 1D NumPy array of residuals |R(x_r)| for each point in current_points.
        - current_points: NxD NumPy array of current collocation points (population P_i).
        
        Returns:
        - collocation_dataset: Updated collocation dataset after sampling.
        """
        N_r = len(collocation_dataset.interior_dataset) # current interior points
        current_points = collocation_dataset.interior_dataset.coords.numpy()  # NxD array of current collocation points
        _, current_idx = collocation_dataset.get_interior_dataset()
        known_values = collocation_dataset.interior_dataset.known_values.numpy()  # Known values for current points
        is_pseudo_label = collocation_dataset.interior_dataset.is_pseudo_label.numpy()  # Boolean array for pseudo-labels
        
        # Step 3: Compute fitness F(x_r) = |R(x_r)| for each x_r in P_i
        residuals = np.abs(residuals)
        
        # Step 4: Compute threshold tau_i = mean F
        tau_i = np.mean(residuals)
        
        # Step 5: Select retained population P_r^i where F > tau_i
        retain_mask = residuals > tau_i
        retained_points = current_points[retain_mask]
        retained_idx = current_idx[retain_mask]
        retained_is_pseudo_label = is_pseudo_label[retain_mask]
        retained_known_value = known_values[retain_mask]
        
        # Minimum retain proportion
        min_retain_proportion = self.conf.min_retain_proportion
        if min_retain_proportion is not None:
            min_retain_count = int(min_retain_proportion * N_r)
            if len(retained_points) < min_retain_count:
                # If not enough points retained, keep at least min_retain_count
                sorted_idx = np.argsort(-residuals[retain_mask])[:min_retain_count]
                retained_points = retained_points[sorted_idx]
                retained_idx = retained_idx[sorted_idx]
                retained_is_pseudo_label = retained_is_pseudo_label[sorted_idx]
                retained_known_value = retained_known_value[sorted_idx]

        # remove removed_points from collocation_dataset
        removed_idx = np.setdiff1d(current_idx, retained_idx)
        collocation_dataset.remove_interior_dataset(removed_idx)

        # Note that add_random_interior_collocation will reset the indexes, it should be called after removing points
        # Generate re-sampled population P_s^i ~ U(Ω), size num_new
        num_new = N_r - len(retained_points) # Compute number of new points to sample
        if num_new > 0:
            collocation_dataset.add_random_interior_collocation(num_new)

        # set replay buffer
        removed_points = current_points[~retain_mask]
        removed_is_pseudo_label = is_pseudo_label[~retain_mask]
        removed_known_value = known_values[~retain_mask]
        collocation_dataset.update_replay_buffer(removed_points, removed_is_pseudo_label, removed_known_value)

        return collocation_dataset