# Adaptive Sampling Methods

This folder contains implementations of various adaptive sampling methods for Physics-Informed Neural Networks (PINNs). These methods help improve the training efficiency by intelligently selecting collocation points based on model performance.

## How to Create a Custom Sampling Method

To implement a new adaptive sampling method, you need to create a class that follows the standardized interface. Here's the template:

### Required Interface

Every sampling method class must implement:

1. **`__init__(self)`** - Constructor to initialize the method
2. **`__print__(self)`** - Method to print information about the sampling method
3. **`sample(self, residuals, collocation_dataset)`** - Main sampling logic

### Template

```python
import numpy as np
from omegaconf import OmegaConf

class your_sampling_method:
    def __init__(self):
        """
        Initialize your sampling method.
        Load configuration files, set parameters, etc.
        """
        # Example: Load configuration if needed
        # self.conf = OmegaConf.load('your_config.yaml')
        pass

    def __print__(self):
        """
        Print information about this sampling method.
        This will be called to identify the method during training.
        """
        print("Adaptive sampling method: your_sampling_method.")

    def sample(self, residuals, collocation_dataset):
        """
        Main sampling logic - this is where you implement your adaptive sampling algorithm.
        
        Args:
            residuals (numpy.ndarray): 1D array of residuals |R(x_r)| for each interior point.
                                     Shape: (N_interior,) where N_interior is number of interior points.
            
            collocation_dataset (pointCloudDataset): Dataset containing all collocation points.
                                                   Use collocation_dataset.interior_dataset to access
                                                   only the interior points and their coordinates.
        
        Returns:
            tuple: (updated_points, removed_points)
                - updated_points (numpy.ndarray): New set of interior points after adaptive sampling.
                                                Shape: (N_new, 2) where N_new can be different from original
                - removed_points (numpy.ndarray): Points that were removed during sampling.
                                                 Shape: (N_removed, 2)
        
        Note:
            - You can access interior coordinates via: collocation_dataset.interior_dataset.coords
            - You can add new random interior points via: collocation_dataset.add_random_interior_collocation(num_points)
            - The collocation_dataset.check_inside_boundary(x, y) method can verify if points are within domain
        """
        
        # Get current interior points
        current_points = collocation_dataset.interior_dataset.coords
        N_current = len(current_points)
        
        # Your sampling logic here
        # Example: Simple random replacement
        retain_mask = np.random.choice([True, False], size=len(residuals), p=[0.7, 0.3])
        retained_points = current_points[retain_mask]
        removed_points = current_points[~retain_mask]
        
        # Generate new points to replace removed ones
        num_new = N_current - len(retained_points)
        if num_new > 0:
            collocation_dataset.add_random_interior_collocation(num_new)
            # Get the newly added points (they are appended to the end)
            all_interior_points = collocation_dataset.interior_dataset.coords
            new_points = all_interior_points[-num_new:]
            updated_points = np.vstack([retained_points, new_points])
        else:
            updated_points = retained_points
        
        return updated_points, removed_points
```

## Available Sampling Methods

### 1. Evolutionary Sampling (`evolutionary_sampling`)

Located in `evo_sampling.py`, this method implements evolutionary-based adaptive sampling:

- **Algorithm**: Retains points with residuals above the mean threshold
- **Configuration**: Uses `evo_sampling_conf.yaml` for parameters
- **Key Features**:
  - Fitness-based selection using residual magnitudes
  - Minimum retention proportion to prevent over-pruning
  - Random resampling for removed points

## Usage in Training

To use a sampling method in your training pipeline:

```python
# Import your sampling method
from util.your_sampling_method import your_sampling_method

# Define adaptive sampling methods list
adaptive_method_list = [your_sampling_method(), 'uncertainty', 'gradient', 'residual']

# Pass to training function
votedSemiSupervisedLearning(conf, model, optimizer, loss_fn, 
                           point_cloud_collection, savePath, device, 
                           adaptive_method_list)
```

## Key Components

### collocation_dataset Methods

The `collocation_dataset` object provides several useful methods:

- `collocation_dataset.interior_dataset.coords` - Get interior point coordinates
- `collocation_dataset.add_random_interior_collocation(num_points)` - Add random points within domain
- `collocation_dataset.check_inside_boundary(x, y)` - Check if point is within domain boundary

### Residuals

The `residuals` array contains the magnitude of physics equation residuals for each interior point. Higher residuals typically indicate areas where the model needs improvement.

## Best Practices

1. **Maintain Point Count**: Many algorithms expect a consistent number of points, so consider maintaining the total count.

2. **Boundary Checking**: Always ensure new points are within the physical domain using `check_inside_boundary()`.

3. **Configuration Management**: Use configuration files (YAML) for hyperparameters to make methods easily tunable.

4. **Error Handling**: Include proper error handling for edge cases (e.g., no points to remove, boundary checking failures).

5. **Efficiency**: For large point sets, consider batch operations and efficient numpy operations.

## Example Configuration File

Create a YAML configuration file for your method:

```yaml
# your_method_config.yaml
min_retain_proportion: 0.3  # Minimum fraction of points to retain
sampling_strategy: "uniform"  # Options: uniform, gaussian, etc.
boundary_buffer: 0.01  # Buffer from boundary for new points
max_iterations: 1000  # Maximum attempts for boundary-constrained sampling
```

## Testing Your Method

Test your sampling method independently:

```python
if __name__ == '__main__':
    # Load test data
    from data.dataLoading import pointCloudCollection
    # ... load your test data ...
    
    # Create synthetic residuals
    residuals = np.random.random(num_interior_points)
    
    # Test your method
    sampler = your_sampling_method()
    updated_points, removed_points = sampler.sample(residuals, collocation_dataset)
    
    print(f"Original points: {len(collocation_dataset.interior_dataset.coords)}")
    print(f"Updated points: {len(updated_points)}")
    print(f"Removed points: {len(removed_points)}")
```
