import subprocess
import itertools
from datetime import datetime

# Define parameter grids
param_grid = {
    'nVertices': [i for i in range(2, 7)],
    'nTracksMax': [i for i in range(2, 10)],
    'numTests': 1,
}

# Fixed parameters
fixed_params = {
    'inputdir': '../datasets',
    'outputdir': '../datasets/trackgen_v2',
}

# Generate all combinations of parameters
param_names = [k for k in param_grid.keys() if isinstance(param_grid[k], list)]
param_values = [v for v in param_grid.values() if isinstance(v, list)]
param_combinations = list(itertools.product(*param_values))

# Start grid search
start_time = datetime.now()

successful_runs = 0
failed_runs = 0
best_test_loss = float('inf')
best_params = None

for i, params in enumerate(param_combinations, 1):
    # Create dictionary of current parameters
    current_params = dict(zip(param_names, params))
    
    # Combine with fixed parameters
    current_params.update(fixed_params)
    
    # Calculate progress percentage
    progress = (i - 1) / len(param_combinations) * 100
    
    # Construct command
    cmd = ['python', 'track_generator_clean.py']
    for key, value in current_params.items():
        cmd.extend([f'--{key}', str(value)])
    
    # Run the command
    try:
        run_start_time = datetime.now()
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        run_end_time = datetime.now()
        run_duration = run_end_time - run_start_time
        
    except subprocess.CalledProcessError as e:
        failed_runs += 1
        continue

end_time = datetime.now()
total_duration = end_time - start_time