import subprocess
import itertools
from datetime import datetime
import logging
import os

# Set up logging
log_dir = 'grid_search_logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('grid_search')
logger.setLevel(logging.INFO)

# Create file handler
log_file = os.path.join(log_dir, f'grid_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Define parameter grids
param_grid = {
    'hidden_dim': [32, 64, 128, 256, 512],
    'num_layers': [4, 6, 8, 12],
    'learning_rate': [1e-4, 5e-5, 1e-5, 5e-6],
    'batch_size': [32, 64, 128],
    # 'activation': ['relu', 'gelu']
}

# Fixed parameters
fixed_params = {
    'dataset': 'trackgen_v1',
    'data_dir': '../datasets/trackgen_v1',
    'epochs': 50,
    'dropout': 0.05,
}

# Generate all combinations of parameters
param_names = list(param_grid.keys())
param_values = list(param_grid.values())
param_combinations = list(itertools.product(*param_values))

# Start grid search
start_time = datetime.now()
logger.info("=" * 80)
logger.info(f"Starting grid search at {start_time}")
logger.info("Parameter grid:")
for param, values in param_grid.items():
    logger.info(f"{param}: {values}")
logger.info("\nFixed parameters:")
for param, value in fixed_params.items():
    logger.info(f"{param}: {value}")
logger.info(f"\nTotal number of combinations: {len(param_combinations)}")
logger.info("=" * 80)

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
    cmd = ['python', 'train.py']
    for key, value in current_params.items():
        cmd.extend([f'--{key}', str(value)])
    
    # Log current run info with separator and progress
    logger.info("-" * 80)
    logger.info(f"Run {i}/{len(param_combinations)} ({progress:.1f}% complete)")
    logger.info("Current best test loss: {:.6f}".format(best_test_loss))
    if best_params:
        logger.info("Best parameters so far:")
        for key, value in best_params.items():
            if key in param_grid:  # Only log grid search parameters
                logger.info(f"{key}: {value}")
    logger.info("\nCurrent parameters:")
    for key, value in current_params.items():
        if key in param_grid:  # Only log grid search parameters
            logger.info(f"{key}: {value}")
    logger.info("\nCommand: " + " ".join(cmd))
    
    # Run the command
    try:
        run_start_time = datetime.now()
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        run_end_time = datetime.now()
        run_duration = run_end_time - run_start_time
        
        # Log success and output
        logger.info(f"Run completed successfully in {run_duration}")
        
        # Try to extract final metrics from the output
        output_lines = process.stdout.split('\n')
        test_loss = None
        for line in reversed(output_lines):
            if "Test Loss:" in line:
                try:
                    test_loss = float(line.split(": ")[1])
                    logger.info(line.strip())
                except:
                    continue
            elif "Test MAE:" in line:
                logger.info(line.strip())
                
        # Update best parameters if this run was better
        if test_loss and test_loss < best_test_loss:
            best_test_loss = test_loss
            best_params = current_params.copy()
            logger.info("\nNew best model found!")
            logger.info(f"Test Loss: {best_test_loss:.6f}")
        
        successful_runs += 1
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in run {i}:")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        failed_runs += 1
        continue

end_time = datetime.now()
total_duration = end_time - start_time

# Log final summary
logger.info("=" * 80)
logger.info("Grid Search Final Summary")
logger.info("=" * 80)
logger.info(f"Completed at: {end_time}")
logger.info(f"Total duration: {total_duration}")
logger.info(f"Total runs: {len(param_combinations)}")
logger.info(f"Successful runs: {successful_runs}")
logger.info(f"Failed runs: {failed_runs}")
logger.info(f"Success rate: {successful_runs/len(param_combinations)*100:.2f}%")

if best_params:
    logger.info("\nBest parameters found:")
    logger.info(f"Test Loss: {best_test_loss:.6f}")
    for key, value in best_params.items():
        if key in param_grid:  # Only log grid search parameters
            logger.info(f"{key}: {value}")