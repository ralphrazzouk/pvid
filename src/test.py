import os
import torch
import logging
import argparse
import numpy as np
from torch_geometric.loader import DataLoader

from model import TrackDataset, VertexGNN
from data import get_file_paths, split_data
from eval import evaluate, plot_predictions

def setup_logging(save_dir):
    """Setup logging configuration."""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'testing.log')
    
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_model_config(model_dir):
    """Extract model configuration from directory name."""
    dir_name = os.path.basename(os.path.normpath(model_dir))
    config = {}
    
    # Parse directory name for parameters
    parts = dir_name.split('_')
    for part in parts:
        if part.startswith('hd'):
            config['hidden_dim'] = int(part[2:])
        elif part.startswith('nl'):
            config['num_layers'] = int(part[2:])
        elif part.startswith('e'):
            config['epochs'] = int(part[1:])
        elif part.startswith('bs'):
            config['batch_size'] = int(part[2:])
        elif part.startswith('lr'):
            config['learning_rate'] = str(part[2:])
        elif part.startswith('wd'):
            config['weight_decay'] = str(part[2:])
        elif part.startswith('d'):
            config['dropout'] = float(part[1:])
    
    return config

def main(args):
    output_dir = os.path.join(args.output_dir, args.data_set)

    # Delete existing log and prediction files if they exist
    testing_log = os.path.join(output_dir, 'testing.log')
    if os.path.exists(testing_log):
        os.remove(testing_log)
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    logging = setup_logging(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model checkpoint
    checkpoint_path = os.path.join(args.model_dir, args.model_name + '.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from directory name
    config_dir = os.path.join(args.model_dir, args.model_name)
    model_config = get_model_config(config_dir)
    logging.info(f"Loaded model: {args.model_name}")
    logging.info(f"Loaded model configuration: {model_config}")
    
    # Initialize model with correct configuration
    model = VertexGNN(
        hidden_dim=model_config.get('hidden_dim', 256),  # Default to 256 if not found
        num_layers=model_config.get('num_layers', 6),    # Default to 6 if not found
        dropout=model_config.get('dropout', 0.1)         # Default to 0.1 if not found
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['best_val_loss']:.4f}")
    
    # Prepare test dataset
    file_paths = get_file_paths(args.data_dir)
    test_dataset = TrackDataset(file_paths)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Evaluate model using imported evaluate function
    test_loss, test_mae = evaluate(model, test_loader, device)
    
    # Collect predictions for plotting
    true_vertices = []
    pred_vertices = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            true_vertices.extend(batch.y.cpu().numpy())
            pred_vertices.extend(out.cpu().numpy())
    
    true_vertices = np.array(true_vertices).flatten()
    pred_vertices = np.array(pred_vertices).flatten()
    
    # Calculate additional metrics
    rmse = np.sqrt(np.mean((true_vertices - pred_vertices) ** 2))
    accuracy = np.mean(np.abs(true_vertices - pred_vertices) < 0.01)
    
    # Log results
    logging.info(f"Test Results:")
    logging.info(f"Test Loss: {test_loss:.6f}")
    logging.info(f"Mean Absolute Error: {test_mae:.6f}")
    logging.info(f"Root Mean Square Error: {rmse:.6f}")
    logging.info(f"Accuracy within 1%: {accuracy:.4f}")
    
    # Generate plots using imported plot_predictions function
    plot_predictions(true_vertices, pred_vertices, 999, output_dir, filename="scatter", train=False, size=20)
    
    # Save detailed results
    results = {
        'test_loss': test_loss,
        'mae': test_mae,
        'rmse': rmse,
        'accuracy': accuracy,
        'model_config': model_config
    }
    
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained vertex finding model')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained model')
    parser.add_argument('--model_name', type=str, required=True, help='Filename of the model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing test data')
    parser.add_argument('--data_set', type=str, required=True, help='Dataset number')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, default='./test', help='Directory to save results')
    args = parser.parse_args()
    
    main(args)