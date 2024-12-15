import os
import logging
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import warnings
# warnings.filterwarnings('ignore')

from data import get_file_paths, split_data, collect_data_from_files, plot_dataset_distributions, print_statistics
from model import TrackDataset, VertexGNN
from eval import evaluate, plot_training_metrics, save_predictions, plot_predictions



class TrainingConfig:
    def __init__(self):
        self.dataset = 'set1'
        self.hidden_dim = 256
        self.num_layers = 6
        self.dropout = 0.05
        self.epochs = 100
        self.batch_size = 64
        self.learning_rate = 1e-5
        self.weight_decay = 1e-4
        self.early_stopping_patience = 50



def setup_logging(config_dir):
    """Setup logging configuration."""
    try:
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Remove any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        log_file = os.path.join(config_dir, 'training.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Setup handlers
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging setup complete. Saving logs to: {log_file}")
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise


def train_epoch(model: torch.nn.Module, loader: DataLoader,optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)

        # Compute standard and physics loss
        primary_loss = torch.abs(out - batch.y).mean()
        physics_loss = model.compute_physics_loss(out, batch)
        
        # Combined loss with weighting
        full_loss = primary_loss + 0.01 * physics_loss  # Adjust weight as needed
        
        full_loss.backward()
        optimizer.step()
        
        total_loss += full_loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)


def main(args):
    config_dir = os.path.join("train", f"{args.dataset}_hd{args.hidden_dim}_nl{args.num_layers}_e{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}_wd{args.weight_decay}_d{args.dropout}")
    os.makedirs(config_dir, exist_ok=True)
    
    # Delete existing log and prediction files if they exist
    predictions_file = os.path.join(config_dir, 'predictions.txt')
    predictions_summary_file = os.path.join(config_dir, 'prediction_summary.txt')
    training_log_file = os.path.join(config_dir, 'training.log')

    if os.path.exists(predictions_file):
        os.remove(predictions_file)
    if os.path.exists(predictions_summary_file):
        os.remove(predictions_summary_file)
    if os.path.exists(training_log_file):
        os.remove(training_log_file)

    setup_logging(config_dir)

    plots_dir = os.path.join(f"{config_dir}", "plots")
    plots_frames_dir = os.path.join(f"{config_dir}", "plots", "frames")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(plots_frames_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Visualize data
    data = collect_data_from_files(args.data_dir)
    plot_dataset_distributions(data, config_dir)
    print_statistics(data)
    
    # Get and split data
    file_paths = get_file_paths(args.data_dir)
    train_files, val_files, test_files = split_data(file_paths)
    
    logging.info(f"Number of files - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Create datasets and dataloaders
    train_dataset = TrackDataset(train_files)
    val_dataset = TrackDataset(val_files)
    test_dataset = TrackDataset(test_files)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = VertexGNN(hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=args.learning_rate,
    #     epochs=args.epochs,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.1,
    #     div_factor=10,
    #     final_div_factor=1e4
    # )
    
    # Initialize history dictionary
    history = defaultdict(list)
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        val_loss, val_mae = evaluate(model, val_loader, device)
        
        # Collect predictions for plotting
        true_vertices = []
        pred_vertices = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                true_vertices.extend(batch.y.cpu().numpy())
                pred_vertices.extend(out.cpu().numpy())
        
        # Calculate accuracy 
        true_vertices = np.array(true_vertices).flatten()
        pred_vertices = np.array(pred_vertices).flatten()
        accuracy = np.mean(np.abs(true_vertices - pred_vertices) < 0.01)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['accuracy'].append(accuracy)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Log progress
        logging.info(f"Epoch {epoch+1}/{args.epochs}: "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val MAE: {val_mae:.4f}, "
                   f"Accuracy: {accuracy:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Plot metrics and predictions
        plot_training_metrics(history, save_dir=plots_dir) # Pass args instead of config
        save_predictions(np.array(true_vertices), np.array(pred_vertices), epoch, save_dir=config_dir)
        plot_predictions(np.array(true_vertices), np.array(pred_vertices), epoch, save_dir=plots_dir)
        logging.info(f"Plots saved!")

        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss
            }, os.path.join(config_dir, 'best_model.pt'))
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                logging.info("Early stopping triggered")
                break
    
    # Load best model and evaluate on test set
    checkpoint = torch.load(os.path.join(config_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_mae = evaluate(model, test_loader, device)
    
    # Final test set predictions
    true_vertices = []
    pred_vertices = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            true_vertices.extend(batch.y.cpu().numpy())
            pred_vertices.extend(out.cpu().numpy())
    
    # Calculate and log final metrics
    true_vertices = np.array(true_vertices).flatten()
    pred_vertices = np.array(pred_vertices).flatten()
    final_accuracy = np.mean(np.abs(np.array(true_vertices) - np.array(pred_vertices)) < 0.01)
    logging.info(f"Final Test Results:")
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test MAE: {test_mae:.4f}")
    logging.info(f"Test Accuracy: {final_accuracy:.4f}")
    
    # Save final metrics
    with open(os.path.join(config_dir, f'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test MAE: {test_mae:.4f}\n")
        f.write(f"Test Accuracy: {final_accuracy:.4f}\n")

    # Plot final test set predictions
    plot_predictions(np.array(true_vertices), np.array(pred_vertices), epoch=999, save_dir=plots_dir, filename="scatter_test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True, help='Name of dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing data files')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden size of the model')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of graph conv layers')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=300, help='Early stopping patience')
    
    args = parser.parse_args()
    
    main(args)