import os
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader


def evaluate(model: torch.nn.Module, 
            loader: DataLoader, 
            device: torch.device) -> Tuple[float, float]:
    """Evaluate model on given loader."""
    model.eval()
    total_loss = 0
    total_abs_error = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = torch.abs(out - batch.y).mean()
            
            total_loss += loss.item() * batch.num_graphs
            total_abs_error += torch.abs(out - batch.y).sum().item()
            num_samples += batch.num_graphs
            
    return total_loss / num_samples, total_abs_error / num_samples


def plot_training_metrics(history: dict, save_dir: str):
    """Generate a plot of training and validation metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Total Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Add last point value
    plt.annotate(f"{history['train_loss'][-1]:.4f}", 
                 (len(history['train_loss'])-1, history['train_loss'][-1]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

    # Plot MAE
    plt.subplot(2, 2, 2)
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    # Add last point value
    plt.annotate(f"{history['val_loss'][-1]:.4f}", 
                 (len(history['val_loss'])-1, history['val_loss'][-1]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

    # Plot accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    # Add last point value
    plt.annotate(f"{history['accuracy'][-1]:.4f}", 
                 (len(history['accuracy'])-1, history['accuracy'][-1]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    # Add last point value
    plt.annotate(f"{history['learning_rate'][-1]:.4e}", 
                 (len(history['learning_rate'])-1, history['learning_rate'][-1]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_metrics.png'))
    plt.close()


def save_predictions(true_vertices: np.ndarray, pred_vertices: np.ndarray, epoch: int, save_dir: str):    
    true_vertices = true_vertices.cpu().numpy() if torch.is_tensor(true_vertices) else true_vertices
    pred_vertices = pred_vertices.cpu().numpy() if torch.is_tensor(pred_vertices) else pred_vertices
    
    filename = os.path.join(save_dir, 'predictions.txt')
    with open(filename, 'a') as f:
        f.write(f"\nEpoch {epoch+1}\n")
        f.write("True_Position\tPredicted_Position\tAbsolute_Error\n")
        for true_pos, pred_pos in zip(true_vertices, pred_vertices):
            abs_error = abs(true_pos - pred_pos)
            f.write(f"{true_pos:.6f}\t{pred_pos:.6f}\t{abs_error:.6f}\n")
    
    summary_file = os.path.join(save_dir, 'prediction_summary.txt')
    with open(summary_file, 'a') as f:
        mean_error = abs(true_vertices - pred_vertices).mean()
        std_error = abs(true_vertices - pred_vertices).std()
        max_error = abs(true_vertices - pred_vertices).max()
        
        f.write(f"\nEpoch {epoch+1}\n")
        f.write("Prediction Summary Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of vertices: {len(true_vertices)}\n")
        f.write(f"Mean absolute error: {mean_error:.6f}\n")
        f.write(f"Standard deviation of error: {std_error:.6f}\n")
        f.write(f"Maximum absolute error: {max_error:.6f}\n")


def plot_predictions(true_vertices: np.ndarray, pred_vertices: np.ndarray, epoch: int, save_dir: str, filename: str = "scatter", train: bool = True, size: int = 2):
    """Generate a scatter plot of true vs predicted vertex positions."""
    plt.figure(figsize=(10, 10))
    
    # Plot predictions vs true values
    plt.scatter(true_vertices, pred_vertices, alpha=0.5, label='Predictions', s=size)
    
    # Plot perfect prediction line
    min_val = min(true_vertices.min(), pred_vertices.min())
    max_val = max(true_vertices.max(), pred_vertices.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', label='Perfect Prediction')
    plt.plot([min_val - 0.01, max_val - 0.01], [min_val + 0.01, max_val + 0.01], 'r--', label='1% Error')
    plt.plot([min_val + 0.01, max_val + 0.01], [min_val - 0.01, max_val - 0.01], 'r--', label='1% Error')

    
    plt.xlabel('True Vertex Position')
    plt.ylabel('Predicted Vertex Position')
    if train:
        plt.title(f'Vertex Position Predictions (Epoch {epoch+1})')
    if not train:
        plt.title(f'Vertex Position Predictions (Test Set)')

    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{filename}.png'))
    
    if train:
        plt.savefig(os.path.join(save_dir, "frames", f'{filename}__epoch{epoch+1}.png'))
    
    plt.close()
