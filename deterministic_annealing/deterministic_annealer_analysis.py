import os
import re
import glob
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

def setup_logging(save_dir):
    """Setup logging configuration."""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'analysis.log')
    
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

def match_vertices(true_vertices, pred_vertices):
    # Use Hungarian algorithm to match vertices
    dist_matrix = cdist(np.array(true_vertices).reshape(-1, 1), np.array(pred_vertices).reshape(-1, 1))
    n = min(len(true_vertices), len(pred_vertices))
    true_matched = []
    pred_matched = []
    
    for i in range(n):
        min_idx = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
        true_matched.append(true_vertices[min_idx[0]])
        pred_matched.append(pred_vertices[min_idx[1]])
        dist_matrix[min_idx[0], :] = float('inf')
        dist_matrix[:, min_idx[1]] = float('inf')
    
    return true_matched, pred_matched

def get_file_number(filename):
    """Extract number from filename using regex"""
    match = re.search(r'_(\d+)\.json$', filename)
    if match:
        return int(match.group(1))
    return None

def plot_comparison(true_data_file, annealer_response_file, plots_dir):
    """Plot comparison between true data and annealer response"""
    # Load data
    with open(true_data_file) as f:
        true_data = json.load(f)
    with open(annealer_response_file) as f:
        annealer_data = json.load(f)
    
    # Extract true data
    true_vertices = []
    tracks = []
    for vertex_data in true_data:
        true_vertices.append(vertex_data[0])
        for track in vertex_data[1]:
            tracks.append(track[0])
    
    # Parse annealer assignments
    energy = annealer_data[0][0]
    assignments = np.array([int(x) for x in annealer_data[0][1]])
    n_tracks = len(tracks)
    n_vertices = len(assignments) // n_tracks
    assignment_matrix = assignments.reshape(n_tracks, n_vertices)
    
    # Get predicted vertex positions
    predicted_vertices = []
    for vertex_idx in range(n_vertices):
        vertex_tracks = [tracks[i] for i in range(n_tracks) if assignment_matrix[i, vertex_idx] == 1]
        if vertex_tracks:
            predicted_vertices.append(np.mean(vertex_tracks))
    
    # Match vertices
    true_matched, pred_matched = match_vertices(true_vertices, predicted_vertices)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Track distribution and vertices
    sns.kdeplot(data=tracks, ax=ax1, label='Track Distribution')
    ax1.vlines(true_vertices, ymin=0, ymax=ax1.get_ylim()[1], color='g', label='True Vertices', linestyles='dashed')
    ax1.vlines(predicted_vertices, ymin=0, ymax=ax1.get_ylim()[1], color='r', label='Predicted Vertices', alpha=0.5)
    ax1.set_title(f'Track Distribution and Vertex Positions\nEnergy: {energy}')
    ax1.legend()
    
    # Plot 2: True vs Predicted comparison
    ax2.scatter(true_matched, pred_matched, alpha=0.5)
    all_vals = true_matched + pred_matched
    min_val, max_val = min(all_vals), max(all_vals)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    ax2.set_xlabel('True Vertex Positions')
    ax2.set_ylabel('Predicted Vertex Positions')
    ax2.set_title('True vs Predicted Vertex Positions')
    
    plt.tight_layout()
    
    # Get file number for consistent naming
    file_num = get_file_number(os.path.basename(true_data_file))
    output_filename = f'annealer_comparison_{file_num}.png'
    output_path = os.path.join(plots_dir, output_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.savefig(output_path)
    plt.close()
    
    # Calculate additional metrics
    accuracy = np.mean(np.abs(np.array(true_matched) - np.array(pred_matched)) < 0.01)

    # Log results
    logging.info(f"\nResults for file {file_num}:")
    logging.info(f"True vertices: {len(true_vertices)}")
    logging.info(f"Predicted vertices: {len(predicted_vertices)}")
    logging.info(f"Matched pairs: {len(true_matched)}")
    logging.info(f"Saved plot to: {output_path}\n")
    logging.info(f"Accuracy within 1%: {accuracy:.4f}")
    logging.info(f"---------------------------------")

    print(accuracy)
    return accuracy

    

def main(args):
    output_dir = os.path.join(args.output_dir)

    # Delete existing log and prediction files if they exist
    analysis_log = os.path.join(output_dir, 'analysis.log')
    if os.path.exists(analysis_log):
        os.remove(analysis_log)
    
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)

    # Get all event files and response files
    event_files = glob.glob(os.path.join(args.dataset_dir, "events_*.json"))
    response_files = glob.glob(os.path.join(args.response_dir, "serializableResponse_*.json"))

    # Sort files by number to ensure matching
    event_files.sort(key=lambda x: get_file_number(x))
    response_files.sort(key=lambda x: get_file_number(x))

    if not event_files:
        print(f"No event files found in {args.dataset_dir}")
        return
        
    if not response_files:
        print(f"No response files found in {args.response_dir}")
        return

    print(f"Found {len(event_files)} event files and {len(response_files)} response files")

    accuracies = []

    # Process each pair of files
    for event_file in event_files:
        file_num = get_file_number(event_file)
        if file_num is None:
            print(f"Couldn't extract file number from {event_file}, skipping...")
            continue
            
        # Find matching response file
        response_file = os.path.join(args.response_dir, f"serializableResponse_{file_num}.json")
        if not os.path.exists(response_file):
            print(f"No matching response file found for {event_file}, skipping...")
            continue
            
        print(f"Processing file pair {file_num}...")
        try:
            accuracy = plot_comparison(event_file, response_file, args.plots_dir)
            accuracies.append(accuracy)
            
        except Exception as e:
            print(f"Error processing file {file_num}: {str(e)}")
            continue

    logger.info(f"Average accuracy within 0.1%: {np.mean(accuracies):.4f}")
    print("Finished processing all files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize performance of the deterministic annealing algorithm')
    # parser.add_argument('--dataset_name', type=str, required=True, help='The name you want to call the dataset')
    parser.add_argument('--dataset_dir', type=str, default="../datasets/test/set1", help='The location of the dataset containing event files')
    parser.add_argument('--output_dir', type=str, default="./test/set1/", help='Directory to save the output plots')
    parser.add_argument('--response_dir', type=str, default="./test/set1/jsons", help='The location of the annealer response files')
    parser.add_argument('--plots_dir', type=str, default="./test/set1/plots", help='Directory to save the output plots')
    args = parser.parse_args()

    main(args)