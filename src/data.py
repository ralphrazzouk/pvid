import os
import json
import numpy as np
import seaborn as sns
from typing import Tuple, List
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_file_paths(data_dir: str) -> List[str]:
    """Get all JSON file paths from the data directory."""
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json') and f != 'serializedEvents.json']


def split_data(file_paths: List[str], val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Split file paths into train, validation and test sets."""
    train_files, test_files = train_test_split(
        file_paths, 
        test_size=test_size, 
        random_state=random_state
    )
    
    train_files, val_files = train_test_split(
        train_files,
        test_size=val_size/(1-test_size),
        random_state=random_state
    )
    
    return train_files, val_files, test_files


def collect_data_from_files(files_dir: str) -> dict:
    vertex_positions = []
    track_z0s = []
    track_uncertainties = []
    track_labels = []
    n_events = []
    n_vertices = []
    n_tracks = []
    
    filenames = sorted([f for f in os.listdir(files_dir) 
                       if f.startswith("events") and f.endswith(".json")])
    
    for filename in filenames:
        with open(os.path.join(files_dir, filename), 'r') as f:
            event = json.load(f)
        
        n_events.append(len(event))
        n_vertices.append(len(event[0]))
        event_tracks = sum(len(tracks) for _, tracks in event)
        n_tracks.append(event_tracks)

        
        for vertex_idx, (vertex_pos, tracks) in enumerate(event):
            for track in tracks:
                vertex_positions.append(vertex_pos)
                track_z0s.append(track[0])
                track_uncertainties.append(track[1])
                track_labels.append(vertex_idx)
    
    return {
        'vertex_positions': np.array(vertex_positions),
        'track_z0s': np.array(track_z0s),
        'track_uncertainties': np.array(track_uncertainties),
        'track_labels': np.array(track_labels),
        'n_events': np.array(n_events),
        'n_vertices': np.array(n_vertices),
        'n_tracks': np.array(n_tracks)
    }


def plot_dataset_distributions(data, save_dir: str):
    fig = plt.figure(figsize=(20, 15))
    
    # Track z0 distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data['track_z0s'], bins=100)
    plt.title('Track z0 Distribution (All Events)')
    plt.xlabel('z0 position')
    
    # Track uncertainty distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data['track_uncertainties'], bins=100)
    plt.title('Track Uncertainty Distribution')
    plt.xlabel('dz0 normalized')
    
    # # Number of vertices per event
    # plt.subplot(3, 2, 3)
    # sns.histplot(data['n_vertices'], bins=np.arange(min(data['n_vertices']), max(data['n_vertices'])+2)-0.5)
    # plt.title('Number of Vertices per Event')
    # plt.xlabel('Number of vertices')
    
    # Number of tracks per event
    plt.subplot(2, 2, 3)
    sns.histplot(data['n_tracks'], bins=30)
    plt.title('Number of Tracks per Event')
    plt.xlabel('Number of tracks')
    
    # # z0 vs uncertainty scatter
    # plt.subplot(3, 2, 5)
    # sns.scatterplot(x=data['track_z0s'], y=data['track_uncertainties'], alpha=0.1, s=5)
    # plt.title('z0 vs Uncertainty')
    # plt.xlabel('z0 position')
    # plt.ylabel('dz0 normalized')
    
    # Delta z0 between tracks and their vertices
    plt.subplot(2, 2, 4)
    delta_z0 = data['track_z0s'] - data['vertex_positions']
    sns.histplot(delta_z0, bins=100)
    plt.title('Track-Vertex Distance Distribution')
    plt.xlabel('z0 - vertex_position')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plots', f'dataset_distributions.png'), dpi=300)
    plt.close()

    # Additional plots
    # Pull distribution
    plt.figure(figsize=(10, 6))
    pull = (data['track_z0s'] - data['vertex_positions']) / data['track_uncertainties']
    sns.histplot(pull, bins=100)
    plt.title('Pull Distribution')
    plt.xlabel('(z0 - vertex_position) / uncertainty')
    plt.savefig(os.path.join(save_dir, 'plots', f'pull_distribution.png'), dpi=300)
    plt.close()
    
    # 2D histogram of z0 vs uncertainty
    plt.figure(figsize=(10, 6))
    plt.hist2d(data['track_z0s'], data['track_uncertainties'], bins=(100, 100), norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(label='Number of tracks')
    plt.title('Track z0 vs Uncertainty (2D)')
    plt.xlabel('z0 position')
    plt.ylabel('dz0 normalized')
    plt.savefig(os.path.join(save_dir, 'plots', f'z0_uncertainty_2d.png'), dpi=300)
    plt.close()


def print_statistics(data):
    stats = {
        'Total Events  ': len(data['n_events']),
        'Total Vertices': np.sum(data['n_vertices']),
        'Total Tracks  ': len(data['track_z0s']),
        'z0 range              ': (np.min(data['track_z0s']), np.max(data['track_z0s'])),
        'Average vertices/event': np.mean(data['n_vertices']),
        'Average tracks/event  ': np.mean(data['n_tracks']),
        'Mean uncertainty  ': np.mean(data['track_uncertainties']),
        'Median uncertainty': np.median(data['track_uncertainties'])
    }
    
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")