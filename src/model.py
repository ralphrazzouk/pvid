import json
import logging
import numpy as np
from typing import List
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrackDataset(Dataset):
    """Custom dataset for track data to predict vertex positions."""
    
    def __init__(self, file_paths: List[str]):
        """
        Initialize dataset from list of file paths.
        
        Args:
            file_paths: List of paths to JSON files containing track data
        """
        super().__init__()
        self.file_paths = file_paths
        self.data_list = []
        self._process_files()
        
    def _process_files(self):
        """Process all data files and create graph objects."""
        for file_path in self.file_paths:
            try:
                with open(file_path, 'r') as f:
                    event = json.load(f)
                
                # Process each vertex and its tracks in the event
                for vertex_pos, tracks in event:
                    data = self._create_graph_from_tracks(vertex_pos, tracks)
                    if data is not None:
                        self.data_list.append(data)
                        
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                
    def _create_graph_from_tracks(self, vertex_pos: float, 
                                tracks: List[List[float]]) -> Data:
        """
        Create a graph object from tracks associated with a vertex.
        
        Args:
            vertex_pos: True vertex position
            tracks: List of [z0, dz0_normalized] for each track
            
        Returns:
            PyG Data object containing the graph
        """
        try:
            counter = 0
            track_array = np.array(tracks)
            
            # Node features: [z0, dz0_normalized]
            x = torch.tensor(track_array, dtype=torch.float)
            
            # Create edges between all pairs of tracks
            edge_index = []
            edge_attr = []
            n_tracks = len(tracks)
            if counter == 0:
                # print("track_array:", track_array)
                # print("x:", x)
                # print("n_tracks:", n_tracks)
                # print("tracks:", tracks)
                counter += 1
                
            for i in range(n_tracks):
                for j in range(n_tracks):
                    if i != j:
                        edge_index.extend([[i, j]])
                        
                        # Edge features: [distance, uncertainty_product]
                        dist = abs(track_array[i, 0] - track_array[j, 0])
                        uncert_prod = track_array[i, 1] * track_array[j, 1]
                        edge_attr.append([dist, uncert_prod])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            # Target vertex position
            y = torch.tensor([vertex_pos], dtype=torch.float)
            
            return Data(x=x, 
                      edge_index=edge_index,
                      edge_attr=edge_attr,
                      y=y)
                      
        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}")
            return None
            
    def len(self):
        return len(self.data_list)
        
    def get(self, idx):
        return self.data_list[idx]


class TrackConv(MessagePassing):
    """Custom Graph Conv layer for track data."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        """
        Initialize the track convolution layer.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
        """
        super().__init__(aggr='mean')  # Mean aggregation
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        residual = x
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        return self.norm(out + residual)
        
    def message(self, x_j, edge_attr):
        return self.mlp(torch.cat([x_j, edge_attr], dim=-1))


class VertexGNN(nn.Module):
    """Graph Neural Network for vertex prediction."""
    
    def __init__(self, node_features: int = 2, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        """
        Initialize the GNN model.
        
        Args:
            node_features: Number of input node features
            hidden_dim: Size of hidden layers
            num_layers: Number of graph conv layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Initial node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # Using GELU instead of ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Graph convolution layers
        self.convs = nn.ModuleList([
            TrackConv(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Layer norms for each conv layer
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Final MLP for vertex prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, data):
        """Forward pass of the model."""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial node embedding
        x = self.node_embed(x)
        
        # Graph convolutions with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_new = norm(conv(x, edge_index, edge_attr))
            x = x + x_new  # Residual connection
            
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final MLP
        vertex_pos = self.mlp(x)
        
        return vertex_pos

    def compute_physics_loss(self, out, data):
        """Compute physics-based loss using track parameters."""
        z0 = data.x[:, 0]  # Track z0 positions
        dz0 = data.x[:, 1]  # Track z0 uncertainties
        
        # Convert output to match z0 dimensions
        out = out.squeeze()  # Remove the extra dimension, now shape [64]
        
        # Each track in data.batch maps to a vertex in out
        vertex_pos_expanded = out[data.batch]  # This will have shape [192]
        
        # Compute chi-square for each track
        vertex_chi2 = ((z0 - vertex_pos_expanded) / dz0) ** 2
        return vertex_chi2.mean()
