import os
import json
import time
import argparse
import numpy as np
import uproot


def sample_from(hist_array, hist_edges, n):
    """
    Sample n values from a histogram using inverse transform sampling
    
    Args:
        hist_array (np.array): Array of bin contents
        hist_edges (np.array): Array of bin edges
        n (int): Number of samples to generate
        
    Returns:
        np.array: Array of sampled values
    """
    # Compute cumulative probabilities
    bin_contents = hist_array
    cdf = np.cumsum(bin_contents)
    cdf /= cdf[-1]  # Normalize to [0,1]

    # Inverse transform sampling
    test_vals = np.random.rand(n)
    val_bins = np.searchsorted(cdf, test_vals)
    
    # Get bin centers for the sampled indices
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    sampled_vals = bin_centers[val_bins]
    
    return sampled_vals


def trackz0Resolution_eta_vectorized(eta_array):
    """
    Calculate track z0 resolution as a function of eta (vectorized version)
    Based on: https://twiki.cern.ch/twiki/bin/view/CMSPublic/TrackingPOGPerformance2017MC
    
    Args:
        eta_array (np.array): Array of eta values
        
    Returns:
        np.array: Array of z0 resolutions in cm
    """
    eta_fabs = np.abs(eta_array)
    dz = np.where(eta_fabs < 0.5, 0.1,
         np.where(eta_fabs < 1, 0.125,
         np.where(eta_fabs < 1.5, 0.175,
         np.where(eta_fabs < 2, 0.275,
         np.where(eta_fabs < 2.5, 0.415,
         np.where(eta_fabs < 4, 0.615, 0))))))
    return dz * 0.1  # in cm


def calculate_dunn_index(vertex_tracks):
    """
    Calculate the Dunn Index for vertex clustering quality
    
    Args:
        vertex_tracks (list): List of tuples containing (vertex_pos, tracks)
        
    Returns:
        float: Dunn Index value
    """
    min_dvivj = float('inf')
    max_dij = 0.0
    
    # Calculate maximum intra-cluster distance for each cluster
    for vertex1, tracks1 in vertex_tracks:
        # Intra-cluster distances
        track_positions = np.array([track[0] for track in tracks1])
        if len(track_positions) > 1:
            dists = np.abs(track_positions[:, None] - track_positions)
            max_dij = max(max_dij, np.max(dists[dists > 0]))
        
        # Inter-cluster distances
        for vertex2, _ in vertex_tracks:
            if vertex1 != vertex2:
                dvivj = abs(vertex1 - vertex2)
                min_dvivj = min(min_dvivj, dvivj)
    
    return min_dvivj / max_dij if max_dij > 0 else 0.0

def combine_event_files(input_dir, output_file):
    all_events = []
    
    # Get all event files
    files = sorted([f for f in os.listdir(input_dir) if f.startswith("events_") and f.endswith(".json")])
    
    for filename in files:
        with open(os.path.join(input_dir, filename), 'r') as f:
            event = json.load(f)
            all_events.append(event)
    
    # Write combined file
    with open(output_file, 'w') as f:
        json.dump(all_events, f)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate toy particle tracks from known CMS distributions')
    parser.add_argument('--nVertices', type=int, default=2, help='The number of primary vertices in the event')
    parser.add_argument('--nTracksMax', type=int, default=5, help='The maximum number of tracks per primary vertex')
    parser.add_argument("--numTests", type=int, default=5, help='Number of tests to run')
    parser.add_argument('--inputdir', type=str, default="./datasets/", help='Where to find the \'vertexInfo.root\' file')
    parser.add_argument('--outputdir', type=str, default="./datasets/trackgen_v1", help='Where to save the output JSON files')
    args = parser.parse_args()

    # Open the ROOT file using uproot
    with uproot.open(os.path.join(args.inputdir, "vertexInfo.root")) as inFile:
        # Read histograms as numpy arrays
        h_genPV_z = inFile["GenPV_Z"].to_numpy()
        h_genPV_nTracks = inFile["GenPV_NumTracks"].to_numpy()
        h_track_pT = inFile["ptSIM"].to_numpy()
        h_track_eta = inFile["etaSIM"].to_numpy()

    # Normalize histogram arrays
    h_genPV_z_array = h_genPV_z[0] / np.sum(h_genPV_z[0])
    h_genPV_nTracks_array = h_genPV_nTracks[0] / np.sum(h_genPV_nTracks[0])
    h_track_pT_array = h_track_pT[0] / np.sum(h_track_pT[0])
    h_track_eta_array = h_track_eta[0] / np.sum(h_track_eta[0])

    # Histogram edges for sampling
    genPV_z_edges = h_genPV_z[1]
    track_pT_edges = h_track_pT[1]
    track_eta_edges = h_track_eta[1]

    nVertices = args.nVertices
    nTracksMax = args.nTracksMax
    print(f"Number of vertices: {nVertices}")
    print(f"Max tracks per vertex: {nTracksMax}")

    # Create output directory if it doesn't exist
    os.makedirs(args.outputdir, exist_ok=True)
    
    total_writing_time = 0
    total_gen_time = 0
    
    for i in range(args.numTests):
        gen_start_time = time.time()
        
        # Initialize vertex-track list for this event
        d_vertextracks = []
        z_vertex_positions = np.random.normal(0, 3.5, nVertices)
        z_vertex_normalized = (z_vertex_positions + 15) / 30
        
        # Generate tracks for each vertex
        for z_vertex in z_vertex_normalized:
            # Number of tracks can be directly set to nTracksMax
            n_tracks = nTracksMax
            
            # Sample track parameters
            track_pT = sample_from(h_track_pT_array, track_pT_edges, n_tracks)
            track_eta = sample_from(h_track_eta_array, track_eta_edges, n_tracks)
            track_dz0 = trackz0Resolution_eta_vectorized(track_eta)
            track_dz0_normalized = track_dz0 / 30
            
            # Generate track z positions
            track_z0 = np.random.normal(z_vertex, track_dz0_normalized)
            
            # Create track list for this vertex
            v_tracks = list(zip(track_z0, track_dz0_normalized))
            d_vertextracks.append((z_vertex, v_tracks))
            
        gen_dur = time.time() - gen_start_time
        total_gen_time += gen_dur

        # Serialize d_vertextracks into a JSON file
        filename = f"events_{nVertices}V_{nTracksMax}T_{i+1}.json"
        output_path = os.path.join(args.outputdir, filename)
        
        print(f"Writing event number {i+1} to {filename}")
        
        # Measure writing time
        start_time = time.time()
        with open(output_path, "w") as outputFile:
            json.dump(d_vertextracks, outputFile)
            
        dur = time.time() - start_time
        total_writing_time += dur
        
        # Calculate and save Dunn index
        dunn_index = calculate_dunn_index(d_vertextracks)
        dunn_filename = os.path.join(args.outputdir, f"dunnIndex_{i+1}.json")
        with open(dunn_filename, "w") as outputFile:
            json.dump(dunn_index, outputFile)
            
    print(f"Total writing time: {round(total_writing_time * 100) / 100} seconds")
    print(f"Total generation time: {round(total_gen_time * 100) / 100} seconds")

    # After generating individual event files, combine them
    combine_event_files(args.outputdir, os.path.join(args.outputdir, "serializedEvents.json"))


if __name__ == "__main__":
    main()