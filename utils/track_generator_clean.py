import os
import json
import time
import argparse
import numpy as np
import uproot


def sample_from(hist_array, hist_edges, n):
    # Compute cumulative probabilities
    bin_contents = hist_array
    cdf = np.cumsum(bin_contents)
    cdf /= cdf[-1]

    # Inverse transform sampling
    test_vals = np.random.rand(n)
    val_bins = np.searchsorted(cdf, test_vals)

    # Get bin centers
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    sampled_vals = bin_centers[val_bins]

    return sampled_vals

def trackz0Resolution_eta_vectorized(eta_array):
    eta_fabs = np.abs(eta_array)
    dz = np.where(eta_fabs < 0.5, 0.1,
         np.where(eta_fabs < 0.75, 0.1125,
         np.where(eta_fabs < 1, 0.125,
         np.where(eta_fabs < 1.25, 0.15,
         np.where(eta_fabs < 1.5, 0.175,
         np.where(eta_fabs < 1.75, 0.225,
         np.where(eta_fabs < 2, 0.275,
         np.where(eta_fabs < 2.25, 0.345,
         np.where(eta_fabs < 2.5, 0.415,
         np.where(eta_fabs < 3, 0.515,
         np.where(eta_fabs < 3.5, 0.565,
         np.where(eta_fabs < 4, 0.615, 0))))))))))))
    return dz * 0.1  # in cm

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

def combine_previous_events(filename):
    try:
        with open(filename, 'r') as f:
            previous_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        previous_data = []
    return previous_data


def main(args):
    with uproot.open(os.path.join(args.input_dir, "vertexInfo.root")) as inFile:
        h_genPV_z = inFile["GenPV_Z"].to_numpy()
        h_genPV_nTracks = inFile["GenPV_NumTracks"].to_numpy()
        h_track_pT = inFile["ptSIM"].to_numpy()
        h_track_eta = inFile["etaSIM"].to_numpy()

    h_genPV_z_array = h_genPV_z[0] / np.sum(h_genPV_z[0])
    h_genPV_nTracks_array = h_genPV_nTracks[0] / np.sum(h_genPV_nTracks[0])
    h_track_pT_array = h_track_pT[0] / np.sum(h_track_pT[0])
    h_track_eta_array = h_track_eta[0] / np.sum(h_track_eta[0])

    genPV_z_edges = h_genPV_z[1]
    track_pT_edges = h_track_pT[1]
    track_eta_edges = h_track_eta[1]

    nVertices = args.nVertices
    nTracksMax = args.nTracksMax
    print(f"Number of vertices: {nVertices}")
    print(f"Max tracks per vertex: {nTracksMax}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or initialize combined events
    try:
        with open(os.path.join(args.output_dir, "serializedEvents.json"), 'r') as f:
            all_events = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_events = []

    total_writing_time = 0
    total_gen_time = 0

    for i in range(args.numTests):
        gen_start_time = time.time()

        d_vertextracks = []
        z_vertex_positions = np.random.normal(0, 3.5, nVertices)
        z_vertex_normalized = (z_vertex_positions + 15) / 30

        for z_vertex in z_vertex_normalized:
            n_tracks = nTracksMax
            track_pT = sample_from(h_track_pT_array, track_pT_edges, n_tracks)
            track_eta = sample_from(h_track_eta_array, track_eta_edges, n_tracks)
            track_dz0 = trackz0Resolution_eta_vectorized(track_eta)
            track_dz0_normalized = track_dz0 / 30

            track_z0 = np.random.normal(z_vertex, track_dz0_normalized)
            v_tracks = list(zip(track_z0, track_dz0_normalized))
            d_vertextracks.append((z_vertex, v_tracks))

        gen_dur = time.time() - gen_start_time
        total_gen_time += gen_dur

        # Add to combined events
        all_events.append(d_vertextracks)

        # Write individual event file
        filename = f"events_{nVertices}V_{nTracksMax}T_{i+1}.json"
        output_path = os.path.join(args.output_dir, filename)
        print(f"writing event number {i+1} to {filename}")

        start_time = time.time()
        with open(output_path, "w") as outputFile:
            json.dump(d_vertextracks, outputFile)
        dur = time.time() - start_time
        total_writing_time += dur

    # Write combined events file
    with open(os.path.join(args.output_dir, "serializedEvents.json"), 'w') as f:
        json.dump(all_events, f)

    print(f"Total writing time: {round(total_writing_time * 100) / 100} seconds")
    print(f"Total generation time: {round(total_gen_time * 100) / 100} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate toy particle tracks from known CMS distributions')
    parser.add_argument('--nVertices', type=int, default=2, help='The number of primary vertices in the event')
    parser.add_argument('--nTracksMax', type=int, default=5, help='The maximum number of tracks per primary vertex')
    parser.add_argument("--numTests", type=int, default=1000, help='Number of tests to run')
    parser.add_argument('--input_dir', type=str, default="./datasets/", help='Where to find the \'vertexInfo.root\' file')
    parser.add_argument('--output_dir', type=str, default="./datasets/train/set1", help='Where to save the output JSON files')
    args = parser.parse_args()

    main(args)