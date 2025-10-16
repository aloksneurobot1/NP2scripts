import numpy as np

try:
    good_clusters = np.load('good_clusters_final.npy', allow_pickle=True)
    if good_clusters.size > 0:
        for i, cluster_info in enumerate(good_clusters):
            print(f"--- Cluster {i} ---")
            print(f"Type: {type(cluster_info)}")
            if isinstance(cluster_info, dict):
                print(f"Keys: {cluster_info.keys()}")
                if 'channel_depth_um' in cluster_info:
                    print(f"Channel Depth: {cluster_info['channel_depth_um']}")
            else:
                print(f"Content: {cluster_info}")
    else:
        print("No good clusters found in the file.")
except FileNotFoundError:
    print("Error: 'good_clusters_final.npy' not found.")