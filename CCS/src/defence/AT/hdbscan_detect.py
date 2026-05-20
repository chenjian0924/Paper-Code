import torch
import numpy as np
from collections import OrderedDict, Counter
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

def hdbscan_detect(
        client_packages: OrderedDict[int, Dict[str, Any]]
) -> List[int]:
    client_ids = list(client_packages.keys())
    num_clients = len(client_ids)
    stat_vectors = []
    for cid in client_ids:
        package = client_packages[cid]
        params_source = package.get("model_params", package.get("client_gradient", {}))
        stat_params_list = [
            param.cpu().view(-1)
            for name, param in params_source.items()
            if 'running_mean' in name or 'running_var' in name
        ]
        client_vector = torch.cat(stat_params_list).numpy()
        stat_vectors.append(client_vector)
    X_stats = np.array(stat_vectors)
    similarity_matrix = cosine_similarity(X_stats)
    X_distance = 1 - similarity_matrix
    np.fill_diagonal(X_distance, 0)
    X_distance[X_distance < 0] = 0
    X_distance = X_distance.astype(np.float64)
    min_size = num_clients // 2 + 2
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        metric='precomputed',
        allow_single_cluster=True,
        min_samples = 1
    )
    clusterer.fit(X_distance)
    labels = clusterer.labels_
    benign_indices = set()
    if labels.max() < 0:
        benign_indices = set(range(num_clients))
    else:
        label_counts = Counter(l for l in labels if l != -1)
        if not label_counts:
            benign_indices = set(range(num_clients))
        else:
            largest_cluster_label = label_counts.most_common(1)[0][0]
            benign_indices = set(np.where(labels == largest_cluster_label)[0])
    all_indices = set(range(num_clients))
    malicious_indices = all_indices - benign_indices
    malicious_client_ids = [client_ids[i] for i in malicious_indices]

    return malicious_client_ids