# read ucidata-zachary/out.ucidata-zachary

import networkx as nx
import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from mining.diameter import diameter, parallel_diam, stream_diam
from mining.triangles import num_triangles, num_triangles_parallel
from mining.clustering.girman_newman import girman_newman, heuristic_girman_newman, parallel_girman_newman, parallel_heuristic_girman_newman
from mining.clustering.spectral import spectral, spectral_parallel, spectral_multi_cluster, spectral_multi_cluster_parallel, spectral_multi_cluster_v2, spectral_multi_cluster_v2_parallel
from mining.clustering.two_means import two_means, parallel_two_means
from mining.clustering.hierarchical import hierarchical, parallel_hierarchical

if __name__ == '__main__':

    G = nx.Graph()
    with open('test/ucidata-zachary/out.ucidata-zachary') as f:
        lines = f.readlines()
        lines = lines[2:]
        for line in lines:
            line = line.strip().split(' ')
            from_node = line[0]
            to_node = line[1]
            G.add_edge(from_node, to_node)

    print(G.number_of_nodes())

    print("Stream diameter: ", stream_diam(G))
    print("Triangles: ", num_triangles(G))
    print("Parallel triangles: ", num_triangles_parallel(G, 4))

    print("Clusterings: ")

    print("Girman-Newman: ", girman_newman(G))
    print("Heuristic Girman-Newman: ", heuristic_girman_newman(G))
    print("Parallel Girman-Newman: ", parallel_girman_newman(G, 4))
    print("Parallel Heuristic Girman-Newman: ", parallel_heuristic_girman_newman(G, 4))

    print("Spectral: ", spectral(G))
    print("Parallel Spectral: ", spectral_parallel(G, 4))
    print("Multi-cluster Spectral: ", spectral_multi_cluster(G))
    print("Parallel Multi-cluster Spectral: ", spectral_multi_cluster_parallel(G, 4))
    print("Multi-cluster Spectral v2: ", spectral_multi_cluster_v2(G))
    # print("Parallel Multi-cluster Spectral v2: ", spectral_multi_cluster_v2_parallel(G, 4))

    print("Hierarchical: ", hierarchical(G))
    print("Parallel Hierarchical: ", parallel_hierarchical(G, 4))

    print("Two-means: ", two_means(G))
    print("Parallel Two-means: ", parallel_two_means(G, 4))

        
