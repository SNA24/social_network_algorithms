import networkx as nx
import os, sys, time
import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from mining.diameter import diameter, parallel_diam, stream_diam
from mining.triangles import num_triangles, num_triangles_parallel
from mining.clustering.girman_newman import girman_newman, heuristic_girman_newman, parallel_girman_newman, parallel_heuristic_girman_newman
from mining.clustering.spectral import spectral, spectral_parallel, spectral_multi_cluster, spectral_multi_cluster_parallel, spectral_multi_cluster_v2, spectral_multi_cluster_v2_parallel
from mining.clustering.two_means import two_means, parallel_two_means
from mining.clustering.hierarchical import hierarchical, parallel_hierarchical

to_test = {
    # diameter: False,
    # parallel_diam: True,
    stream_diam: False,
    num_triangles: False,
    num_triangles_parallel: True,
    girman_newman: False,
    heuristic_girman_newman: False,
    parallel_girman_newman: True,
    parallel_heuristic_girman_newman: True,
    spectral: False,
    spectral_parallel: True,
    spectral_multi_cluster: False,
    spectral_multi_cluster_parallel: True,
    spectral_multi_cluster_v2: False,
    spectral_multi_cluster_v2_parallel: True,
    hierarchical: False,
    parallel_hierarchical: True,
    two_means: False,
    parallel_two_means: True
}

def test_network(G, name, to_test, topic):

    num_jobs = [2, 4, 6, 8, 10, 12, 14, 16]

    dataframe = pd.DataFrame(columns=['function', 'num_jobs', 'execution_time'])
    
    for func, parallel in to_test.items():
        if parallel:
            for j in num_jobs:
                start = time.time()
                func(G, j)
                end = time.time()
                dataframe = dataframe._append({'function': func.__name__, 'num_jobs': j, 'execution_time': end - start}, ignore_index=True)
        else:
            start = time.time()
            func(G)
            end = time.time()
            dataframe = dataframe._append({'function': func.__name__, 'num_jobs': 1, 'execution_time': end - start}, ignore_index=True)
        
    dataframe.to_csv(f'test/{topic}_results_{name}.csv', index=False)

if __name__ == '__main__':

    zachary = nx.Graph()
    with open('test/ucidata-zachary/out.ucidata-zachary') as f:
        lines = f.readlines()
        lines = lines[2:]
        for line in lines:
            line = line.strip().split(' ')
            from_node = line[0]
            to_node = line[1]
            zachary.add_edge(from_node, to_node)

    graphs = {
        'zachary': zachary
    }

    for name, G in graphs.items():
        test_network(G, name, to_test, 'mining')

        
