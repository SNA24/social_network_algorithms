import networkx as nx
import os, sys, time
import pandas as pd
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from mining.diameter import diameter, parallel_diam, stream_diam
from mining.triangles import num_triangles #, num_triangles_parallel, find_triangles_directed, find_triangles_undirected
from mining.clustering.girvan_newman import girvan_newman, heuristic_girvan_newman, parallel_girvan_newman, parallel_heuristic_girvan_newman
from mining.clustering.spectral import spectral, spectral_parallel, spectral_multi_cluster, spectral_multi_cluster_parallel, spectral_multi_cluster_v2, spectral_multi_cluster_v2_parallel
from mining.clustering.two_means import two_means, parallel_two_means
from mining.clustering.hierarchical import hierarchical, parallel_hierarchical
from centrality_measures.betweenness import betweenness, parallel_betweenness
from centrality_measures.closeness import closeness, parallel_closeness
from centrality_measures.degree import degree, parallel_degree
from centrality_measures.shapley_degree import shapley_degree, parallel_shapley_degree
from centrality_measures.shapley_threshold import shapley_threshold, parallel_shapley_threshold
from centrality_measures.shapley_closeness import shapley_closeness, parallel_shapley_closeness
from centrality_measures.vote_rank import vote_rank, parallel_vote_rank
from centrality_measures.page_rank import page_rank, parallel_page_rank
from centrality_measures.HITS import hits_both, parallel_hits_both, hits_authority, parallel_hits_authority, hits_hubbiness, parallel_hits_hubbiness

to_test_mining = {
    # diameter: False,
    # parallel_diam: True,
    # stream_diam: False,
    # num_triangles: False,
    # find_triangles_directed: False,
    # find_triangles_undirected: False,
    # num_triangles_parallel: True,
    # girvan_newman: False,
    # heuristic_girvan_newman: False,
    # parallel_girvan_newman: True,
    # parallel_heuristic_girvan_newman: True,
    # spectral: False,
    # spectral_parallel: True,
    # spectral_multi_cluster: False,
    # spectral_multi_cluster_parallel: True,
    # spectral_multi_cluster_v2: False,
    # spectral_multi_cluster_v2_parallel: True,
    # hierarchical: False,
    # parallel_hierarchical: True,
    # two_means: False,
    # parallel_two_means: True
}

to_test_cenetrality_measures = {
    # betweenness: False,
    # parallel_betweenness: True,
    # closeness: False,
    # parallel_closeness: True,
    # degree: False,
    # parallel_degree: True,
    # shapley_degree: False,
    # parallel_shapley_degree: True,
    # shapley_threshold: False,
    # parallel_shapley_threshold: True,
    # shapley_closeness: False,
    # parallel_shapley_closeness: True,
    # vote_rank: False,
    # parallel_vote_rank: True,
    # page_rank: False,
    # parallel_page_rank: True,
    # hits_both: False,
    # parallel_hits_both: True,
    # hits_authority: False,
    # parallel_hits_authority: True,
    # hits_hubbiness: False,
    # parallel_hits_hubbiness: True
}

def test_network(G, name, to_test, topic):

    num_jobs = [2, 4, 6, 8, 10, 12, 14, 16]

    dataframe = pd.DataFrame(columns=['function', 'num_jobs', 'execution_time'])
    
    for func, parallel in to_test.items():
        if parallel:
            for j in num_jobs:
                start = time.time()
                print(name, func.__name__, func(G, j), j)
                end = time.time()
                dataframe = dataframe._append({'function': func.__name__, 'num_jobs': j, 'execution_time': end - start}, ignore_index=True)
        else:
            start = time.time()
            print(name, func.__name__, func(G))
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

    moreno = nx.DiGraph()
    with open('test/moreno_rhesus/out.moreno_rhesus_rhesus') as f:
        lines = f.readlines()
        lines = lines[2:]
        for line in lines:
            line = line.strip().split(' ')
            from_node = line[0]
            to_node = line[1]
            moreno.add_edge(from_node, to_node)

    moreno2 = nx.DiGraph()
    with open('test/moreno_sampson/out.moreno_sampson_sampson') as f:
        lines = f.readlines()
        lines = lines[2:]
        for line in lines:
            line = line.strip().split(' ')
            from_node = line[0]
            to_node = line[1]
            moreno2.add_edge(from_node, to_node)

    graphs = {
        'zachary': zachary,
        'moreno': moreno,
        'moreno2': moreno2
    }

    for name, G in graphs.items():
        test_network(G, name, to_test_mining, 'mining')
        test_network(G, name, to_test_cenetrality_measures, 'centrality_measures')


        
