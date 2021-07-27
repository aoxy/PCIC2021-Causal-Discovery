# -*- coding: utf-8 -*-
from __future__ import print_function  # do not delete this line if you want to save your log file.

import os

import pandas as pd 
import numpy as np 
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.algorithms import TTPM
from castle.competition import submission

from naie.context import Context
import moxing as mox 
from naie.datasets import get_data_reference


def arrs_to_csv(arrs, input_path='submit.csv'):
    """
    This can be used to generate the submission file in .csv

    Parameters:
        arrs: list of your solutions for each dataset; each element should be a numpy array of 0 or 1
        input_path: where to save your file; e.g., submit.csv
    -------

    """
    arrs_str = [arr_to_string(arr) for arr in arrs]
    pd.DataFrame(arrs_str).to_csv(input_path, index=False)
    # this copy the output for final submission
    mox.file.copy(input_path, os.path.join(Context.get_output_path(), 'submit.csv'))


def arr_to_string(mat):
    """
    Parameters
        mat: numpy array with each entry either 0 or 1

    Returns:
        string of the input array
    """
    mat_int = mat.astype(int)
    mat_flatten = mat_int.flatten().tolist()
    for m in mat_flatten:
        if m not in [0, 1]:
            raise TypeError("Value not in {0, 1}.")
    mat_str = ' '.join(map(str, mat_flatten))
    return mat_str


def remove_diagnal_entries(mat):
    """
    set the diagonal of a matrix to be 0
    """
    mat_copy = np.copy(mat)
    indices_diag = np.diag_indices(len(mat_copy))
    mat_copy[indices_diag] = 0
    return mat_copy


datasets = {}
## this is used for constructing dags with right shapes
mat_shapes_phase1 = [10, 11, 12, 13, 13, 14, 15, 16, 17, 18, 13, 20, 21, 16, 18, 24, 25, 26, 27, 29]

## import data; note that you need to first register dataset from naie
data_reference = get_data_reference(dataset='DatasetService', dataset_entity='discovery_train', enable_local_cache=True)


for i in range(1, 21):
    base_dir = os.path.join('/cache/datasets/DatasetService/discovery_train/', str(i))
    X = pd.read_csv(os.path.join(base_dir, 'Alarm.csv'), encoding='utf-8')
    ## just to create a dag with right shapes, not the true dags
    ## we provided true dags for datasets 1-4, so you may use them here
    true_dag = np.ones((mat_shapes_phase1[i-1], mat_shapes_phase1[i-1]))
    
    if os.path.exists(os.path.join(base_dir, 'Topology.npy')):
        topology_matrix = np.load(os.path.join(base_dir, 'Topology.npy'))
    else:
        topology_matrix = None
    
    datasets[i] = (X, topology_matrix, true_dag)
    

results = {}

for k in datasets:
    print(k, datasets[k][0].shape)
    X, topology_matrix, true_causal_matrix = datasets[k]
    train_data = X.iloc[:, 0:3]
    train_data.columns = ['event', 'node', 'timestamp']
    train_data = train_data.reindex(columns=['event', 'timestamp', 'node'])

    if not isinstance(topology_matrix, np.ndarray):
        num_nodes = len(set(train_data['node']))
        topology_matrix = np.zeros((num_nodes, num_nodes))
        ttpm = TTPM(topology_matrix, max_iter=20)
        ttpm.learn(train_data, max_hop=0)
    else:
        ttpm = TTPM(topology_matrix, max_iter=20)
        ttpm.learn(train_data, max_hop=1)
    
    ## with default setting, ttpm has diagonal entries, which means a node in the past may affect itself in the current
    ## the true graphs (DAGs) have ignored self-impact, so we remove the diagonal entries here  
    
    results[k] = (remove_diagnal_entries(ttpm.causal_matrix.values), true_causal_matrix)
   

total_rank_score = 0

for k in range(1, 21):
    est_graph_matrix, true_causal_matrix = results[k]
    gscore = MetricsDAG(est_graph_matrix, true_causal_matrix).metrics['gscore']
    total_rank_score += gscore
    print(k)
    print(MetricsDAG(est_graph_matrix, true_causal_matrix).metrics)

print('average score', total_rank_score/20)


### generate submission file
submit_list = [results[k][0] for k in range(1,21)]
arrs_to_csv(submit_list, input_path='/cache/submit.csv')

