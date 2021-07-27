import pandas as pd
import numpy as np
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, Topology
from castle.algorithms import TTPM
from castle.competition import submission

# Read historic alarm data and topology data.
alarm_data = pd.read_csv('./datasets/2/Alarm.csv', encoding ='utf')
topology_matrix = np.load('./datasets/2/Topology.npy')
# Data preprocessing and causal structure learning
X = alarm_data.iloc[:,0:3]
X.columns=['event','node','timestamp']
X = X.reindex(columns=['event','timestamp','node'])
# causal structure learning using TTPM
ttpm = TTPM(topology_matrix)
ttpm.learn(X, max_hop=1,max_iter=20)
# Obtain estimated causal structure and save it
est_causal_matrix = ttpm.causal_matrix.to_numpy()
np.save('./est_graphs/2.npy',est_causal_matrix)