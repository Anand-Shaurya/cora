import numpy as np
from K_means import k
from K_means import b_labels
from K_means import cluster_labels
#from smkmeans import smk_cluster_assignments
# Initialize an empty dictionary to store citation data
citation_data = {}

c_labels = np.array(b_labels)

# Read the 'cora.cites' file
with open('cora.cites', 'r') as file:
    for line in file:
        # Split each line into two paper IDs
        paper1, paper2 = map(int, line.strip().split())
        
        # Check if paper1 is already in the dictionary
        if paper1 in citation_data:
            citation_data[paper1].append(paper2)
        else:
            citation_data[paper1] = [paper2]

# Now citation_data is a dictionary where each paper ID maps to its cited papers

# Calculate degree centrality for each node in each cluster
cluster_central_nodes = {}
for cluster in range(7):
    cluster_nodes = np.where(c_labels == cluster)[0]

    # Initialize a dictionary to store cluster_degrees
    cluster_degrees = {}
    
    # Iterate over papers in this cluster
    for paper_id in cluster_nodes:
        # Check if paper_id exists in citation_data
        if paper_id in citation_data:
            cluster_degrees[paper_id] = sum(1 for cited_id in citation_data[paper_id] if cited_id in cluster_nodes)
    
    # Check if there are nodes in this cluster
    if cluster_degrees:
        central_node = max(cluster_degrees, key=cluster_degrees.get)
        cluster_central_nodes[cluster] = central_node

# Now cluster_central_nodes contains the degree central node for each cluster
print(cluster_central_nodes)
