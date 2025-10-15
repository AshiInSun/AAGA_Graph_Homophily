import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

#Pour lire l'article et notre code : on a V les noeuds, n le nombre de noeuds, E les arêtes.
#Chaque v a un label y de v appartenant a {0, ... , m}
#On a d(v) le dégré d'un noeud i.e. le nombre d'arêtes connectées à v.
#On a N(v) l'ensemble des voisins de v i.e. les noeuds connectés à v par une arête. |N(v)| = d(v)
#On a n(k) le nombre de noeuds de label k. D(k) la somme des dégrés des noeuds de label k.



G = nx.erdos_renyi_graph(n=30, p=0.2, seed=42)
# Assigner des labels aléatoires (par exemple 3 classes)
labels = {}
for node in G.nodes():
    labels[node] = random.randint(0, 2)  # Classes 0, 1, 2
nx.set_node_attributes(G, labels, 'label')

print(f"\nGraphe avec labels : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
print(f"Distribution des labels : {[list(labels.values()).count(i) for i in range(3)]}")

def plot_graph(G):
    pos = nx.spring_layout(G)
    node_labels = nx.get_node_attributes(G, 'label')
    unique_labels = set(node_labels.values())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    node_colors = [color_map[node_labels[node]] for node in G.nodes()]
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500, font_color='white')
    plt.title("Graphe avec labels")
    plt.show()

def edge_homophily(G):
    total_of_edges = G.number_of_edges()
    homo_edges = 0
    for edge in G.edges():
        if G.nodes[edge[0]]['label'] == G.nodes[edge[1]]['label']:
            homo_edges+=1
    edge_homophilic_measure = homo_edges / total_of_edges
    return edge_homophilic_measure

g_edge_homophily = edge_homophily(G)
print(f"\nGraphe homophily : {g_edge_homophily}")
plot_graph(G)