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

def node_homophily(G):
    total_of_nodes = G.number_of_nodes()
    homo_node = 0
    for node in G.nodes():
        local_homo = 0
        for neighbor in G.neighbors(node):
            if G.nodes[neighbor]['label'] == G.nodes[node]['label']:
                local_homo+=1
        local_homo = local_homo/G.degree(node)
        homo_node += local_homo
    homo_node = homo_node / total_of_nodes
    return homo_node

def somme_degres_label_k(G, k):
    somme = 0
    for node in G.nodes():
        if G.nodes[node]['label'] == k:
            somme += G.degree(node)
    return somme

#on peut tomber sur 0 : c'est normal (je pense).
def class_homophily(G):
    m = set(nx.get_node_attributes(G, 'label').values())
    homophilia = 0
    for label in m:

        somme_interieur = 0
        nk = 0
        for v in G.nodes():
            if G.nodes[v]['label'] == label:
                nk += 1
                nb_homo_neighbor = 0
                for neighbor in G.neighbors(v):
                    if G.nodes[neighbor]['label'] == label:
                        nb_homo_neighbor += 1
                somme_interieur += nb_homo_neighbor
        Dk = somme_degres_label_k(G, label)
        somme_interieur = somme_interieur / Dk
        nksurn = nk / G.number_of_nodes()
        resultat_intermediaire = somme_interieur - nksurn
        if resultat_intermediaire < 0:
            resultat_intermediaire = 0
        homophilia += resultat_intermediaire
    homophilia = homophilia / (m.__len__())
    return homophilia

g_edge_homophily = edge_homophily(G)
g_node_homophily = node_homophily(G)
g_class_homophily = class_homophily(G)
print(f"\nGraphe edge homophily : {g_edge_homophily}")
print(f"Graphe node homophily : {g_node_homophily}")
print(f"Graphe class homophily : {g_class_homophily}")
plot_graph(G)


def adjusted_homophily(G):
    edje_homophily= edge_homophily(G)
    m = set(nx.get_node_attributes(G, 'label').values())
    my_term = (2*G.number_of_edges())**2
    my_firstSum=0
    for label in m:
        my_firstSum+=somme_degres_label_k(G,label)**2/my_term

    my_nominator= edje_homophily-my_firstSum
    my_deniminator = 1-my_firstSum

    adjusted_homophily = my_nominator/my_deniminator
    return adjusted_homophily
