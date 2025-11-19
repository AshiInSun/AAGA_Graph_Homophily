import networkx as nx
import numpy as np

def edge_homophily(G, class_attr):
    total_of_edges = G.number_of_edges()
    homo_edges = 0
    for edge in G.edges():
        if G.nodes[edge[0]][class_attr] == G.nodes[edge[1]][class_attr]:
            homo_edges+=1
    edge_homophilic_measure = homo_edges / total_of_edges
    return edge_homophilic_measure

def node_homophily(G, class_attr):
    total_of_nodes = G.number_of_nodes()
    homo_node = 0
    for node in G.nodes():
        local_homo = 0
        for neighbor in G.neighbors(node):
            if G.nodes[neighbor][class_attr] == G.nodes[node][class_attr]:
                local_homo+=1
        local_homo = local_homo/G.degree(node)
        homo_node += local_homo
    if total_of_nodes==0:
        return 0
    #c'est que soit le graph est vide, soit il n'est composé que de noeuds sans aucune arete
    #et on peut considerer qu'il n'est pas du tout homophile.
    homo_node = homo_node / total_of_nodes
    return homo_node

def somme_degres_label_k(G, k, class_attr):
    somme = 0
    for node in G.nodes():
        if G.nodes[node][class_attr] == k:
            somme += G.degree(node)
    return somme

#on peut tomber sur 0 : c'est normal (je pense).
def class_homophily(G, class_attr):
    m = set(nx.get_node_attributes(G, class_attr).values())
    homophilia = 0
    for label in m:

        somme_interieur = 0
        nk = 0
        for v in G.nodes():
            if G.nodes[v][class_attr] == label:
                nk += 1
                nb_homo_neighbor = 0
                for neighbor in G.neighbors(v):
                    if G.nodes[neighbor][class_attr] == label:
                        nb_homo_neighbor += 1
                somme_interieur += nb_homo_neighbor
        Dk = somme_degres_label_k(G, label, class_attr)
        somme_interieur = somme_interieur / Dk
        nksurn = nk / G.number_of_nodes()
        resultat_intermediaire = somme_interieur - nksurn
        if resultat_intermediaire < 0:
            resultat_intermediaire = 0
        homophilia += resultat_intermediaire
    homophilia = homophilia / (m.__len__())
    return homophilia

def adjusted_homophily(G, class_attr):
    edje_homophily= edge_homophily(G, class_attr)
    m = set(nx.get_node_attributes(G, class_attr).values())
    my_term = (2*G.number_of_edges())**2
    my_firstSum=0
    for label in m:
        my_firstSum+=somme_degres_label_k(G,label, class_attr)**2/my_term

    my_nominator= edje_homophily-my_firstSum
    my_deniminator = 1-my_firstSum
    if my_deniminator==0:
        return 1

    adjusted_homophily = my_nominator/my_deniminator
    return adjusted_homophily

def compute_cii(G, class_atr):
    """
    Calcule uniquement les cii normalisés : (définition 3, page 5 de l'article)
    cii = (# arêtes intra-classe i) / |E|
    Retourne un array des cii dans un ordre déterminé par les classes triées.
    """

    E = G.number_of_edges()
    classes = sorted(set(nx.get_node_attributes(G, class_atr).values()))
    cii = dict.fromkeys(classes, 0)

    for u, v in G.edges():
        cu = G.nodes[u][class_atr]
        cv = G.nodes[v][class_atr]
        if cu == cv:
            cii[cu] += 1

    # normalisation : diviser par |E|
    cii_vec = np.array([cii[c] / E for c in classes], dtype=float)
    return cii_vec


def unbiased_homophily(G, class_attr):
    cii_vec = compute_cii(G, class_attr)

    S = np.sum(np.sqrt(cii_vec))
    numerator = S**2 - 1
    denominator = S**2 + 1 - 2 * np.sum(cii_vec)

    if denominator == 0:    #une seule classe dans le graphe
        return 1.0  

    return numerator / denominator
