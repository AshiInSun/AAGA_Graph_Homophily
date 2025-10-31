import networkx as nx


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

    adjusted_homophily = my_nominator/my_deniminator
    return adjusted_homophily