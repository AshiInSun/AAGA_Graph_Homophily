import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import graph_homophily_measures
import experimental_comparaison

#Pour lire l'article et notre code : on a V les noeuds, n le nombre de noeuds, E les arêtes.
#Chaque v a un label y de v appartenant a {0, ... , m}
#On a d(v) le dégré d'un noeud i.e. le nombre d'arêtes connectées à v.
#On a N(v) l'ensemble des voisins de v i.e. les noeuds connectés à v par une arête. |N(v)| = d(v)
#On a n(k) le nombre de noeuds de label k. D(k) la somme des dégrés des noeuds de label k.

path_dataset_one = "datasets/TUD_DD_GML"
path_dataset_two = "datasets/TUD_ZINC_GML"


# Assigner des labels aléatoires (par exemple 3 classes)
#labels = {}
#for node in G.nodes():
#    labels[node] = random.randint(0, 2)  # Classes 0, 1, 2
#nx.set_node_attributes(G, labels, 'label')

#print(f"\nGraphe avec labels : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
#print(f"Distribution des labels : {[list(labels.values()).count(i) for i in range(3)]}")

def plot_graph(G, class_attr):
    pos = nx.spring_layout(G)
    node_labels = nx.get_node_attributes(G, class_attr)
    unique_labels = set(node_labels.values())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    node_colors = [color_map[node_labels[node]] for node in G.nodes()]
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500, font_color='white')
    plt.title("Graphe avec labels")
    plt.show()



# Tests
def testing_one_graph():
    label_G = 'chem'
    G = nx.read_gml("datasets/OGB_CODE2_GML/graph_10.gml")
    experimental_comparaison.normalize_inplace(G)
    g_edge_homophily = graph_homophily_measures.edge_homophily(G, class_attr=label_G)
    g_node_homophily = graph_homophily_measures.node_homophily(G, class_attr=label_G)
    g_class_homophily = graph_homophily_measures.class_homophily(G, class_attr=label_G)
    g_adjusted_homophily = graph_homophily_measures.adjusted_homophily(G, class_attr=label_G)
    g_unbiased_homophily = graph_homophily_measures.unbiased_homophily(G, class_attr=label_G)

    print(f"\nGraph edge homophily : {g_edge_homophily}")
    print(f"Graph node homophily : {g_node_homophily}")
    print(f"Graph class homophily : {g_class_homophily}")
    print(f"Graph adjusted homophily : {g_adjusted_homophily}")
    print(f"Graph unbiased homophily : {g_unbiased_homophily}")

    plot_graph(G, class_attr=label_G)

def main():

    print("Starting the test\n")
    print("Testing MOLPCBA from OGB, molecular graph, 26 node and 28 edge per graph on average.")
    print("Results :\n")
    experimental_comparaison.experimental_comparaison(path_dataset_one, label_G="chem")
    print("\n")
    print("Testing CODE2 from OGB, AST graph, 125 node and 124 edge per graph on average.")
    print("Results :\n")
    experimental_comparaison.experimental_comparaison(path_dataset_two, label_G="chem")
    print("\n")

    print("Table 5 loading...")
    experimental_comparaison.all_homophilia_onaverage_all_datasets(label_G="chem")
    print("\n")
    print("Ending the test")

main()