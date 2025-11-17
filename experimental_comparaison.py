import os
import ogb
import networkx as nx
from torch_geometric.utils import to_networkx
from tqdm import trange

import graph_homophily_measures

path_dataset = "datasets/OGB_CODE2_GML"

def normalize_inplace(G):
    """
    Supprime tous les nœuds de degré 0 en modifiant `G` directement.
    """
    nodes_to_remove = [n for n, d in G.degree() if d == 0]
    if nodes_to_remove:
        G.remove_nodes_from(nodes_to_remove)

def experimental_comparaison(path, label_G):
    chemin_dossier = path
    max_files = 2000  # Nombre maximum de fichiers à traiter
    noms_fichiers = [f for f in os.listdir(chemin_dossier) if os.path.isfile(os.path.join(chemin_dossier, f))]

    edge_node_agreement = 0
    edge_class_agreement = 0
    edge_adjusted_agreement = 0
    edge_unbiased_agreement = 0
    node_class_agreement = 0
    node_adjusted_agreement = 0
    node_unbiased_agreement = 0
    class_adjusted_agreement = 0
    class_unbiased_agreement = 0
    adjusted_unbiased_agreement = 0

    num_pairs = min(len(noms_fichiers), max_files) // 2

    # Boucle pour ouvrir deux fichiers à chaque itération
    for k in trange(num_pairs, desc='Traitement des paires', unit='paire'):
        i = k * 2
        if i + 1 < len(noms_fichiers):  # Vérifie qu'il y a au moins deux fichiers restants
            fichier1 = os.path.join(chemin_dossier, noms_fichiers[i])
            fichier2 = os.path.join(chemin_dossier, noms_fichiers[i + 1])
            try:
                G1 = nx.read_gml(fichier1)
                G2 = nx.read_gml(fichier2)
                normalize_inplace(G1)
                normalize_inplace(G2)

                g1_edge_homophily = graph_homophily_measures.edge_homophily(G1, class_attr=label_G)
                g1_node_homophily = graph_homophily_measures.node_homophily(G1, class_attr=label_G)
                g1_class_homophily = graph_homophily_measures.class_homophily(G1, class_attr=label_G)
                g1_adjusted_homophily = graph_homophily_measures.adjusted_homophily(G1, class_attr=label_G)
                g2_edge_homophily = graph_homophily_measures.edge_homophily(G2, class_attr=label_G)
                g2_node_homophily = graph_homophily_measures.node_homophily(G2, class_attr=label_G)
                g2_class_homophily = graph_homophily_measures.class_homophily(G2, class_attr=label_G)
                g2_adjusted_homophily = graph_homophily_measures.adjusted_homophily(G2, class_attr=label_G)
                g1_unbiased_homophily = graph_homophily_measures.unbiased_homophily(G1, class_attr=label_G)
                g2_unbiased_homophily = graph_homophily_measures.unbiased_homophily(G2, class_attr=label_G)

                edge_result = g1_edge_homophily > g2_edge_homophily
                node_result = g1_node_homophily > g2_node_homophily
                class_result = g1_class_homophily > g2_class_homophily
                adjusted_result = g1_adjusted_homophily > g2_adjusted_homophily
                unbiased_result = g1_unbiased_homophily > g2_unbiased_homophily

                edge_node_result = edge_result == node_result
                edge_class_result = edge_result == class_result
                edge_adjusted_result = edge_result == adjusted_result
                edge_unbiased_result = edge_result == unbiased_result
                node_class_result = node_result == class_result
                node_adjusted_result = node_result == adjusted_result
                node_unbiased_result = node_result == unbiased_result
                class_adjusted_result = class_result == adjusted_result
                class_unbiased_result = class_result == unbiased_result
                adjusted_unbiased_result = adjusted_result == unbiased_result

                edge_node_agreement += edge_node_result
                edge_class_agreement += edge_class_result
                edge_adjusted_agreement += edge_adjusted_result
                edge_unbiased_agreement += edge_unbiased_result
                node_class_agreement += node_class_result
                node_adjusted_agreement += node_adjusted_result
                node_unbiased_agreement += node_unbiased_result
                class_adjusted_agreement += class_adjusted_result
                class_unbiased_agreement += class_unbiased_result
                adjusted_unbiased_agreement += adjusted_unbiased_result

            except Exception as e:
                print(f"Erreur lors du traitement des fichiers {fichier1} et {fichier2} : {e}")

    # Puis, on normalise les resultats par le nombre de comparaisons effectuées
    num_comparisons = min(max_files, len(noms_fichiers)) // 2
    if num_comparisons > 0:
        edge_node_agreement /= num_comparisons
        edge_class_agreement /= num_comparisons
        edge_adjusted_agreement /= num_comparisons
        edge_unbiased_agreement /= num_comparisons
        node_class_agreement /= num_comparisons
        node_adjusted_agreement /= num_comparisons
        node_unbiased_agreement /= num_comparisons
        class_adjusted_agreement /= num_comparisons
        class_unbiased_agreement /= num_comparisons
        adjusted_unbiased_agreement /= num_comparisons

    print(f"Accord entre edge et node homophily: {edge_node_agreement}")
    print(f"Accord entre edge et class homophily: {edge_class_agreement}")
    print(f"Accord entre edge et adjusted homophily: {edge_adjusted_agreement}")
    print(f"Accord entre edge et unbiased homophily: {edge_unbiased_agreement}")
    print(f"Accord entre node et class homophily: {node_class_agreement}")
    print(f"Accord entre node et adjusted homophily: {node_adjusted_agreement}")
    print(f"Accord entre node et unbiased homophily: {node_unbiased_agreement}")
    print(f"Accord entre class et adjusted homophily: {class_adjusted_agreement}")
    print(f"Accord entre class et unbiased homophily: {class_unbiased_agreement}")
    print(f"Accord entre adjusted et unbiased homophily: {adjusted_unbiased_agreement}")

