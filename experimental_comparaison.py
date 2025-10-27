import os

import networkx as nx

import test
from test import normalize_inplace
path_dataset = "datasets/Mutagenicity_GML"
def experimental_comparaison(path, label_G):
    chemin_dossier = path
    max_files = 2000  # Nombre maximum de fichiers à traiter
    noms_fichiers = [f for f in os.listdir(chemin_dossier) if os.path.isfile(os.path.join(chemin_dossier, f))]

    edge_node_agreement = 0
    edge_class_agreement = 0
    edge_adjusted_agreement = 0
    node_class_agreement = 0
    node_adjusted_agreement = 0
    class_adjusted_agreement = 0

    # Boucle pour ouvrir deux fichiers à chaque itération
    for i in range(0, min(len(noms_fichiers), max_files), 2):
        if i % 100 == 0:
            print(f"Bip boup on est au fichier {i}")
        if i + 1 < len(noms_fichiers):  # Vérifie qu'il y a au moins deux fichiers restants
            fichier1 = os.path.join(chemin_dossier, noms_fichiers[i])
            fichier2 = os.path.join(chemin_dossier, noms_fichiers[i + 1])
            try:
                G1 = nx.read_gml(fichier1)
                G2 = nx.read_gml(fichier2)
                normalize_inplace(G1)
                normalize_inplace(G2)

                g1_edge_homophily = test.edge_homophily(G1, class_attr=label_G)
                g1_node_homophily = test.node_homophily(G1, class_attr=label_G)
                g1_class_homophily = test.class_homophily(G1, class_attr=label_G)
                g1_adjusted_homophily = test.adjusted_homophily(G1, class_attr=label_G)
                g2_edge_homophily = test.edge_homophily(G2, class_attr=label_G)
                g2_node_homophily = test.node_homophily(G2, class_attr=label_G)
                g2_class_homophily = test.class_homophily(G2, class_attr=label_G)
                g2_adjusted_homophily = test.adjusted_homophily(G2, class_attr=label_G)

                #On récupère maintenant 4 booleen, un pour chaque mesure, qui indique si la mesure est plus grande dans G1 que dans G2

                edge_result = g1_edge_homophily > g2_edge_homophily
                node_result = g1_node_homophily > g2_node_homophily
                class_result = g1_class_homophily > g2_class_homophily
                adjusted_result = g1_adjusted_homophily > g2_adjusted_homophily

                #Puis, on compare ces resultats entre les mesures

                edge_node_result = edge_result == node_result
                edge_class_result = edge_result == class_result
                edge_adjusted_result = edge_result == adjusted_result
                node_class_result = node_result == class_result
                node_adjusted_result = node_result == adjusted_result
                class_adjusted_result = class_result == adjusted_result

                #Et on cumule les resultats
                edge_node_agreement += edge_node_result
                edge_class_agreement += edge_class_result
                edge_adjusted_agreement += edge_adjusted_result
                node_class_agreement += node_class_result
                node_adjusted_agreement += node_adjusted_result
                class_adjusted_agreement += class_adjusted_result
            except Exception as e:
                print(f"Erreur lors du traitement des fichiers {fichier1} et {fichier2} : {e}")
    #Puis, on normalise les resultats par le nombre de comparaisons effectuées
    num_comparisons = min(max_files, len(noms_fichiers)) // 2
    edge_node_agreement /= num_comparisons
    edge_class_agreement /= num_comparisons
    edge_adjusted_agreement /= num_comparisons
    node_class_agreement /= num_comparisons
    node_adjusted_agreement /= num_comparisons
    class_adjusted_agreement /= num_comparisons

    print(f"Accord entre edge et node homophily: {edge_node_agreement}")
    print(f"Accord entre edge et class homophily: {edge_class_agreement}")
    print(f"Accord entre edge et adjusted homophily: {edge_adjusted_agreement}")
    print(f"Accord entre node et class homophily: {node_class_agreement}")
    print(f"Accord entre node et adjusted homophily: {node_adjusted_agreement}")
    print(f"Accord entre class et adjusted homophily: {class_adjusted_agreement}")
experimental_comparaison(path_dataset, label_G="chem")