import os
import ogb
import networkx as nx
from torch_geometric.utils import to_networkx
from tqdm import trange

import graph_homophily_measures

#path_dataset = "datasets/OGB_CODE2_GML"

def normalize_inplace(G):
    """
    Supprime tous les nœuds de degré 0 en modifiant `G` directement.
    """
    nodes_to_remove = [n for n, d in G.degree() if d == 0]
    if nodes_to_remove:
        G.remove_nodes_from(nodes_to_remove)

def all_homophilia_onaverage_all_datasets(label_G, root_path="datasets"):
    """
    Pour chaque sous-dossier dans `datasets/`, calcule les moyennes d'homophilie
    et retourne un dictionnaire des résultats.
    """

    results = {}

    # Liste des sous-dossiers dans datasets/
    datasets = [
        d for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    ]

    if not datasets:
        print("Aucun dataset trouvé dans le dossier", root_path)
        return None

    for dataset_name in datasets:
        print(f"\n=== Traitement du dataset : {dataset_name} ===")
        dataset_path = os.path.join(root_path, dataset_name)
        dataset_results = all_homophilia_onaverage_single(dataset_path, label_G)
        results[dataset_name] = dataset_results

    print("\n\n=== Résultats finaux (Tableau 5) ===", flush=True)
    for name, res in results.items():
        print(f"\nDataset : {name}", flush=True)
        print(res, flush=True)

    return results

def all_homophilia_onaverage_single(path, label_G):
    noms_fichiers = [
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    ]

    sum_edge = 0
    sum_node = 0
    sum_class = 0
    sum_adjusted = 0
    sum_unbiased = 0
    count = 0

    for f in trange(len(noms_fichiers), desc="Calcul des homophilies", unit="graphe"):
        fichier = os.path.join(path, noms_fichiers[f])

        try:
            G = nx.read_gml(fichier)
            normalize_inplace(G)

            h_edge = graph_homophily_measures.edge_homophily(G, class_attr=label_G)
            h_node = graph_homophily_measures.node_homophily(G, class_attr=label_G)
            h_class = graph_homophily_measures.class_homophily(G, class_attr=label_G)
            h_adjusted = graph_homophily_measures.adjusted_homophily(G, class_attr=label_G)
            h_unbiased = graph_homophily_measures.unbiased_homophily(G, class_attr=label_G)

            sum_edge += h_edge
            sum_node += h_node
            sum_class += h_class
            sum_adjusted += h_adjusted
            sum_unbiased += h_unbiased

            count += 1

        except Exception as e:
            print(f"Erreur dans {fichier} : {e}")

    if count == 0:
        return None

    return {
        "edge": float(sum_edge / count),
        "node": float(sum_node / count),
        "class": float(sum_class / count),
        "adjusted": float(sum_adjusted / count),
        "unbiased": float(sum_unbiased / count)
    }


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

