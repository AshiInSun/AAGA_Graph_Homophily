import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import graph_homophily_measures
import experimental_comparaison
import argparse
import os

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


def testing_one_graph(graph_file, class_attr="chem"):
    """
    Test sur un seul graphe pour visualisation
    """
    G = nx.read_gml(graph_file)
    experimental_comparaison.normalize_inplace(G)
    g_edge_homophily = graph_homophily_measures.edge_homophily(G, class_attr=class_attr)
    g_node_homophily = graph_homophily_measures.node_homophily(G, class_attr=class_attr)
    g_class_homophily = graph_homophily_measures.class_homophily(G, class_attr=class_attr)
    g_adjusted_homophily = graph_homophily_measures.adjusted_homophily(G, class_attr=class_attr)

    print(f"\nGraphe edge homophily : {g_edge_homophily}")
    print(f"Graphe node homophily : {g_node_homophily}")
    print(f"Graphe class homophily : {g_class_homophily}")
    print(f"Graphe adjusted homophily : {g_adjusted_homophily}")

    plot_graph(G, class_attr=class_attr)


def main():
    parser = argparse.ArgumentParser(description="Lancer les expériences d’homophilie sur un dossier de graphes GML")
    parser.add_argument("--dataset", type=str, required=True, help="Chemin vers le dossier contenant les fichiers .gml")
    parser.add_argument("--plot_one_graph", type=str, default=None, help="Optionnel : chemin d'un graphe GML à visualiser")
    args = parser.parse_args()

    dataset_path = args.dataset

    if not os.path.isdir(dataset_path):
        print(f" Le dossier {dataset_path} n'existe pas.")
        return

    print(f" Lancement des expériences sur le dataset : {dataset_path}\n")
    experimental_comparaison.experimental_comparaison(dataset_path, label_G="chem")

    # Optionnel : visualiser un graphe spécifique
    if args.plot_one_graph:
        if os.path.isfile(args.plot_one_graph):
            print(f"\n📊 Visualisation du graphe : {args.plot_one_graph}")
            testing_one_graph(args.plot_one_graph, class_attr="chem")
        else:
            print(f" Fichier {args.plot_one_graph} introuvable.")


if __name__ == "__main__":
    main()
