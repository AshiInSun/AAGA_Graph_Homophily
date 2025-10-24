import os
import xml.etree.ElementTree as ET
import networkx as nx
from convert_gxl_to_gml import read_gxl

# --- Vérification ---
def verify_graph_conversion(gxl_path, gml_path):
    G_gxl = read_gxl(gxl_path)
    G_gml = nx.read_gml(gml_path)

    print(f"\n🔹 Vérification de {os.path.basename(gxl_path)} → {os.path.basename(gml_path)}")

    # Vérifier le nombre de nœuds et arêtes
    print("Nombre de nœuds :", G_gxl.number_of_nodes(), "→", G_gml.number_of_nodes())
    print("Nombre d’arêtes :", G_gxl.number_of_edges(), "→", G_gml.number_of_edges())

    # Vérifier les nœuds et attributs
    for n in G_gxl.nodes():
        if n not in G_gml.nodes():
            print("Nœud manquant dans GML :", n)
        else:
            if G_gxl.nodes[n] != G_gml.nodes[n]:
                print("Attributs différents pour le nœud", n)
                print("GXL :", G_gxl.nodes[n])
                print("GML :", G_gml.nodes[n])

    # Vérifier les arêtes et attributs
    for u, v in G_gxl.edges():
        if not G_gml.has_edge(u, v):
            print("Arête manquante dans GML :", u, v)
        else:
            if G_gxl.edges[u, v] != G_gml.edges[u, v]:
                print("Attributs différents pour l’arête", u, v)
                print("GXL :", G_gxl.edges[u, v])
                print("GML :", G_gml.edges[u, v])

    print("✅ Vérification terminée.")


# --- Partie principale ---
if __name__ == "__main__":
    gxl_file = "datasets/Mutagenicity/data/molecule_1.gxl"
    gml_file = "datasets/Mutagenicity_GML/molecule_1.gml"
    verify_graph_conversion(gxl_file, gml_file)