import os
import ogb
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
from tqdm import tqdm

import tests
from tests import normalize_inplace

from torch.serialization import add_safe_globals
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from ogb.graphproppred import PygGraphPropPredDataset

# Sécurité : autoriser certains objets dans torch.load
safe_classes = [DataTensorAttr, DataEdgeAttr, GlobalStorage]
add_safe_globals(safe_classes)

# Paramètres globaux
max_graphs = 2000
output_dir = "../datasets/OGB_CODE2_GML"
path_dataset = "../datasets/Mutagenicity_GML"


def retrieve_ogb_molpcba_dataset():
    print("Téléchargement / chargement du dataset OGB (ogbg-code2)...")
    dataset = PygGraphPropPredDataset(name='ogbg-code2', root='data/ogb')
    print("Dataset chargé avec succès !")
    print(f"Nombre total de graphes : {len(dataset)}")
    print(f"Exemple de graphe : {dataset[0]}")
    return dataset


def conversion(dataset, output_dir="datasets/OGB_CODE2_GML", max_graphs=2000):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nDébut de la conversion des {max_graphs} premiers graphes du dataset OGB en .gml...")
    print(f"Les fichiers seront sauvegardés dans : {output_dir}")

    num_converted = 0
    num_failed = 0

    for i in tqdm(range(min(max_graphs, len(dataset))), desc="Conversion en cours"):
        try:
            data = dataset[i]
            G = to_networkx(data, to_undirected=True)

            # Vérification basique
            if G.number_of_nodes() == 0:
                print(f"Graphe {i} vide, ignoré.")
                num_failed += 1
                continue

            # Ajouter un label de nœud
            if hasattr(data, "x") and data.x is not None:
                node_labels = {n: int(data.x[n][0].item()) for n in range(data.num_nodes)}
                nx.set_node_attributes(G, node_labels, name="chem")
            else:
                nx.set_node_attributes(G, {n: 0 for n in G.nodes()}, name="chem")

            # Sauvegarde du graphe
            nx.write_gml(G, os.path.join(output_dir, f"graph_{i}.gml"))
            num_converted += 1

            # Petits checkpoints de progression
            if i % 100 == 0 and i > 0:
                print(f"{i} graphes traités, {num_converted} sauvegardés avec succès...")

        except Exception as e:
            print(f"Erreur lors du traitement du graphe {i} : {e}")
            num_failed += 1

    print(f"\nConversion terminée !")
    print(f"{num_converted} graphes sauvegardés avec succès dans : {output_dir}")
    print(f"{num_failed} graphes n'ont pas pu être convertis.\n")


def main_retrieve():
    print("=== DÉMARRAGE DU PIPELINE ===")
    dataset = retrieve_ogb_molpcba_dataset()
    conversion(dataset, output_dir=output_dir, max_graphs=max_graphs)
    print("=== FIN DU PIPELINE ===")


# Pour exécuter :
main_retrieve()
