import os
import ogb
import networkx as nx

import torch
from torch_geometric.utils import to_networkx, from_networkx
from tqdm import tqdm

from experimental_comparaison import normalize_inplace
from torch_geometric.datasets import TUDataset

from torch.serialization import add_safe_globals
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data import Data
from ogb.graphproppred import PygGraphPropPredDataset

from typing import Optional, List

# Sécurité : autoriser certains objets dans torch.load
safe_classes = [DataTensorAttr, DataEdgeAttr, GlobalStorage]
add_safe_globals(safe_classes)

# Paramètres globaux
max_graphs = 2000
output_dir = "../datasets/OGB_MOLPCBA_GML"
output_dir2 = "../datasets/OGB_CODE2_GML"
output_dir3 = "../datasets/TUD_DD_GML"
output_dir4 = "../datasets/TUD_ZINC_GML"
path_dataset = "../datasets/Mutagenicity_GML"


def retrieve_ogb_molpcba_dataset(nameds):
    print("🔹 Téléchargement / chargement du dataset OGB (ogbg-molpcba)...")
    dataset = PygGraphPropPredDataset(name=nameds, root='data/ogb')
    print("Dataset chargé avec succès !")
    print(f"Nombre total de graphes : {len(dataset)}")
    print(f"Exemple de graphe : {dataset[0]}")
    return dataset

def retrieve_tudataset():
    print("🔹 Téléchargement / chargement du dataset TUD NCI1...")
    dataset = TUDataset(root="data/DD", name="DD")
    print("Dataset chargé avec succès !")
    print(f"Nombre total de graphes : {len(dataset)}")
    return dataset

def retrieve_tudatasetbin():
    print("🔹 Téléchargement / chargement du dataset TUD IMDB-BINARY...")
    dataset = TUDataset(root="data/ZINC_test", name="ZINC_test")
    print("Dataset chargé avec succès !")
    print(f"Nombre total de graphes : {len(dataset)}")
    return dataset

def retrieve_protein_dataset(path: str = '../datasets/Protein_GML',
                             limit: Optional[int] = None,
                             use_tqdm: bool = False) -> List[Data]:
    """
    Charge tous les .gml de `path`, convertit chaque NetworkX Graph en torch_geometric.data.Data,
    crée `data.x` à partir de l'attribut de nœud 'chem' si présent (sinon 0) et retourne une liste.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Le répertoire {path} n'existe pas.")

    files = sorted([f for f in os.listdir(path) if f.lower().endswith('.gml')])
    if limit is not None:
        files = files[:limit]

    dataset: List[Data] = []
    iterator = tqdm(files, desc="Chargement GML") if use_tqdm else files

    for fname in iterator:
        full_path = os.path.join(path, fname)
        try:
            G = nx.read_gml(full_path)

            if G.number_of_nodes() == 0:
                continue

            # indexation contiguë des nœuds
            G = nx.convert_node_labels_to_integers(G, first_label=0)

            # récupérer l'attribut 'chem' par nœud (fallback 0)
            chem_vals = []
            for n in G.nodes():
                v = G.nodes[n].get('chem', 0)
                # gérer cas où la valeur est une chaîne/float dans le GML
                try:
                    v = int(float(v))
                except Exception:
                    v = 0
                chem_vals.append(v)

            # conversion NetworkX -> PyG Data
            data = from_networkx(G)

            # forcer la présence de data.x comme tenseur [num_nodes, 1]
            data.x = torch.tensor([[val] for val in chem_vals], dtype=torch.long)
            data.num_nodes = data.x.size(0)

            dataset.append(data)

        except Exception as e:
            print(f"Erreur lors du chargement de {fname} : {e}")

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
            # python
            # Bloc sécurisé pour extraire les labels et les appliquer comme attribut "chem"
            if hasattr(data, "x") and data.x is not None:
                x = data.x
                # Normaliser en tenseur 1D de labels entiers
                if x.dim() == 1:
                    labels = x.to(torch.long)
                else:
                    labels = x[:, 0].to(torch.long)  # première colonne comme label

                labels = labels.cpu().numpy().tolist()
                num_nodes = G.number_of_nodes()
                # Ajuster si longueur différente : tronquer ou compléter par 0
                if len(labels) < num_nodes:
                    labels = labels + [0] * (num_nodes - len(labels))
                elif len(labels) > num_nodes:
                    labels = labels[:num_nodes]

                node_labels = {n: int(labels[n]) for n in range(num_nodes)}
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

def code2_retrieve():
    dataset = retrieve_ogb_molpcba_dataset("ogbg-code2")
    conversion(dataset, output_dir=output_dir2, max_graphs=max_graphs)
def molpcba_retrieve():
    dataset = retrieve_ogb_molpcba_dataset("ogbg-molpcba")
    conversion(dataset, output_dir=output_dir, max_graphs=max_graphs)
def tud_nci1_retrieve():
    dataset = retrieve_tudataset()
    conversion(dataset, output_dir=output_dir3, max_graphs=max_graphs)
def tud_imdbbinary_retrieve():
    dataset = retrieve_tudatasetbin()
    conversion(dataset, output_dir=output_dir4, max_graphs=max_graphs)


def main_retrieve():
    print("=== DÉMARRAGE DU PIPELINE ===")
    tud_nci1_retrieve()
    print("=== FIN DU PIPELINE ===")


# Pour exécuter :
main_retrieve()
