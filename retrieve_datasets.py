import os
import ogb
import networkx as nx
from torch_geometric.utils import to_networkx

import test
from test import normalize_inplace

path_dataset = "datasets/Mutagenicity_GML"

import torch
from torch.serialization import add_safe_globals
from torch_geometric.data.data import DataEdgeAttr
from torch_geometric.data.data import DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm
from ogb.graphproppred import PygGraphPropPredDataset

safe_classes = []
safe_classes += [DataTensorAttr, DataEdgeAttr]
safe_classes.append(GlobalStorage)
add_safe_globals(safe_classes)

max_graphs = 2000
output_dir = "datasets/OGB_MOLPCBA_GML"


def retrieve_ogb_molpcba_dataset():

    dataset = PygGraphPropPredDataset(name='ogbg-molpcba', root='data/ogb')
    print("Nombre de graphes :", len(dataset))
    print("Exemple:", dataset[0])   # (Data, label)
    return dataset

def conversion(dataset, output_dir="datasets/OGB_MOLPCBA_GML", max_graphs=2000):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Conversion des {max_graphs} premiers graphes du dataset OGB en .gml...")

    for i in tqdm(range(min(max_graphs, len(dataset)))):
        data = dataset[i]
        G = to_networkx(data, to_undirected=True)

        # Exemple : on utilise la première feature du vecteur x comme "type d’atome"
        if hasattr(data, "x"):
            node_labels = {n: int(data.x[n][0].item()) for n in range(data.num_nodes)}
            nx.set_node_attributes(G, node_labels, name="chem")
        else:
            # Fallback si pas de x -> un seul label pour tout le graphe
            nx.set_node_attributes(G, {n: 0 for n in G.nodes()}, name="chem")

        # On ignore data.y (trop complexe pour un label de nœud)
        nx.write_gml(G, os.path.join(output_dir, f"graph_{i}.gml"))

    print(f"✅ {max_graphs} graphes sauvegardés dans : {output_dir}")

def main_retrieve():
    dataset = retrieve_ogb_molpcba_dataset()
    conversion(dataset, output_dir=output_dir, max_graphs=max_graphs)
