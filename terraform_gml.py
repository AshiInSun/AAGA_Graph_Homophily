# python
import shutil
from pathlib import Path
import networkx as nx

INPUT_DIR = Path('datasets/Protein_GML')
BACKUP_DIR = Path('datasets/Protein_GML_conn_bak')
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    for i in range(1, 1000):
        candidate = dest.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"No unique name found for {dest}")

for path in INPUT_DIR.glob('*.gml'):
    try:
        G = nx.read_gml(path)

        # choisir composants: weak pour DiGraph, connected pour Graph
        if G.is_directed():
            components = list(nx.weakly_connected_components(G))
        else:
            components = list(nx.connected_components(G))

        if len(components) > 1:
            largest = max(components, key=len)
            H = G.subgraph(largest).copy()
            action = f"kept largest ({len(largest)} nodes) of {len(components)} components"
        else:
            H = G
            action = "already single component"

        # backup de l'original dans BACKUP_DIR (toujours)
        bak_path = BACKUP_DIR / path.name
        bak_path = unique_dest(bak_path)
        shutil.copy2(path, bak_path)

        # écriture du (potentiellement modifié) graphe dans le fichier original
        nx.write_gml(H, path)

        print(f"Processed {path.name}: {action}, backup -> {bak_path.name}")

    except Exception as e:
        print(f"Error processing {path.name}: {e}")
