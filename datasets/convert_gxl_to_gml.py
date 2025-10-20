import os
import xml.etree.ElementTree as ET
import networkx as nx

# Convertisseur écrit pat ChatGPT-4
# Nous avions besoin de convertir des fichiers GXL en GML pour les utiliser avec NetworkX.

def read_gxl(path):
    """Lit un fichier GXL et retourne un graphe NetworkX avec les attributs."""
    tree = ET.parse(path)
    root = tree.getroot()
    G = nx.Graph()

    # Lecture des nœuds
    for node in root.findall('.//node'):
        node_id = node.get('id')
        attrs = {}
        for attr in node.findall('attr'):
            name = attr.get('name')
            val_int = attr.find('int')
            val_float = attr.find('float')
            val_str = attr.find('string')
            if val_int is not None:
                try:
                    attrs[name] = int(val_int.text)
                except ValueError:
                    attrs[name] = val_int.text  # fallback si ce n'est pas convertible
            elif val_float is not None:
                attrs[name] = float(val_float.text)
            elif val_str is not None:
                attrs[name] = val_str.text
        G.add_node(node_id, **attrs)

    # Lecture des arêtes
    for edge in root.findall('.//edge'):
        src = edge.get('from')
        tgt = edge.get('to')
        attrs = {}
        for attr in edge.findall('attr'):
            name = attr.get('name')
            val_int = attr.find('int')
            val_float = attr.find('float')
            val_str = attr.find('string')
            if val_int is not None:
                try:
                    attrs[name] = int(val_int.text)
                except ValueError:
                    attrs[name] = val_int.text
            elif val_float is not None:
                attrs[name] = float(val_float.text)
            elif val_str is not None:
                attrs[name] = val_str.text
        G.add_edge(src, tgt, **attrs)

    return G



# --- Partie principale ---
if __name__ == "__main__":
    input_folder = "Protein/data"
    output_folder = "Protein_GML"
    os.makedirs(output_folder, exist_ok=True)

    # Trouver le premier fichier .gxl
    gxl_files = [f for f in os.listdir(input_folder) if f.endswith(".gxl")]
    if not gxl_files:
        print("❌ Aucun fichier .gxl trouvé dans", input_folder)
        exit()

    for gxl_file in gxl_files:
        gxl_path = os.path.join(input_folder, gxl_file)
        gml_path = os.path.join(output_folder, gxl_file.replace(".gxl", ".gml"))

        print(f"➡ Conversion de {gxl_file} → {os.path.basename(gml_path)}")
        G = read_gxl(gxl_path)
        nx.write_gml(G, gml_path)

    print("\n✅ Toutes les conversions sont terminées !")
    print(f"📄 Les fichiers GML sont dans : {output_folder}")
