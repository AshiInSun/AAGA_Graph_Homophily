#!/bin/bash
set -e  # Arrête le script si une commande échoue

# ===============================
#  VérIFICATION DES ARGUMENTS
# ===============================
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 /chemin/vers/dossier_GML [graphe_a_visualiser.gml]"
    exit 1
fi

DATASET_PATH="$1"   # Le chemin du dataset est passé en argument
GRAPH_TO_PLOT="${2:-}"   
LABEL_ATTR="chem"   # Nom de l'attribut de classe pour l'homophilie

echo "=============================="
echo "  Lancement des expériences d’homophilie "
echo "=============================="
echo " Dataset : $DATASET_PATH"
if [ -n "$GRAPH_TO_PLOT" ]; then
    echo "Graphe à visualiser : $GRAPH_TO_PLOT"
fi
echo " Label : $LABEL_ATTR"

# Vérifie que le dossier existe et n'est pas vide
if [ ! -d "$DATASET_PATH" ] || [ -z "$(ls -A $DATASET_PATH)" ]; then
    echo " Erreur : le dossier $DATASET_PATH est introuvable ou vide."
    exit 1
fi

# ===============================
#  Création et activation d'un environnement virtuel Python
# ===============================
if [ ! -d "venv" ]; then
    echo " Création d'un environnement virtuel Python..."
    python3 -m venv venv
fi

echo "Activation de l'environnement virtuel..."
source venv/bin/activate

# ===============================
#  Installation des dépendances nécessaires
# ===============================
echo "Installation des packages Python requis..."
pip install --upgrade pip
pip install networkx matplotlib numpy torch torch-geometric tqdm ogb

# ===============================
# Exécution du script principal
# ===============================
echo " Exécution du script principal "
if [ -n "$GRAPH_TO_PLOT" ]; then
    python test.py --dataset "$DATASET_PATH" --plot_one_graph "$GRAPH_TO_PLOT"
else
    python test.py --dataset "$DATASET_PATH"
fi


echo "=============================="
echo "Toutes les simulations ont été exécutées avec succès !"
echo "=============================="

# Désactivation de l'environnement virtuel
deactivate
