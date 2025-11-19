#!/bin/bash

# ======================================
#   Création + activation de l'environnement virtuel
# ======================================

VENV_DIR="venv"

echo ">>> Vérification de l'environnement virtuel..."
if [ ! -d "$VENV_DIR" ]; then
    echo ">>> Création de l'environnement virtuel..."
    python3 -m venv $VENV_DIR
fi

echo ">>> Activation de l'environnement virtuel..."
source $VENV_DIR/bin/activate

echo ">>> Installation des dépendances..."
pip install --upgrade pip
pip install networkx matplotlib numpy tqdm ogb torch torch-geometric

# ======================================
#   Vérification du dossier datasets
# ======================================

if [ ! -d "datasets" ]; then
    echo ">>> Dossier 'datasets' introuvable. Veuillez le placer au même niveau que ce script."
    exit 1
fi

# ======================================
#   Exécution du script Python principal
# ======================================

echo ">>> Lancement des calculs sur tous les datasets..."
python3 tests.py

echo ">>> Terminé."
