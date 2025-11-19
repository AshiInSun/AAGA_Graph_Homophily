#!/bin/bash

# ======================================
#   Création + activation de l'environnement virtuel
# ======================================

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo ">>> Création de l'environnement virtuel..."
    python3 -m venv "$VENV_DIR" || { echo "Erreur : impossible de créer le venv"; exit 1; }
fi

echo ">>> Activation de l'environnement virtuel..."
source "$VENV_DIR/bin/activate" || { echo "Erreur : impossible d'activer le venv"; exit 1; }

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
#   Choix de la commande
# ======================================

if [ "$1" == "mean" ]; then
    echo ">>> Calcul des moyennes d'homophilie sur tous les datasets..."
    python3 - <<END
import tests
tests.main()
END

elif [ "$1" == "measure" ]; then
    echo ">>> Calcul des mesures expérimentales sur tous les datasets..."
    python3 - <<END
import tests
tests.main_experimental()
END

else
    echo ">>> Commande invalide. Utilisez :"
    echo "    bash bash.sh mean      # Pour lancer main() : moyennes sur tous les datasets"
    echo "    bash bash.sh measure   # Pour lancer main_experimental() : mesures expérimentales"
fi

echo ">>> Terminé."
