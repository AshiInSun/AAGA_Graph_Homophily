## Revisiting Graph Homophily Measures Semaine II 

Projet dans le cadre de l'UE d'AAGA. Nous appliquons les mesures d'homophilie sur plusieurs graphes et discutons des résultats trouvés à la lumière de l'article étudié 


## Pour commencer

Télécharger le zip et dézippé puis ouvrir un terminal.
### Pré-requis

Requis pour le projet:

- Python 3.8+
- Pip

###  Démarrage

Lancez le script bash en executant la commande ./bash.sh. Donnez les droit d'execution si nécéssaire avec chmod +x run.sh

Le script bash mettra en place un environnement virtuel et téléchargera toutes les dépendances requises, ceci peux prendre environ 3 min lors du premier lancement. Ensuite toutes les mesures d'homophile sur chaque graphe s'afficheront, ceci peut prendre environ 1 min.

Format des resultats d'homophile: 

{
    "edge": valeur,
    "node": valeur,
    "class": valeur,
    "adjusted": valeur,
    "unbiased": valeur
}



## Fabriqué avec

Librairies utilisées:

* networkx
* matplotlib
* numpy
* tqdm
* ogb
* torch
* torch-geometric

## Auteurs
[ T.orandeau]  (https://github.com/AshiInSun)
[O. Crouzet] (https://github.com/OrianeCrouzet)
[F. Ayed] (https://github.com/FatiH2O)

## License

Ce projet est sous licence ``MIT`` 




