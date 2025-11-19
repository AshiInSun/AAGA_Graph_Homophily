## Revisiting Graph Homophily Measures Semaine II 

Projet dans le cadre de l'UE d'AAGA. Nous appliquons différentes mesures d'homophilie sur plusieurs graphes et discutons des résultats trouvés à la lumière de l'article étudié 


## Pour commencer

Récuperer la branch main et l'ouvrir dans un  terminal, préférablement sur linux.
### Pré-requis

Requis pour le projet:
- Environnement linux (possible sur wsl mais prend beaucoup de temps)
- Python 3.8+
- Pip

###  Démarrage

Lancez le script bash en executant la commande ```./bash.sh``` avec les une des options suivante: 
* mean pour les moyennes des mesures d'homophilie sur tous les graphes (./bash.sh mean)
* measure pour les mesures d'accord d'homophilie (./bash.sh measure) 

Donnez les droits d'execution si nécéssaire avec chmod +x bash.sh

Le script bash mettra en place un environnement virtuel et téléchargera toutes les dépendances requises, ceci peux prendre du temps lors du premier lancement. 


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
* [T.orandeau](https://github.com/AshiInSun)
* [O. Crouzet](https://github.com/OrianeCrouzet)
* [F. Ayed](https://github.com/FatiH2O)

## License

Ce projet est sous licence ``MIT`` 




