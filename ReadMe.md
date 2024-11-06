# Projet Streamlit - Analyse et Préparation des Données

Ce projet est une application web créée avec Streamlit qui permet d'explorer, préparer, et analyser deux jeux de données : « Tips » et « Iris ». Le projet est divisé en deux parties : préparation des données et analyse des données, chacune présentée sur une page différente de l'application.

## Structure du Projet

- **Partie 1 : Préparation des Données**
  - Encodage des variables catégorielles (« sex », « day », « smoker », « time »).
  - Affichage des données numériques et concaténation avec les données encodées.
  - Entraînement d'un modèle de régression linéaire et évaluation du score R².

- **Partie 2 : Analyse des Données**
  - Encodage de la variable « species » du dataset Iris.
  - Affichage des histogrammes des caractéristiques (« sepal_length », « sepal_width », « petal_length », « petal_width »).
  - Standardisation des caractéristiques avec `StandardScaler`.
  - Entraînement d'un modèle de régression logistique et évaluation des performances (évaluation avec la précision et le score F1).

## Fonctionnalités de l'Application

- **Interface Utilisateur Moderne** : L'application offre une interface conviviale avec navigation multi-pages, images d'introduction, et des cartes de métriques pour présenter les résultats de façon claire.
- **Progress Bars et Animations** : Ajout de spinners pour indiquer les processus en cours (ex. : encodage des données, entraînement des modèles).
- **Visualisation Interactives** : Graphiques interactifs pour une meilleure analyse visuelle des données.

## Installation

Pour exécuter ce projet, vous devez avoir Python 3 installé ainsi que les bibliothèques requises. Voici les étapes d'installation :

1. Clonez ce dépôt :
   ```sh
   git clone https://github.com/Mathieu-Soussignan/data_cleaning.git
   cd data_cleaning
   ```
2. Installez les dépendances :
   ```sh
   pip install -r requirements.txt
   ```
3. Lancez l'application Streamlit :
   ```sh
   streamlit run app.py
   ```

## Fichiers Importants

- **app.py** : Script principal contenant le code de l'application Streamlit.
- **assets/** : Dossier contenant des images utilisées dans l'application.
- **requirements.txt** : Liste des bibliothèques Python requises pour exécuter l'application (ex. : pandas, streamlit, scikit-learn, plotly).

## Utilisation

- Rendez-vous sur l'URL locale générée par Streamlit (en général `http://localhost:8501`) pour interagir avec l'application.
- Utilisez la barre latérale pour naviguer entre les différentes pages :
  - **Partie 1 : Préparation des Données** permet de préparer et encoder les données, puis de visualiser le modèle de régression.
  - **Partie 2 : Analyse des Données** offre des graphiques interactifs et une analyse des données Iris avec un modèle de régression logistique.

## Pré-requis

- **Python 3.x**
- Bibliothèques : `streamlit`, `pandas`, `seaborn`, `scikit-learn`, `matplotlib`, `plotly`

## Fonctionnalités Améliorées

- **Visualisations interractives avec Plotly** : Pour rendre les graphiques plus interactifs.
- **Métriques sous forme de cartes** : Les résultats comme la Précision et le Score F1 sont présentés sous forme de cartes pour une meilleure lisibilité.
- **Expérience utilisateur immersive** : Utilisation d'emojis, de messages de succès et de spinners pour rendre l'application plus engageante.

## Auteurs

- [Mathieu Soussignan](https://www.mathieu-soussignan.com) - Développeur IA / Data.

## Licence

Ce projet est sous licence MIT.

