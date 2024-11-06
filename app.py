# Fichier: app.py

import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import time

# Création d'un Dashboard multi-pages avec Streamlit
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Partie 1 : Préparation des Données", "Partie 2 : Analyse des Données"])

# Charger les données une fois
if page == "Partie 1 : Préparation des Données" or page == "Partie 2 : Analyse des Données":
    if page == "Partie 1 : Préparation des Données":
        data = sns.load_dataset('tips')
    elif page == "Partie 2 : Analyse des Données":
        data = sns.load_dataset('iris')

# 1. Partie 1 : Préparation des Données
def partie_1(data):
    st.title("Partie 1 : Préparation des Données")
    st.write("Aperçu des premières lignes du dataset 'tips':", data.head())

    # Encoder les variables catégorielles : 'sex', 'day', 'smoker', 'time'
    columns_to_encode = ['sex', 'day', 'smoker', 'time']
    label_encoder = LabelEncoder()
    for column in columns_to_encode:
        data[column] = label_encoder.fit_transform(data[column])

    st.subheader("Données après encodage des variables catégorielles")
    st.write(data.head())

    # Sélectionner les données numériques
    df_numeric = data.select_dtypes(include=['int64', 'float64'])
    st.subheader("Données numériques")
    st.write(df_numeric.head())

    # Concaténer les données encodées avec les colonnes numériques sans duplication
    df = pd.concat([data, df_numeric], axis=1).loc[:, ~pd.concat([data, df_numeric], axis=1).columns.duplicated()]
    st.subheader("Données concaténées")
    st.write(df.head())

# 2. Partie 2 : Analyse des Données
def partie_2(data):
    st.title("Partie 2 : Analyse des Données du Dataset Iris")
    st.write("Aperçu des premières lignes du dataset 'iris':", data.head())

    # Encoder la variable 'species'
    label_encoder = LabelEncoder()
    data['species_encoded'] = label_encoder.fit_transform(data['species'])
    st.subheader("Données après encodage de 'species'")
    st.write(data.head())

    # Afficher les histogrammes des features
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    st.subheader("Histogrammes des Features")
    for feature in features:
        fig, ax = plt.subplots()
        ax.hist(data[feature], bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogramme de {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Fréquence')
        st.pyplot(fig)

    # Standardiser les features
    scaler = StandardScaler()
    features_to_standardize = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    data[features_to_standardize] = scaler.fit_transform(data[features_to_standardize])
    st.subheader("Données après standardisation")
    st.write(data.head())

    # Diviser les données et entraîner le modèle
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
    X_train = df_train.iloc[:, 0:-2]  # Toutes les colonnes sauf la cible et la colonne encodée
    y_train = df_train['species_encoded']
    X_test = df_test.iloc[:, 0:-2]
    y_test = df_test['species_encoded']

    clf = LogisticRegression(max_iter=1000)

    # Mesurer le temps d'entraîment
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()

    training_time = end_time - start_time
    st.subheader("Temps d'entraîment du modèle")
    st.write(f"Temps d'entraîment du modèle : {training_time:.4f} secondes")

    # Prédictions et métriques de performance
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.subheader("Performance du modèle")
    st.write(f"Précision : {accuracy:.4f}")
    st.write(f"Score F1 (weighted) : {f1:.4f}")

# Affichage des pages
if page == "Partie 1 : Préparation des Données":
    partie_1(data)
elif page == "Partie 2 : Analyse des Données":
    partie_2(data)