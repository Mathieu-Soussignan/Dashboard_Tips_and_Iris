import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import time

# Création d'un Dashboard multi-pages avec Streamlit
st.set_page_config(page_title="Machine Learning Dashboard", layout="wide", initial_sidebar_state="expanded")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["🏠 Partie 1 : Préparation des Données", "📊 Partie 2 : Analyse des Données"])

# Image de couverture
st.image("./assets/machine_learning.jpg", caption="L'art de l'analyse de données !", use_column_width=True)

# Charger les données une fois
if page == "🏠 Partie 1 : Préparation des Données" or page == "📊 Partie 2 : Analyse des Données":
    if page == "🏠 Partie 1 : Préparation des Données":
        data = sns.load_dataset('tips')
    elif page == "📊 Partie 2 : Analyse des Données":
        data = sns.load_dataset('iris')

# Partie 1 : Préparation des Données
def partie_1(data):
    st.title("🏠 Partie 1 : Préparation des Données")

    # Afficher les premières lignes
    with st.expander("👀 Aperçu des premières lignes du dataset 'tips'"):
        st.dataframe(data.head())

    # Progress bar pour l'encodage des données
    with st.spinner("Encodage des variables catégorielles..."):
        columns_to_encode = ['sex', 'day', 'smoker', 'time']
        label_encoder = LabelEncoder()
        for column in columns_to_encode:
            data[column] = label_encoder.fit_transform(data[column])
        st.success("Encodage terminé!")

    # Visualisation des données après encodage
    st.subheader("📊 Données après encodage des variables catégorielles")
    st.write(data.head())

    # Affichage des données numériques
    df_numeric = data.select_dtypes(include=['int64', 'float64'])
    st.subheader("🔢 Données numériques")
    st.write(df_numeric.head())

    # Concaténer les données encodées avec les colonnes numériques
    df = pd.concat([data, df_numeric], axis=1).loc[:, ~pd.concat([data, df_numeric], axis=1).columns.duplicated()]
    st.subheader("🗂️ Données concaténées")
    st.write(df.head())

    # Entraînement du modèle de régression linéaire avec animation
    st.subheader("📈 Entraînement du modèle de régression linéaire")
    X = df.drop(columns=['tip'])  # Prédicteurs
    y = df['tip']  # Variable cible

    model = LinearRegression()
    start_time = time.time()
    with st.spinner('Entraînement du modèle...'):
        model.fit(X, y)
    end_time = time.time()

    # Mesurer le temps d'entraînement
    training_time = end_time - start_time
    st.success(f"Temps d'entraînement du modèle : {training_time:.4f} secondes")

    # Évaluation du modèle
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    st.metric(label="📊 Score R² du modèle", value=f"{r2:.4f}")

# Partie 2 : Analyse des Données
def partie_2(data):
    st.title("📊 Partie 2 : Analyse des Données du Dataset Iris")

    # Aperçu des premières lignes
    with st.expander("👀 Aperçu des premières lignes du dataset 'iris'"):
        st.dataframe(data.head())

    # Encodage de la variable 'species'
    with st.spinner("Encodage de la variable 'species'..."):
        label_encoder = LabelEncoder()
        data['species_encoded'] = label_encoder.fit_transform(data['species'])
    st.success("Encodage terminé!")

    # Visualisation des histogrammes des features
    st.subheader("📊 Histogrammes des Features")
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for feature in features:
        fig = px.histogram(data, x=feature, nbins=20, title=f"Histogramme de {feature}", template="plotly_dark")
        st.plotly_chart(fig)

    # Standardiser les features
    st.subheader("🔄 Standardisation des Features")
    scaler = StandardScaler()
    features_to_standardize = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    data[features_to_standardize] = scaler.fit_transform(data[features_to_standardize])
    st.write(data.head())

    # Entraînement du modèle de régression logistique
    st.subheader("📈 Entraînement du modèle de Régression Logistique")
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
    X_train = df_train.iloc[:, 0:-2]  # Toutes les colonnes sauf la cible et la colonne encodée
    y_train = df_train['species_encoded']
    X_test = df_test.iloc[:, 0:-2]
    y_test = df_test['species_encoded']

    clf = LogisticRegression(max_iter=1000)
    start_time = time.time()
    with st.spinner('Entraînement du modèle...'):
        clf.fit(X_train, y_train)
    end_time = time.time()

    # Temps d'entraînement
    training_time = end_time - start_time
    st.success(f"Temps d'entraînement du modèle : {training_time:.4f} secondes")

    # Prédictions et métriques de performance
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Afficher les métriques sous forme de cartes
    st.subheader("📊 Performance du modèle")
    col1, col2 = st.columns(2)
    col1.metric(label="🎯 Précision", value=f"{accuracy:.4f}")
    col2.metric(label="🎯 Score F1 (weighted)", value=f"{f1:.4f}")

# Affichage des pages
if page == "🏠 Partie 1 : Préparation des Données":
    partie_1(data)
elif page == "📊 Partie 2 : Analyse des Données":
    partie_2(data)