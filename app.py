import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Configuration de la page
st.set_page_config(page_title="Analyse Client", layout="wide")
st.title("Application d'analyse e-commerce")

# Sidebar pour charger les données
st.sidebar.header("Chargement des données")
uploaded_file = st.sidebar.file_uploader("Uploader un fichier CSV ou Excel", type=["csv", "xlsx"])

# Fonction pour charger les données de manière sécurisée
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file, encoding='utf-8', header=0)
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding='latin1', header=0)

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")
    return df

# Statistiques descriptives
def show_descriptive_stats(df):
    st.subheader("Statistiques descriptives")
    st.write(df.describe())
    st.write("**Distribution des variables numériques :**")
    for col in df.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution de {col}')
        st.pyplot(fig)

# Analyse FP-Growth
def apply_fp_growth(df):
    st.subheader("Analyse par FP-Growth")

    required_columns = {'InvoiceNo', 'Description', 'Quantity'}
    if not required_columns.issubset(df.columns):
        st.error(f"Colonnes requises manquantes : {required_columns - set(df.columns)}")
        return

    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    try:
        freq_items = fpgrowth(basket, min_support=0.02, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=1)
        st.write("Règles générées :")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    except Exception as e:
        st.error(f"Erreur pendant le calcul FP-Growth : {e}")

# K-means clustering
def apply_kmeans(df):
    st.subheader("Segmentation par K-means")
    k = st.slider("Choisissez le nombre de clusters", 2, 10, 4)
    features = df.select_dtypes(include='number')

    if features.shape[1] < 2:
        st.error("Au moins deux colonnes numériques sont requises pour K-means.")
        return

    X_scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(X_scaled)
    st.write("Extrait des clusters :")
    st.dataframe(df.head())

    fig, ax = plt.subplots()
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['Cluster'], palette='Set2')
    ax.set_xlabel(features.columns[0])
    ax.set_ylabel(features.columns[1])
    st.pyplot(fig)

# RFM analysis
def apply_rfm(df):
    st.subheader("Segmentation RFM")

    required_columns = {'Recence', 'Frequence', 'Montant'}
    if not required_columns.issubset(df.columns):
        st.error(f"Colonnes RFM manquantes : {required_columns - set(df.columns)}")
        return

    rfm = df[['Recence', 'Frequence', 'Montant']].copy()
    rfm['R'] = pd.qcut(rfm['Recence'], 4, labels=[4,3,2,1])
    rfm['F'] = pd.qcut(rfm['Frequence'], 4, labels=[1,2,3,4])
    rfm['M'] = pd.qcut(rfm['Montant'], 4, labels=[1,2,3,4])
    rfm['RFM_Score'] = rfm[['R','F','M']].astype(int).sum(axis=1)

    st.dataframe(rfm.head())

    fig, ax = plt.subplots()
    sns.histplot(rfm['RFM_Score'], bins=10, kde=True)
    ax.set_title("Distribution des scores RFM")
    st.pyplot(fig)

# Main logic
if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Aperçu des données :")
    st.dataframe(df.head())
    st.write("Colonnes détectées :", df.columns.tolist())

    model_choice = st.sidebar.selectbox("Choisir une analyse", [
        "Statistiques descriptives", "FP-Growth", "K-means", "RFM"
    ])

    if model_choice == "Statistiques descriptives":
        show_descriptive_stats(df)
    elif model_choice == "FP-Growth":
        apply_fp_growth(df)
    elif model_choice == "K-means":
        apply_kmeans(df)
    elif model_choice == "RFM":
        apply_rfm(df)
else:
    st.info("Veuillez charger un fichier pour commencer.")
