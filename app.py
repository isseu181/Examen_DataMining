import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import fpgrowth, association_rules

st.set_page_config(page_title="Analyse Client", layout="wide")
st.title("Application d'analyse e-commerce")

# Sidebar
st.sidebar.header("Chargement des données")
uploaded_file = st.sidebar.file_uploader("Uploader un fichier CSV ou Excel", type=["csv", "xlsx"])

# Chargement des données avec gestion des encodages
@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file, encoding="ISO-8859-1")
        except UnicodeDecodeError:
            return pd.read_csv(file, encoding="cp1252")

# Statistiques descriptives
def show_descriptive_stats(df):
    st.subheader("Statistiques descriptives")
    st.write(df.describe())
    st.write("**Distribution des variables numériques :**")
    for col in df.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

# Analyse FP-Growth
def apply_fp_growth(df):
    st.subheader("Analyse par FP-Growth")
    if {'InvoiceNo', 'Description', 'Quantity'}.issubset(df.columns):
        basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        freq_items = fpgrowth(basket, min_support=0.02, use_colnames=True)
        if freq_items.empty:
            st.warning("Aucun item fréquent trouvé. Essayez un seuil plus bas.")
            return
        rules = association_rules(freq_items, metric="lift", min_threshold=1)
        st.write("Règles générées :")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        st.error("Colonnes requises manquantes : 'InvoiceNo', 'Description', 'Quantity'")

# Segmentation K-means
def apply_kmeans(df):
    st.subheader("Segmentation par K-means")
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        st.error("Il faut au moins 2 variables numériques pour appliquer K-means.")
        return

    k = st.slider("Choisissez le nombre de clusters", 2, 10, 4)
    X_scaled = StandardScaler().fit_transform(numeric_df)
    model = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = model.fit_predict(X_scaled)
    st.write("Extrait des clusters :")
    st.dataframe(df.head())

    fig, ax = plt.subplots()
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['Cluster'], palette='Set2')
    st.pyplot(fig)

# Segmentation RFM
def apply_rfm(df):
    st.subheader("Segmentation RFM")
    rfm_columns = ['Recence', 'Frequence', 'Montant']
    if all(col in df.columns for col in rfm_columns):
        rfm = df[rfm_columns].copy()
        rfm['R'] = pd.qcut(rfm['Recence'], 4, labels=[4,3,2,1])
        rfm['F'] = pd.qcut(rfm['Frequence'], 4, labels=[1,2,3,4])
        rfm['M'] = pd.qcut(rfm['Montant'], 4, labels=[1,2,3,4])
        rfm['RFM_Score'] = rfm[['R','F','M']].astype(int).sum(axis=1)
        st.dataframe(rfm.head())
        fig, ax = plt.subplots()
        sns.histplot(rfm['RFM_Score'], bins=10, kde=True)
        st.pyplot(fig)
    else:
        st.error("Les colonnes 'Recence', 'Frequence', et 'Montant' sont requises pour l'analyse RFM.")

# Application principale
if uploaded_file:
    try:
        df = load_data(uploaded_file)
        st.write("Aperçu des données :")
        st.dataframe(df.head())

        model_choice = st.sidebar.selectbox("Choisir une analyse", ["Statistiques descriptives", "FP-Growth", "K-means", "RFM"])

        if model_choice == "Statistiques descriptives":
            show_descriptive_stats(df)
        elif model_choice == "FP-Growth":
            apply_fp_growth(df)
        elif model_choice == "K-means":
            apply_kmeans(df)
        elif model_choice == "RFM":
            apply_rfm(df)

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement ou traitement des données : {e}")
else:
    st.info("📂 Veuillez charger un fichier CSV ou Excel pour commencer.")
