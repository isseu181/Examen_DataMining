import streamlit as st

# Configuration de la page (DOIT ÊTRE LA PREMIÈRE COMMANDE)
st.set_page_config(
    page_title="Analyse e-commerce",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injection du script de gestion d'erreurs
st.components.v1.html("""
<script>
window.onerror = function(message, source, lineno, colno, error) {
  if (message.includes('removeChild') || 
      message.includes('NotFoundError') ||
      message.includes('nœud à supprimer')) {
    console.warn("[Streamlit] Erreur DOM ignorée:", message);
    return true;
  }
  return false;
};
</script>
""", height=0)

# Le reste des imports et du code...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import mlxtend
import squarify      
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn import metrics
import warnings

# Ignorer les warnings
warnings.filterwarnings("ignore")

# Fonctions utilitaires
def load_data(uploaded_file):
    """Charge les données depuis un fichier uploadé"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Essayer différents encodages
            try:
                df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
            except:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format de fichier non supporté. Veuillez uploader un fichier CSV ou Excel.")
            return None
        
        st.success("Données chargées avec succès!")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {e}")
        return None

# ... (le reste des fonctions reste inchangé) ...

# Interface principale
def main():
    st.title("🛒 Plateforme d'Analyse e-commerce")
    st.markdown("""
    Cette application permet d'analyser les données clients d'un site e-commerce à l'aide de trois approches:
    - **Règles d'association (FP-Growth)**: Découvrir quels produits sont fréquemment achetés ensemble
    - **Segmentation (K-means)**: Grouper les clients en clusters similaires
    - **Analyse RFM**: Segmenter les clients basé sur la Récence, Fréquence et Montant des achats
    """)
    
    # Chargement des données
    st.sidebar.header("Chargement des Données")
    uploaded_file = st.sidebar.file_uploader("Uploader votre fichier de données (CSV ou Excel)", type=['csv', 'xlsx'])
    
    df = None
    df_clean = None
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            # Onglets principaux
            tab_stats, tab_fp, tab_kmeans, tab_rfm = st.tabs([
                "Statistiques", 
                "FP-Growth", 
                "K-means", 
                "RFM"
            ])
            
            with tab_stats:
                # Statistiques globales
                st.header("Statistiques Globales")
                stats_df = global_stats(df)
                safe_display_dataframe(stats_df)
                
                # Nettoyage des données
                if st.button("Nettoyer les données", key="clean_data_btn"):
                    with st.spinner("Nettoyage en cours..."):
                        df_clean = clean_data(df)
                    st.success("Données nettoyées avec succès!")
                    st.session_state.df_clean = df_clean
                
                # Afficher les stats descriptives si données nettoyées
                if 'df_clean' in st.session_state:
                    st.header("Données Après Nettoyage")
                    st.write(f"Dimensions: {st.session_state.df_clean.shape[0]} lignes, {st.session_state.df_clean.shape[1]} colonnes")
                    show_descriptive_stats(st.session_state.df_clean)
                else:
                    st.info("Cliquez sur 'Nettoyer les données' pour commencer l'analyse")
            
            # Onglet FP-Growth
            with tab_fp:
                st.header("Analyse par Règles d'Association (FP-Growth)")
                if 'df_clean' in st.session_state:
                    perform_fpgrowth_analysis(st.session_state.df_clean)
                else:
                    st.info("Veuillez d'abord nettoyer les données dans l'onglet Statistiques")
            
            # Onglet K-means
            with tab_kmeans:
                st.header("Segmentation des Clients (K-means)")
                if 'df_clean' in st.session_state:
                    perform_kmeans_analysis(st.session_state.df_clean)
                else:
                    st.info("Veuillez d'abord nettoyer les données dans l'onglet Statistiques")
            
            # Onglet RFM
            with tab_rfm:
                st.header("Analyse RFM (Récence, Fréquence, Montant)")
                if 'df_clean' in st.session_state:
                    perform_rfm_analysis(st.session_state.df_clean)
                else:
                    st.info("Veuillez d'abord nettoyer les données dans l'onglet Statistiques")
    else:
        st.info("Veuillez uploader un fichier de données pour commencer l'analyse")

if __name__ == "__main__":
    main()
