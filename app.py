import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import squarify
import mlxtend
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Ignorer les warnings
warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="Analyse e-commerce",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions utilitaires
def load_data(uploaded_file):
    """Charge les données depuis un fichier uploadé"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
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

def clean_data(df):
    """Nettoie les données"""
    # Suppression des lignes entièrement vides
    df = df.dropna(how='all')
    
    # Suppression des doublons
    df = df.drop_duplicates()
    
    # Suppression des transactions sans CustomerID
    if 'CustomerID' in df.columns:
        df = df[~df['CustomerID'].isna()]
    
    # Suppression des valeurs manquantes restantes
    df = df.dropna()
    
    # Conversion de la date si présente
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Réinitialiser les index
    df.reset_index(drop=True, inplace=True)
    
    # Ajout du montant total si possible
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['Mont_total'] = df['Quantity'] * df['UnitPrice']
    
    # Filtrer les codes stock invalides
    if 'StockCode' in df.columns:
        liste = df['StockCode'].unique()
        stock_to_del = [el for el in liste if not el[0].isdigit()]
        df = df[df['StockCode'].map(lambda x: x not in stock_to_del)]
    
    return df

def global_stats(df):
    """Calcule les statistiques globales du DataFrame"""
    stats = {
        "Indicateur": [
            "Nombre de variables",
            "Nombre d'observations",
            "Nombre de valeurs manquantes",
            "Pourcentage de valeurs manquantes",
            "Nombre de lignes dupliquées",
            "Pourcentage de lignes dupliquées",
            "Nombre de lignes entièrement vides",
            "Pourcentage de lignes vides",
            "Nombre de colonnes vides",
            "Pourcentage de colonnes vides"
        ],
        "Valeur": [
            df.shape[1],
            df.shape[0],
            df.isna().sum().sum(),
            "{:.2%}".format(df.isna().sum().sum() / df.size),
            df.duplicated().sum(),
            "{:.2%}".format(df.duplicated().sum() / len(df)),
            df.isna().all(axis=1).sum(),
            "{:.2%}".format(df.isna().all(axis=1).sum() / len(df)),
            df.isnull().all().sum(),
            "{:.2%}".format(df.isnull().all().sum() / df.shape[1])
        ]
    }

    return pd.DataFrame(stats)

def show_descriptive_stats(df):
    """Affiche des statistiques descriptives complètes avec visualisations"""
    st.header("📊 Analyse Descriptive Complète")
    
    tab_stats, tab_dist, tab_box, tab_bivar = st.tabs([
        "Statistiques de Base", 
        "Distributions", 
        "Boîtes à Moustaches", 
        "Analyse Bivariée"
    ])
    
    with tab_stats:
        st.subheader("Statistiques Summaries")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Aperçu des Données**")
            st.dataframe(df.head())
            
            st.write("**Types des Variables**")
            types_df = pd.DataFrame(df.dtypes, columns=['Type']).reset_index()
            types_df.columns = ['Variable', 'Type']
            st.dataframe(types_df)
            
        with col2:
            st.write("**Statistiques Numériques**")
            st.dataframe(df.describe())
            
            st.write("**Valeurs Manquantes**")
            missing = df.isnull().sum().reset_index()
            missing.columns = ['Variable', 'Nombre']
            missing['Pourcentage'] = (missing['Nombre'] / len(df)) * 100
            st.dataframe(missing)
    
    with tab_dist:
        st.subheader("Distributions des Variables")
        
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if num_cols:
                selected_num = st.selectbox("Variable numérique", num_cols)
                
                if selected_num:
                    fig = px.histogram(
                        df, 
                        x=selected_num, 
                        nbins=50, 
                        title=f"Distribution de {selected_num}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune variable numérique trouvée")
        
        with col2:
            if cat_cols:
                selected_cat = st.selectbox("Variable catégorielle", cat_cols)
                
                if selected_cat:
                    count_data = df[selected_cat].value_counts().reset_index()
                    count_data.columns = ['Catégorie', 'Count']
                    
                    fig = px.bar(
                        count_data, 
                        x='Catégorie', 
                        y='Count',
                        title=f"Distribution de {selected_cat}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune variable catégorielle trouvée")
    
    with tab_box:
        st.subheader("Boîtes à Moustaches et Variance")
        
        if num_cols:
            selected_box = st.multiselect(
                "Variables numériques à comparer", 
                num_cols, 
                default=num_cols[:min(3, len(num_cols))]
            )
            
            if selected_box:
                fig = px.box(
                    df.melt(value_vars=selected_box),
                    x='variable',
                    y='value',
                    title="Comparaison des Distributions"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Matrice de Corrélation")
                corr_matrix = df[selected_box].corr().round(2)
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Corrélations entre Variables"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune variable numérique trouvée pour les boîtes à moustaches")
    
    with tab_bivar:
        st.subheader("Analyse Bivariée")
        
        if len(num_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("Variable X", num_cols)
            with col2:
                y_var = st.selectbox("Variable Y", num_cols, index=1)
            
            fig = px.scatter(
                df,
                x=x_var,
                y=y_var,
                title=f"Relation entre {x_var} et {y_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if num_cols:
                corr = df[[x_var, y_var]].corr().iloc[0,1]
                st.metric("Coefficient de Corrélation", f"{corr:.2f}")
        else:
            st.warning("Au moins 2 variables numériques requises pour l'analyse bivariée")

def perform_fpgrowth_analysis(df):
    """Effectue l'analyse FP-Growth"""
    with st.spinner("Préparation des données pour FP-Growth..."):
        # Création du panier
        basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].count().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Support minimum", 0.01, 0.5, 0.01, 0.01)
    with col2:
        min_lift = st.slider("Lift minimum", 1.0, 10.0, 1.0, 0.1)
    
    if st.button("Exécuter FP-Growth"):
        with st.spinner("Calcul des règles d'association..."):
            frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_lift)
            
            # Nettoyage des règles
            rules['pair_key'] = rules.apply(lambda row: tuple(sorted([row['antecedents'], row['consequents']])), axis=1)
            rules = rules.drop_duplicates(subset='pair_key')
            rules.drop(columns='pair_key', inplace=True)
            
            # Conversion en liste
            rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
            rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
            
            # Formatage pour l'affichage
            rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ", ".join(x))
            rules['consequents_str'] = rules['consequents'].apply(lambda x: ", ".join(x))
        
        st.success(f"FP-Growth terminé! {len(rules)} règles générées")
        
        # Affichage des résultats
        st.subheader("Top 20 des Règles d'Association")
        st.dataframe(rules.sort_values(by='lift', ascending=False).head(20))
        
        # Visualisation
        st.subheader("Visualisation des Règles")
        
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Nombre de règles à visualiser", 5, 50, 10)
            
            fig1 = px.scatter(
                rules.head(top_n),
                x='support',
                y='confidence',
                size='lift',
                color='lift',
                hover_data=['antecedents_str', 'consequents_str'],
                title="Support vs Confiance"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                rules.sort_values(by='lift', ascending=False).head(top_n),
                x='lift',
                y='antecedents_str',
                color='consequents_str',
                orientation='h',
                title="Top Règles par Lift"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Téléchargement des résultats
        st.subheader("Télécharger les résultats")
        csv = rules.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les règles en CSV",
            data=csv,
            file_name="regles_association.csv",
            mime='text/csv'
        )

def perform_kmeans_analysis(df):
    """Effectue l'analyse K-means"""
    st.subheader("Sélection des Variables pour le Clustering")
    
    # Préparation des données RFM
    if 'InvoiceDate' in df.columns and 'InvoiceNo' in df.columns and 'Mont_total' in df.columns:
        date_ref = df['InvoiceDate'].max() + timedelta(days=1)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (date_ref - x.max()).days,
            'InvoiceNo': 'nunique',
            'Mont_total': 'sum'
        }).reset_index()
        rfm.columns = ['CustomerID', 'Recence', 'Frequence', 'Montant']
        df_rfm = rfm
    else:
        st.warning("Colonnes manquantes pour le calcul RFM. Utilisation de toutes les colonnes numériques.")
        df_rfm = df.select_dtypes(include=np.number)
    
    # Sélection des variables
    num_cols = df_rfm.select_dtypes(include=np.number).columns.tolist()
    selected_cols = st.multiselect(
        "Sélectionnez les variables pour le clustering", 
        num_cols, 
        default=num_cols[:min(3, len(num_cols))]
    )
    
    if not selected_cols:
        st.warning("Veuillez sélectionner au moins une variable numérique.")
        return
    
    # Standardisation des données
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_rfm[selected_cols])
    
    # Détermination du nombre de clusters
    st.subheader("Détermination du Nombre Optimal de Clusters")
    max_clusters = st.slider("Nombre maximum de clusters à tester", 3, 15, 10)
    
    if st.button("Trouver le nombre optimal de clusters"):
        with st.spinner("Calcul en cours..."):
            wcss = []  # Within-Cluster Sum of Square
            silhouette_scores = []
            
            for i in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                wcss.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
            
            # Trouver le meilleur nombre de clusters basé sur le score de silhouette
            best_n = np.argmax(silhouette_scores) + 2
        
        # Affichage des résultats
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Méthode du Coude")
            fig1, ax1 = plt.subplots()
            ax1.plot(range(2, max_clusters + 1), wcss, marker='o')
            ax1.set_title('Méthode du Coude')
            ax1.set_xlabel('Nombre de clusters')
            ax1.set_ylabel('WCSS')
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Score de Silhouette")
            fig2, ax2 = plt.subplots()
            ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color='green')
            ax2.set_title('Score de Silhouette')
            ax2.set_xlabel('Nombre de clusters')
            ax2.set_ylabel('Score de Silhouette')
            st.pyplot(fig2)
        
        st.success(f"Le nombre optimal de clusters est : {best_n} (score de silhouette: {silhouette_scores[best_n-2]:.3f})")
        st.session_state.best_n = best_n
        st.session_state.scaled_data = scaled_data
        st.session_state.df_rfm = df_rfm
        st.session_state.selected_cols = selected_cols
    
    # Application de K-means
    if 'best_n' in st.session_state:
        st.subheader("Application de K-means")
        n_clusters = st.slider(
            "Nombre de clusters", 
            2, max_clusters, st.session_state.best_n
        )
        
        if st.button("Exécuter K-means"):
            with st.spinner("Clustering en cours..."):
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
                clusters = kmeans.fit_predict(st.session_state.scaled_data)
                silhouette_avg = silhouette_score(st.session_state.scaled_data, clusters)
                
                # Ajout des clusters aux données
                df_clustered = st.session_state.df_rfm.copy()
                df_clustered['Cluster'] = clusters
            
            st.success(f"Clustering terminé! Score de silhouette: {silhouette_avg:.3f}")
            
            # Visualisation des clusters
            st.subheader("Visualisation des Clusters")
            
            if len(st.session_state.selected_cols) >= 2:
                fig = px.scatter(
                    df_clustered, 
                    x=st.session_state.selected_cols[0], 
                    y=st.session_state.selected_cols[1], 
                    color='Cluster',
                    title="Visualisation des Clusters",
                    hover_data=df_clustered.columns
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Profil des clusters
            st.subheader("Profil des Clusters")
            cluster_profile = df_clustered.groupby('Cluster').agg({
                'Recence': ['mean', 'min', 'max'],
                'Frequence': ['mean', 'min', 'max'],
                'Montant': ['mean', 'min', 'max'],
                'CustomerID': 'count'
            }).reset_index()
            
            st.dataframe(cluster_profile)
            
            # Interprétation
            st.subheader("Interprétation des Clusters")
            for cluster in sorted(df_clustered['Cluster'].unique()):
                cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
                recence_moy = cluster_data['Recence'].mean()
                frequence_moy = cluster_data['Frequence'].mean()
                montant_moy = cluster_data['Montant'].mean()
                
                st.markdown(f"**Cluster {cluster}** (Nombre: {len(cluster_data)})")
                st.markdown(f"- Récence moyenne: {recence_moy:.1f} jours")
                st.markdown(f"- Fréquence moyenne: {frequence_moy:.1f} commandes")
                st.markdown(f"- Montant moyen: {montant_moy:.1f} €")
                st.markdown("---")

def perform_rfm_analysis(df):
    """Effectue l'analyse RFM"""
    # Vérification des colonnes nécessaires
    required_cols = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Mont_total']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Colonnes manquantes pour l'analyse RFM: {', '.join(missing_cols)}")
        return
    
    # Calcul RFM
    date_ref = df['InvoiceDate'].max() + timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (date_ref - x.max()).days,
        'InvoiceNo': 'nunique',
        'Mont_total': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recence', 'Frequence', 'Montant']
    
    # Calcul des quartiles
    quantiles = rfm[['Recence', 'Frequence', 'Montant']].quantile([0.25, 0.5, 0.75])
    quantiles = quantiles.to_dict()
    
    # Fonctions de scoring
    def r_score(x):
        if x <= quantiles['Recence'][0.25]:
            return 4
        elif x <= quantiles['Recence'][0.5]:
            return 3
        elif x <= quantiles['Recence'][0.75]:
            return 2
        else:
            return 1
    
    def fm_score(x, var):
        if x <= quantiles[var][0.25]:
            return 1
        elif x <= quantiles[var][0.5]:
            return 2
        elif x <= quantiles[var][0.75]:
            return 3
        else:
            return 4
    
    # Application des scores
    rfm['R'] = rfm['Recence'].apply(r_score)
    rfm['F'] = rfm['Frequence'].apply(lambda x: fm_score(x, 'Frequence'))
    rfm['M'] = rfm['Montant'].apply(lambda x: fm_score(x, 'Montant'))
    rfm['RFM_Score'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)
    
    # Segmentation
    contrat = {
        r'[1-2][1-2]': 'en hibernation',
        r'[1-2][3-4]': 'à risque',
        r'[1-2]5': 'à ne pas perdre',
        r'3[1-2]': 'sur le point de dormir',
        r'33': 'nécessite de l\'attention',
        r'[3-4][4-5]': 'clients fidèles',
        r'41': 'prometteurs',
        r'51': 'nouveaux clients',
        r'[4-5][2-3]': 'clients potentiellement fidèles',
        r'5[4-5]': 'champions'
    }
    
    rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str)
    rfm['Segment'] = rfm['Segment'].replace(contrat, regex=True)
    
    # Affichage des résultats
    st.subheader("Résultats de l'analyse RFM")
    st.dataframe(rfm.head(10))
    
    # Visualisation
    st.subheader("Visualisation des Segments RFM")
    
    # Distribution des segments
    segment_counts = rfm['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Nombre de Clients']
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(
            segment_counts,
            x='Segment',
            y='Nombre de Clients',
            title="Répartition des Segments RFM"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.pie(
            segment_counts,
            names='Segment',
            values='Nombre de Clients',
            title="Distribution des Segments RFM"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Treemap
    st.subheader("Treemap des Segments RFM")
    plt.figure(figsize=(12, 8))
    colors = ['#f94144', '#f3722c', '#f9c74f', '#90be6d', '#43aa8b',
              '#577590', '#277da1', '#4d908e', '#f9844a', '#8ac926']
    
    squarify.plot(
        sizes=segment_counts['Nombre de Clients'],
        label=segment_counts['Segment'],
        color=colors,
        alpha=.9,
        text_kwargs={'fontsize': 12, 'weight': 'bold'}
    )
    
    plt.title("Segmentation RFM des Clients", fontsize=16)
    plt.axis('off')
    st.pyplot(plt)
    
    # Recommandations
    st.subheader("Recommandations par Segment")
    recommendations = {
        'champions': "Récompenser, offres exclusives, programmes VIP",
        'clients fidèles': "Fidélisation, offres personnalisées",
        'clients potentiellement fidèles': "Encourager à acheter plus",
        'à ne pas perdre': "Offres spéciales, attention particulière",
        'prometteurs': "Encourager à devenir fidèles",
        'à risque': "Campagnes de réactivation",
        'sur le point de dormir': "Relances par email, offres de retour",
        'nécessite de l\'attention': "Offres ciblées, rappels",
        'nouveaux clients': "Bienvenue, guide d'utilisation",
        'en hibernation': "Campagnes agressives de réactivation"
    }
    
    rec_df = pd.DataFrame.from_dict(recommendations, orient='index', columns=['Recommandation'])
    st.dataframe(rec_df)

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
                st.dataframe(stats_df)
                
                # Nettoyage des données
                if st.button("Nettoyer les données"):
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
