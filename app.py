import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
from statsmodels.tsa.seasonal import seasonal_decompose

# Configuration de la page
st.set_page_config(
    page_title="Analyse Clients e-commerce",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions utilitaires
def load_data(uploaded_file):
    """Charge les données depuis un fichier uploadé"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
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

def prepare_fpgrowth_data(df):
    """Prépare les données pour l'analyse FP-Growth"""
    try:
        basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        return basket
    except Exception as e:
        st.error(f"Erreur lors de la préparation des données pour FP-Growth: {e}")
        return None

def perform_fpgrowth_analysis(df, min_support, min_confidence):
    """Effectue l'analyse FP-Growth et retourne les règles"""
    try:
        frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        return rules.sort_values(by='lift', ascending=False)
    except Exception as e:
        st.error(f"Erreur lors de l'analyse FP-Growth: {e}")
        return None

def perform_kmeans_clustering(df, n_clusters, n_init=10):
    """Effectue le clustering K-means"""
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.select_dtypes(include=np.number))
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        silhouette = silhouette_score(scaled_data, clusters)
        return clusters, silhouette, kmeans.cluster_centers_
    except Exception as e:
        st.error(f"Erreur lors du clustering K-means: {e}")
        return None, None, None

def perform_rfm_analysis(df, customer_id_col, invoice_date_col, amount_col, invoice_no_col):
    """Effectue l'analyse RFM"""
    try:
        df[invoice_date_col] = pd.to_datetime(df[invoice_date_col])
        snapshot_date = df[invoice_date_col].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby(customer_id_col).agg({
            invoice_date_col: lambda x: (snapshot_date - x.max()).days,
            invoice_no_col: 'nunique',
            amount_col: 'sum'
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        rfm['R'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm['F'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm['M'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        rfm['RFM_Score'] = rfm['R'] + rfm['F'] + rfm['M']
        rfm['Segment'] = 'Low-Value'
        rfm.loc[rfm['RFM_Score'] >= 9, 'Segment'] = 'High-Value'
        rfm.loc[(rfm['RFM_Score'] >= 6) & (rfm['RFM_Score'] < 9), 'Segment'] = 'Mid-Value'
        
        segment_map = {
            r'[1-2][1-2][1-2]': 'Hibernating',
            r'[1-2][1-2][3-4]': 'At Risk',
            r'[1-2][3-4][1-2]': 'Needs Attention',
            r'[1-2][3-4][3-4]': 'Promising',
            r'[1-2][4-5][4-5]': 'New Customers',
            r'[3-4][1-2][1-2]': 'About to Sleep',
            r'[3-4][1-2][3-4]': 'Potential Loyalists',
            r'[3-4][3-4][1-2]': 'Loyal',
            r'[3-4][3-4][3-4]': 'Loyal',
            r'[3-4][4-5][4-5]': 'Champions',
            r'[5][1-2][1-2]': 'Can\'t Lose Them',
            r'[5][1-2][3-4]': 'Can\'t Lose Them',
            r'[5][3-4][1-2]': 'Loyal',
            r'[5][3-4][3-4]': 'Champions',
            r'[5][4-5][4-5]': 'Super Champions'
        }
        
        rfm['RFM_Combined'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
        rfm['Detailed_Segment'] = rfm['RFM_Combined'].replace(segment_map, regex=True)
        
        return rfm
    except Exception as e:
        st.error(f"Erreur lors de l'analyse RFM: {e}")
        return None

def export_results(data, file_name_prefix):
    """Permet d'exporter les résultats au format CSV ou Excel"""
    export_format = st.selectbox("Format d'export", ['CSV', 'Excel'])
    
    if export_format == 'CSV':
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger en CSV",
            data=csv,
            file_name=f"{file_name_prefix}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
    else:
        excel = io.BytesIO()
        with pd.ExcelWriter(excel, engine='openpyxl') as writer:
            data.to_excel(writer, index=False)
        excel.seek(0)
        st.download_button(
            label="Télécharger en Excel",
            data=excel,
            file_name=f"{file_name_prefix}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

def time_based_analysis(df, date_col, value_col):
    """Analyse temporelle des données"""
    st.subheader("Analyse Temporelle")
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    time_period = st.selectbox("Période d'agrégation", ['Journalière', 'Hebdomadaire', 'Mensuelle', 'Trimestrielle', 'Annuelle'])
    
    if time_period == 'Journalière':
        df_time = df.set_index(date_col)[value_col].resample('D').sum()
    elif time_period == 'Hebdomadaire':
        df_time = df.set_index(date_col)[value_col].resample('W').sum()
    elif time_period == 'Mensuelle':
        df_time = df.set_index(date_col)[value_col].resample('M').sum()
    elif time_period == 'Trimestrielle':
        df_time = df.set_index(date_col)[value_col].resample('Q').sum()
    else:
        df_time = df.set_index(date_col)[value_col].resample('Y').sum()
    
    fig = px.line(
        df_time.reset_index(),
        x=date_col,
        y=value_col,
        title=f"Évolution {time_period.lower()} de {value_col}"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Analyse de Saisonnalité")
    
    if len(df_time) > 12:
        decomposition = seasonal_decompose(df_time.fillna(0), model='additive', period=12)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        decomposition.trend.plot(ax=ax1)
        ax1.set_title('Tendance')
        decomposition.seasonal.plot(ax=ax2)
        ax2.set_title('Saisonnalité')
        decomposition.resid.plot(ax=ax3)
        ax3.set_title('Résidus')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Pas assez de données pour l'analyse de saisonnalité")

def customer_journey_analysis(df, customer_col, date_col, product_col):
    """Analyse du parcours client"""
    st.subheader("Analyse du Parcours Client")
    
    selected_customer = st.selectbox("Sélectionnez un client", df[customer_col].unique())
    
    customer_data = df[df[customer_col] == selected_customer].sort_values(date_col)
    
    fig = px.timeline(
        customer_data,
        x_start=date_col,
        x_end=date_col,
        y=product_col,
        title=f"Historique d'achats pour le client {selected_customer}"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Produits les plus achetés par ce client")
    top_products = customer_data[product_col].value_counts().head(10).reset_index()
    top_products.columns = ['Produit', 'Nombre d\'achats']
    
    fig2 = px.bar(
        top_products,
        x='Produit',
        y='Nombre d\'achats',
        title="Top 10 des produits achetés"
    )
    st.plotly_chart(fig2, use_container_width=True)

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
            selected_num = st.selectbox("Variable numérique", num_cols)
            
            if selected_num:
                fig = px.histogram(
                    df, 
                    x=selected_num, 
                    nbins=50, 
                    title=f"Distribution de {selected_num}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Test de Normalité (QQ-Plot)")
                fig, ax = plt.subplots()
                stats.probplot(df[selected_num].dropna(), plot=ax)
                st.pyplot(fig)
        
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
                    
                    fig = px.pie(
                        count_data,
                        names='Catégorie',
                        values='Count',
                        title=f"Répartition de {selected_cat}"
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
                default=num_cols[:3]
            )
            
            if selected_box:
                fig = px.box(
                    df.melt(value_vars=selected_box),
                    x='variable',
                    y='value',
                    title="Comparaison des Distributions"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Matrice de Variance-Covariance")
                cov_matrix = df[selected_box].cov().round(2)
                st.dataframe(cov_matrix)
                
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
                trendline="ols",
                title=f"Relation entre {x_var} et {y_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            corr = df[[x_var, y_var]].corr().iloc[0,1]
            st.metric("Coefficient de Corrélation", f"{corr:.2f}")
            
            if cat_cols:
                color_var = st.selectbox("Colorier par", [None] + cat_cols)
                
                if color_var:
                    fig = px.scatter(
                        df,
                        x=x_var,
                        y=y_var,
                        color=color_var,
                        title=f"Relation {x_var} vs {y_var} par {color_var}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Au moins 2 variables numériques requises pour l'analyse bivariée")

# Interface Streamlit
def main():
    st.title("🛒 Analyse des Données Clients e-commerce")
    st.markdown("""
    Cette application permet d'analyser les données clients d'un site e-commerce à l'aide de trois méthodes:
    - **Règles d'association (FP-Growth)**: Découvrir quels produits sont fréquemment achetés ensemble
    - **Segmentation (K-means)**: Grouper les clients en clusters similaires
    - **Analyse RFM**: Segmenter les clients basé sur la Récence, Fréquence et Montant des achats
    """)
    
    # Chargement des données
    st.sidebar.header("Chargement des Données")
    uploaded_file = st.sidebar.file_uploader("Uploader votre fichier de données (CSV ou Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            tab1, tab2, tab3, tab4 = st.tabs(["Statistiques", "FP-Growth", "K-means", "RFM"])
            
            with tab1:
                show_descriptive_stats(df)
                
                if st.checkbox("Afficher l'analyse temporelle"):
                    date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    
                    if date_cols and numeric_cols:
                        col1, col2 = st.columns(2)
                        with col1:
                            selected_date_col = st.selectbox("Colonne de date", date_cols)
                        with col2:
                            selected_value_col = st.selectbox("Colonne à analyser", numeric_cols)
                        
                        time_based_analysis(df, selected_date_col, selected_value_col)
                    else:
                        st.warning("Veuillez vous assurer que votre dataset contient des colonnes de date et des valeurs numériques")
                
                if st.checkbox("Afficher l'analyse du parcours client"):
                    customer_cols = df.columns[df.nunique() < 100].tolist()
                    product_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if date_cols and customer_cols and product_cols:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            selected_customer_col = st.selectbox("Colonne ID Client", customer_cols)
                        with col2:
                            selected_date_col = st.selectbox("Colonne Date", date_cols)
                        with col3:
                            selected_product_col = st.selectbox("Colonne Produit", product_cols)
                        
                        customer_journey_analysis(df, selected_customer_col, selected_date_col, selected_product_col)
                    else:
                        st.warning("Veuillez vous assurer que votre dataset contient des colonnes appropriées pour l'analyse du parcours client")
            
            with tab2:
                st.header("Analyse par Règles d'Association (FP-Growth)")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_support = st.slider("Support minimum", 0.01, 0.5, 0.05, 0.01)
                with col2:
                    min_confidence = st.slider("Confiance minimum", 0.1, 1.0, 0.7, 0.05)
                
                if st.button("Exécuter FP-Growth"):
                    with st.spinner("Préparation des données..."):
                        fpgrowth_data = prepare_fpgrowth_data(df)
                    
                    if fpgrowth_data is not None:
                        with st.spinner("Calcul des règles d'association..."):
                            rules = perform_fpgrowth_analysis(fpgrowth_data, min_support, min_confidence)
                        
                        if rules is not None:
                            st.success("Analyse FP-Growth terminée!")
                            
                            st.subheader("Top 10 des règles d'association par lift")
                            st.dataframe(rules.head(10))
                            
                            fig = px.scatter(
                                rules, 
                                x='support', 
                                y='confidence', 
                                size='lift', 
                                color='lift',
                                hover_data=['antecedents', 'consequents'],
                                title="Règles d'Association: Support vs Confiance"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Export des Résultats")
                            export_results(rules, "regles_association")
                            
                            st.subheader("Analyse des Règles")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                min_lift = st.slider("Filtrer par lift minimum", float(rules['lift'].min()), float(rules['lift'].max()), 1.0)
                                filtered_rules = rules[rules['lift'] >= min_lift]
                                st.write(f"{len(filtered_rules)} règles avec lift >= {min_lift}")
                            
                            with col2:
                                selected_consequent = st.selectbox("Filtrer par produit conséquence", ['Tous'] + rules['consequents'].astype(str).unique().tolist())
                                if selected_consequent != 'Tous':
                                    filtered_rules = filtered_rules[filtered_rules['consequents'].astype(str) == selected_consequent]
                            
                            st.dataframe(filtered_rules.sort_values(by='lift', ascending=False))
            
            with tab3:
                st.header("Segmentation des Clients (K-means)")
                
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                selected_cols = st.multiselect("Sélectionnez les variables pour le clustering", numeric_cols, default=numeric_cols[:3])
                
                if len(selected_cols) >= 2:
                    st.subheader("Détermination du nombre optimal de clusters")
                    
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df[selected_cols])
                    
                    wcss = []
                    silhouette_scores = []
                    max_clusters = min(10, len(df)-1)
                    
                    for i in range(2, max_clusters+1):
                        kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
                        kmeans.fit(scaled_data)
                        wcss.append(kmeans.inertia_)
                        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
                    
                    fig1, ax1 = plt.subplots()
                    ax1.plot(range(2, max_clusters+1), wcss, marker='o')
                    ax1.set_title('Méthode du Coude')
                    ax1.set_xlabel('Nombre de clusters')
                    ax1.set_ylabel('WCSS')
                    st.pyplot(fig1)
                    
                    fig2, ax2 = plt.subplots()
                    ax2.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
                    ax2.set_title('Scores de Silhouette')
                    ax2.set_xlabel('Nombre de clusters')
                    ax2.set_ylabel('Score Silhouette')
                    st.pyplot(fig2)
                    
                    n_clusters = st.slider("Nombre de clusters", 2, max_clusters, 3)
                    n_init = st.slider("Nombre d'initialisations", 1, 20, 10)
                    
                    if st.button("Exécuter K-means"):
                        with st.spinner("Clustering en cours..."):
                            clusters, silhouette, centers = perform_kmeans_clustering(df[selected_cols], n_clusters, n_init)
                        
                        if clusters is not None:
                            st.success(f"Clustering terminé! Score de silhouette: {silhouette:.3f}")
                            
                            df_clustered = df.copy()
                            df_clustered['Cluster'] = clusters
                            
                            cluster_stats = df_clustered.groupby('Cluster')[selected_cols].mean().reset_index()
                            st.subheader("Caractéristiques moyennes par cluster")
                            st.dataframe(cluster_stats)
                            
                            if len(selected_cols) >= 2:
                                fig = px.scatter(
                                    df_clustered,
                                    x=selected_cols[0],
                                    y=selected_cols[1],
                                    color='Cluster',
                                    title="Visualisation des Clusters",
                                    hover_data=selected_cols
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Export des Résultats")
                            export_results(df_clustered, "clusters_client")
                            
                            st.subheader("Analyse Comparative des Clusters")
                            comparison_var = st.selectbox("Variable à comparer", selected_cols)
                            
                            fig = px.box(
                                df_clustered,
                                x='Cluster',
                                y=comparison_var,
                                color='Cluster',
                                title=f"Distribution de {comparison_var} par Cluster"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Profilage des Clusters")
                            profile_df = df_clustered.groupby('Cluster')[selected_cols].mean().T
                            st.dataframe(profile_df.style.background_gradient(cmap='Blues'))
                            
                            if len(selected_cols) > 2:
                                fig = go.Figure()
                                
                                for cluster in sorted(df_clustered['Cluster'].unique()):
                                    cluster_data = profile_df[cluster].values
                                    fig.add_trace(go.Scatterpolar(
                                        r=cluster_data,
                                        theta=selected_cols,
                                        fill='toself',
                                        name=f'Cluster {cluster}'
                                    ))
                                
                                fig.update_layout(
                                    polar=dict(radialaxis=dict(visible=True)),
                                    title="Comparaison des Clusters (Radar Chart)"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Proposition de Contrat de Maintenance")
                            st.markdown("""
                            **Pour maintenir l'efficacité du modèle de segmentation:**
                            - **Recalcul mensuel:** Réexécuter le clustering avec les nouvelles données
                            - **Surveillance:** Suivre l'évolution des caractéristiques des clusters
                            - **Alerte:** Configurer des alertes si le score de silhouette diminue significativement
                            - **Réévaluation:** Revoir le nombre de clusters tous les 6 mois
                            """)
                else:
                    st.warning("Veuillez sélectionner au moins 2 variables numériques.")
            
            with tab4:
                st.header("Analyse RFM (Récence, Fréquence, Monétaire)")
                
                st.subheader("Configuration des colonnes RFM")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    customer_col = st.selectbox("Colonne ID Client", df.columns)
                with col2:
                    date_col = st.selectbox("Colonne Date", df.columns)
                with col3:
                    amount_col = st.selectbox("Colonne Montant", df.columns)
                with col4:
                    invoice_col = st.selectbox("Colonne Numéro de Facture", df.columns)
                
                if st.button("Exécuter l'analyse RFM"):
                    with st.spinner("Calcul des scores RFM..."):
                        rfm = perform_rfm_analysis(df, customer_col, date_col, amount_col, invoice_col)
                    
                    if rfm is not None:
                        st.success("Analyse RFM terminée!")
                        
                        st.subheader("Distribution des Segments RFM")
                        segment_counts = rfm['Detailed_Segment'].value_counts().reset_index()
                        segment_counts.columns = ['Segment', 'Nombre de Clients']
                        
                        fig1 = px.bar(
                            segment_counts,
                            x='Segment',
                            y='Nombre de Clients',
                            title="Répartition des Segments RFM"
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        fig2 = px.treemap(
                            segment_counts,
                            path=['Segment'],
                            values='Nombre de Clients',
                            title="Treemap des Segments RFM"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        st.subheader("Distribution des Scores RFM par Segment")
                        
                        fig3 = px.box(
                            rfm,
                            x='Detailed_Segment',
                            y='Recency',
                            title="Récence par Segment"
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        fig4 = px.box(
                            rfm,
                            x='Detailed_Segment',
                            y='Frequency',
                            title="Fréquence par Segment"
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                        
                        fig5 = px.box(
                            rfm,
                            x='Detailed_Segment',
                            y='Monetary',
                            title="Valeur Monétaire par Segment"
                        )
                        st.plotly_chart(fig5, use_container_width=True)
                        
                        st.subheader("Détails des Segments")
                        st.dataframe(rfm.groupby('Detailed_Segment').agg({
                            'Recency': 'mean',
                            'Frequency': 'mean',
                            'Monetary': ['mean', 'count']
                        }).sort_values(by=('Monetary', 'mean'), ascending=False))
                        
                        st.subheader("Recommandations par Segment")
                        recommendations = {
                            'Champions': "Récompenser, encourager à continuer, up-sell/cross-sell",
                            'Loyal': "Programmes de fidélité, offres personnalisées",
                            'Potential Loyalists': "Encourager à acheter plus, offres groupées",
                            'New Customers': "Bienvenue, guide d'utilisation, offres de premier achat",
                            'Promising': "Engager, offres ciblées, rappels",
                            'Needs Attention': "Offres limitées, rappels, réactivation",
                            'About to Sleep': "Relances, offres de retour, feedback",
                            'At Risk': "Offres personnalisées, feedback, rappels",
                            'Can\'t Lose Them': "Offres spéciales, rappels personnalisés",
                            'Hibernating': "Campagnes de réactivation agressives, enquêtes"
                        }
                        
                        rec_df = pd.DataFrame.from_dict(recommendations, orient='index', columns=['Recommandation'])
                        st.dataframe(rec_df)
                        
                        st.subheader("Export des Résultats")
                        export_results(rfm, "segmentation_rfm")
                        
                        st.subheader("Analyse RFM Avancée")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Top Clients par Valeur Monétaire**")
                            top_customers = rfm.sort_values('Monetary', ascending=False).head(10)
                            st.dataframe(top_customers)
                        
                        with col2:
                            st.write("**Clients à Risque (Récence élevée)**")
                            at_risk = rfm[rfm['Recency'] > rfm['Recency'].quantile(0.75)].sort_values('Monetary', ascending=False).head(10)
                            st.dataframe(at_risk)
                        
                        st.subheader("Matrice RFM")
                        
                        rfm_matrix = rfm.groupby(['R', 'F']).agg({
                            'CustomerID': 'count',
                            'Monetary': 'mean'
                        }).reset_index().rename(columns={'CustomerID': 'Nombre de Clients'})
                        
                        fig = px.scatter(
                            rfm_matrix,
                            x='R',
                            y='F',
                            size='Nombre de Clients',
                            color='Monetary',
                            title="Matrice RFM: Récence vs Fréquence",
                            labels={'R': 'Récence (1=ancien, 5=récent)', 'F': 'Fréquence (1=rare, 5=fréquent)'},
                            hover_data=['Nombre de Clients', 'Monetary']
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Section d'aide et d'informations
    st.sidebar.markdown("---")
    st.sidebar.header("Aide & Informations")

    with st.sidebar.expander("À propos de cette application"):
        st.markdown("""
        **Application d'Analyse Client e-commerce**
        
        Cette application permet d'analyser les données clients avec trois approches:
        1. **FP-Growth**: Détection des règles d'association entre produits
        2. **K-means**: Segmentation des clients en groupes homogènes
        3. **RFM**: Segmentation basée sur la récence, fréquence et valeur monétaire
        
        Développé par [Votre Nom] - © 2024
        """)

    with st.sidebar.expander("Guide d'utilisation"):
        st.markdown("""
        1. **Charger** votre fichier de données (CSV ou Excel)
        2. Naviguer entre les différents onglets d'analyse
        3. **Configurer** les paramètres pour chaque modèle
        4. **Exporter** les résultats pour un usage externe
        5. Consulter les **visualisations** interactives
        """)

    with st.sidebar.expander("Exigences techniques"):
        st.markdown("""
        - Format des données: CSV ou Excel
        - Colonnes requises pour RFM:
            - Identifiant client unique
            - Date de transaction
            - Montant de la transaction
            - Numéro de commande
        - Pour FP-Growth: historique des produits achetés par commande
        """)

if __name__ == "__main__":
    main()
