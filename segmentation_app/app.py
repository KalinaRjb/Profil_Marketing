import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px


st.set_page_config(page_title="Segmentation Client", layout="wide")


# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv(r"data\marketing_campaign_clean_cluster.csv")

df = load_data()


# Titre principal
st.title("📊 Application de Segmentation Client")

# Tabs : Accueil, Vue globale, Analyse par cluster
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Accueil", "🌍 Vue Globale", "🔍 Analyse par Cluster", "📚 Conclusion sur les cluster" ])



# ---------- Tab 1: Accueil ----------
with tab1:
    st.header("Bienvenue !")
    st.markdown("""
    Cette application permet d'explorer une segmentation client issue d'une analyse de **clustering**.
    
    Elle permet à l'entreprise de mieux comprendre ses clients et d'adapter plus facilement ses produits 
    aux besoins, comportements et préoccupations spécifiques de chaque type de clientèle.
    
     permet à une entreprise d'adapter son produit en fonction de sa clientèle cible issue de différents 
    segments de clientèle. Par exemple, au lieu de dépenser de l'argent pour commercialiser un nouveau produit 
    auprès de chaque client de sa base de données, une entreprise peut analyser le segment de clientèle le plus 
    susceptible d'acheter le produit et le commercialiser uniquement auprès de ce segment.
    
    **Fonctionnalités :**
    - Visualisation des clusters
    - Analyse descriptive des segments
    - Comparaison des caractéristiques
    
    Données : `marketing_campaign_clean_cluster.csv` avec les colonnes principales et la variable `Cluster`.
    """)


    # ---------- Tab 2: Vue Globale ----------

with tab2:
    st.header("🌍 Visualisation Globale des Clusters")

    # PCA pour visualisation
    features = df.drop(columns=["Cluster"])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    df["PCA1"], df["PCA2"] = pca_result[:, 0], pca_result[:, 1]

    fig = px.scatter(df, x="PCA1", y="PCA2", color=df["Cluster"].astype(str),
                     title="Projection PCA des clients par cluster",
                     labels={"color": "Cluster"})
    st.plotly_chart(fig, use_container_width=True)

    # Stats globales
    st.subheader("📈 Statistiques générales")

    df = df.drop(columns=['Unnamed: 0' , 'PCA1' , 'PCA2'])

    st.subheader("👤 Personnes")
    st.markdown("""**Caratéristiques générales**""")
    df_personne = df[['Cluster', 'age_Customer', 'Income', 'Children_num', 'Recency']]
    st.dataframe(df_personne.groupby("Cluster").mean(numeric_only=True).round(2))

    # Liste des variables à tracer
    variables = [col for col in df_personne.columns if col != "Cluster"]

    # Nombre de colonnes dans la grille (modulable)
    n_cols = 2

    # Création de lignes de boxplots
    for i in range(0, len(variables), n_cols):
        cols = st.columns(n_cols)
        for j, var in enumerate(variables[i:i + n_cols]):
            with cols[j]:
                fig = px.box(
                    df_personne,
                    x="Cluster",
                    y=var,
                    color="Cluster",
                    title=f"{var} par cluster",
                    labels={"Cluster": "Cluster", var: var}
                )
                st.plotly_chart(fig, use_container_width=True)
        

    st.markdown("""**Statut marital % de personnes en couple**""")
    df_relationship = df[['Cluster','relationship']]

    # Initialisation d'un DataFrame vide avec la colonne 'Cluster'
    result_relashionship = df_relationship[["Cluster"]].drop_duplicates().reset_index(drop=True)
    for var in df_relationship.columns:
        if var != "Cluster":
            temp = (
                df_relationship.groupby("Cluster")[var]
                .apply(lambda x: ((x == 1).mean() * 100).round(0))
                .reset_index(name=f"{var}_1")
            )               
            if result_relashionship.empty:
                 result_relashionship = temp
            else:
                result_relashionship = result_relashionship.merge(temp, on="Cluster")

    # Affichage final
    st.dataframe(result_relashionship)
   

    st.markdown("""**Education, % de personne dans le niveau d'éducation**""")
    df_education = df[['Cluster','Education_2n Cycle', 'Education_Basic', 'Education_Graduation', 
                       'Education_Master', 'Education_PhD']]

    # Initialisation d'un DataFrame vide avec la colonne 'Cluster'
    result_education = df_education[["Cluster"]].drop_duplicates().reset_index(drop=True)
    for var in df_education.columns:
        if var != "Cluster":
            temp = (
                df_education.groupby("Cluster")[var]
                .apply(lambda x: ((x == 1).mean() * 100).round(0))
                .reset_index(name=f"{var}_1")
            )               
            if result_education.empty:
                 result_education = temp
            else:
                result_education = result_education.merge(temp, on="Cluster")

    # Affichage final
    st.dataframe(result_education)

    st.markdown("""**Complaintes, % de personnes ayant fait au moins une réclamation**""")
    df_complaintes = df[['Cluster','Complain']]
    result_complaintes = df_complaintes[["Cluster"]].drop_duplicates().reset_index(drop=True)
    for var in df_complaintes.columns:
        if var != "Cluster":
            temp = (
                df_complaintes.groupby("Cluster")[var]
                .apply(lambda x: ((x == 1).mean() * 100).round(2))
                .reset_index(name=f"{var}_1")
            )                
            if result_complaintes.empty:
                 result_complaintes = temp
            else:
                result_complaintes = result_complaintes.merge(temp, on="Cluster")
    
    # Affichage final
    st.dataframe(result_complaintes)


    st.subheader("🥕🍷🍬 Produits achetés, montants moyens")
    df_produits = df[['Cluster','MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                      'MntSweetProducts', 'MntGoldProds','MntTotal']]
    st.dataframe(df_produits.groupby("Cluster").mean(numeric_only=True).round(0))

    # Liste des variables à tracer
    variables = [col for col in df_produits.columns if col != "Cluster"]

    # Nombre de colonnes dans la grille (modulable)
    n_cols = 2

    # Création de lignes de boxplots
    for i in range(0, len(variables), n_cols):
        cols = st.columns(n_cols)
        for j, var in enumerate(variables[i:i + n_cols]):
            with cols[j]:
                fig = px.box(
                    df_produits,
                    x="Cluster",
                    y=var,
                    color="Cluster",
                    title=f"{var} par cluster",
                    labels={"Cluster": "Cluster", var: var}
                )
                st.plotly_chart(fig, use_container_width=True)
            

    st.subheader("📉 Promotion, nombre moyen")
    df_promotion = df[['Cluster','NumDealsPurchases','Total_Accepted_Campaigns']]
    st.dataframe(df_promotion.groupby("Cluster").mean(numeric_only=True).round(2))
     # Liste des variables à tracer
    variables = [col for col in df_promotion.columns if col != "Cluster"]

    # Nombre de colonnes dans la grille (modulable)
    n_cols = 2

    # Création de lignes de boxplots
    for i in range(0, len(variables), n_cols):
        cols = st.columns(n_cols)
        for j, var in enumerate(variables[i:i + n_cols]):
            with cols[j]:
                fig = px.box(
                    df_promotion,
                    x="Cluster",
                    y=var,
                    color="Cluster",
                    title=f"{var} par cluster",
                    labels={"Cluster": "Cluster", var: var}
                )
                st.plotly_chart(fig, use_container_width=True)
            


    st.subheader("🏬 Lieu d'achat, nombre d'achats moyen effectués sur le lieu")
    df_lieu = df[['Cluster','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth']]
    st.dataframe(df_lieu.groupby("Cluster").mean(numeric_only=True).round(0))
    # Liste des variables à tracer
    variables = [col for col in df_lieu.columns if col != "Cluster"]

    # Nombre de colonnes dans la grille (modulable)
    n_cols = 2

    # Création de lignes de boxplots
    for i in range(0, len(variables), n_cols):
        cols = st.columns(n_cols)
        for j, var in enumerate(variables[i:i + n_cols]):
            with cols[j]:
                fig = px.box(
                    df_lieu,
                    x="Cluster",
                    y=var,
                    color="Cluster",
                    title=f"{var} par cluster",
                    labels={"Cluster": "Cluster", var: var}
                )
                st.plotly_chart(fig, use_container_width=True)
            
    
    # ---------- Tab 3: Analyse par Cluster ----------
with tab3:
    st.header("🔍 Analyse détaillée par cluster")

    # Sélection du cluster
    cluster_ids = sorted(df["Cluster"].unique())
    selected_cluster = st.selectbox("Choisir un cluster à analyser :", cluster_ids)

    cluster_df = df[df["Cluster"] == selected_cluster]
    st.write(f"Nombre de clients dans ce cluster : {len(cluster_df)}")

    # ------------------ Variables quantitatives ------------------
    st.subheader("📌 Moyennes des variables quantitatives pour ce cluster")
    quanti_df = cluster_df.drop(columns=[
        'AcceptedCmp1', 'AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5',
        'Complain','Response','relationship','Puchases_total'
    ])
    st.dataframe(quanti_df.describe().T[["mean", "std"]].round(2))

    st.subheader("📊 Distribution des variables")

    st.markdown("### 📈 Variables numériques")
    numeric_cols = quanti_df.select_dtypes(include='number').columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Choisissez une variable numérique :", numeric_cols)

        fig = px.histogram(cluster_df, x=selected_col, nbins=20,
                           title=f"Distribution de {selected_col} (Cluster {selected_cluster})",
                           labels={selected_col: selected_col})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune variable numérique trouvée dans ce cluster.")

    # ------------------ Variables qualitatives binaires ------------------
    st.markdown("### 🧾 Variables qualitatives (binaires 0/1)")
    qual_df = cluster_df[['Education_2n Cycle', 'Education_Basic', 'Education_Graduation', 
                          'Education_Master', 'Education_PhD','Complain','Response','relationship']]

    binary_vars = [col for col in qual_df.columns if qual_df[col].dropna().isin([0, 1]).all() and qual_df[col].nunique() <= 2]

    if binary_vars:
            selected_var = st.selectbox("Choisissez une variable binaire :", binary_vars)

            # Compter les 0 et 1 dans le cluster sélectionné
            counts = cluster_df[selected_var].value_counts().reset_index()
            counts.columns = [selected_var, 'Nombre de clients']
            total = counts['Nombre de clients'].sum()

            # Calcul du pourcentage
            counts['%'] = round(100 * counts['Nombre de clients'] / total, 1)

            fig = px.bar(counts,
                        x=selected_var,
                        y='%',
                        text='%',
                        title=f"Distribution (%) de la variable binaire '{selected_var}' (Cluster {selected_cluster})",
                        labels={selected_var: selected_var, '%': 'Pourcentage'})

            fig.update_traces(textposition='outside')
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1]),
                            yaxis=dict(title='Pourcentage (%)'))

            st.plotly_chart(fig, use_container_width=True)


    # ------------------ Conclusion sur les cluster ------------------
with tab4:
    st.markdown("""
                
    ### **Cluster 0**

**Caractéristiques générales :**

Ce groupe se compose de clients plus âgés, avec un revenu élevé, peu d’enfants, et des clients récents. 59 % sont en couple. Tous ont obtenu un diplôme. Ils ont tendance à faire des réclamations.

**Comportement d’achat :**

Ils réalisent le **montant d’achat total le plus élevé**, à égalité avec le Cluster 1. Toutefois, ils achètent **moins de fruits, poissons, sucreries et or**, mais **plus de vin** que le Cluster 1.

**Promotions et campagnes :**

Ils achètent **peu de produits en promotion**, mais sont ceux qui **acceptent le plus d’offres lors des campagnes marketing**.

**Canaux d’achat :**

Ils effectuent **principalement leurs achats en grande surface** et **consultent très peu le site web**.

---

### **Cluster 1**

**Caractéristiques générales :**

Deuxième groupe le plus âgé, avec un revenu élevé et peu d’enfants. Les clients sont plus anciens que ceux du Cluster 0. 59 % sont en couple, tous diplômés, et ils sont ceux qui font **le plus de réclamations**.

**Comportement d’achat :**

Ils partagent avec le Cluster 0 **le montant d’achat total le plus élevé**, mais consomment **davantage de fruits, poissons, sucreries et or**, et **moins de vin** que le Cluster 0.

**Promotions et campagnes :**

Ils achètent **peu de produits en promotion**, mais **acceptent un grand nombre d’offres commerciales**.

**Canaux d’achat :**

Comme le Cluster 0, ils privilégient **les grandes surfaces** et **utilisent peu le site web**.

---

### **Cluster 2**

**Caractéristiques générales :**

Le groupe le plus jeune, avec les **revenus les plus faibles** et **plus d’enfants**. Ce sont les clients **les plus anciens** parmi les 4 clusters. 66 % sont en couple. Leur **niveau d’éducation est élevé**, avec une majorité ayant un **master ou un doctorat**. Ils font **peu de réclamations**.

**Comportement d’achat :**

Ils présentent le **montant d’achat total le plus faible**, à égalité avec le Cluster 3, bien que légèrement supérieur. Ils achètent **davantage de fruits, viandes, poissons, sucreries et or**, et **moins de vin** que le Cluster 3.

**Promotions et campagnes :**

Ils achètent **beaucoup de produits en promotion**, mais **acceptent peu d’offres commerciales**.

**Canaux d’achat :**

Leurs achats se font **en grande surface ou sur le site web**.

---

### **Cluster 3**

**Caractéristiques générales :**

Groupe également très jeune, avec **revenus bas** et **plus d’enfants**. Clients très anciens comme dans le Cluster 2. 59 % sont en couple. Le **niveau d’éducation est élevé**, avec un grand nombre de **masters ou doctorats**. Ils font également **peu de réclamations**.

**Comportement d’achat :**

Ils ont le **montant d’achat total le plus faible**, à égalité avec le Cluster 2, mais légèrement en dessous. Ils achètent **moins de fruits, viandes, poissons, sucreries et or**, mais **plus de vin** que le Cluster 2.

**Promotions et campagnes :**

Ils sont ceux qui **achètent le plus en promotion**, mais **acceptent le moins d’offres de campagne**.

**Canaux d’achat :**

Ils effectuent leurs achats **en grande surface ou via le site web**.""")