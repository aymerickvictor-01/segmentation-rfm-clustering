import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage as scipy_linkage
import joblib
import time

# ---------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------
st.set_page_config(
    page_title="CRM Analytics - Segmentation RFM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Styles CSS personnalisés (compatibles mode clair & sombre)
# ---------------------------------------------------------
# Toutes les couleurs de surface/texte utilisent les variables de thème
# natives de Streamlit (--background-color, --secondary-background-color,
# --text-color, --primary-color) plutôt que des couleurs codées en dur.
# Cela garantit un rendu correct aussi bien en mode clair qu'en mode sombre.
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Animations d'entree ---------- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        0%   { opacity: 0; transform: scale(0.92); }
        100% { opacity: 1; transform: scale(1); }
    }
    .main .block-container {
        animation: fadeIn 0.6s ease-out;
    }
    div[data-testid="stMetric"], .pred-card, .stPlotlyChart {
        animation: popIn 0.5s ease-out;
    }

    /* ---------- Cartes de metriques ---------- */
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 18px 20px;
        border-radius: 16px;
        transition: all 0.25s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: #4A90E2;
        box-shadow: 0 12px 24px rgba(74, 144, 226, 0.25);
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        opacity: 0.85;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }

    /* ---------- Boutons ---------- */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.1em;
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        font-weight: 600;
        font-size: 1.02em;
        border: none;
        letter-spacing: 0.2px;
        transition: all 0.25s ease;
        box-shadow: 0 4px 10px rgba(74, 144, 226, 0.25);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(74, 144, 226, 0.4);
        filter: brightness(1.05);
    }
    .stButton>button:active {
        transform: translateY(0px) scale(0.98);
    }

    /* ---------- En-tetes de section ---------- */
    .header-style {
        font-size: 30px;
        font-weight: 700;
        color: #4A90E2;
        margin-bottom: 18px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(74, 144, 226, 0.35);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    /* ---------- Carte resultat de prediction ---------- */
    .pred-card {
        background: linear-gradient(135deg, rgba(74,144,226,0.16) 0%, rgba(80,227,194,0.10) 100%);
        padding: 32px 24px;
        border-radius: 20px;
        border: 1.5px solid rgba(74, 144, 226, 0.45);
        text-align: center;
        box-shadow: 0 8px 20px rgba(74, 144, 226, 0.15);
    }
    .pred-card h3 {
        margin: 0;
        font-weight: 600;
        opacity: 0.85;
        font-size: 16px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .pred-card h1 {
        color: #4A90E2;
        font-size: 38px;
        margin: 12px 0 6px 0;
        font-weight: 700;
    }
    .pred-card p {
        font-size: 15px;
        opacity: 0.75;
        margin: 0;
    }
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        background: rgba(74, 144, 226, 0.18);
        color: #4A90E2;
        font-weight: 600;
        font-size: 13px;
        margin-top: 10px;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* ---------- Tableaux ---------- */
    .stDataFrame, [data-testid="stTable"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- Divider plus discret ---------- */
    hr {
        opacity: 0.18;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Chargement des fichiers (Modèles & Données)
# ---------------------------------------------------------
@st.cache_resource
def charger_modeles():
    try:
        scaler = joblib.load("scaler.pkl")
        kmeans = joblib.load("kmeans_model.pkl")
        mapping_segment = joblib.load("mapping_segment.pkl")
        return scaler, kmeans, mapping_segment
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        return None, None, None

@st.cache_data
def charger_donnees():
    try:
        return pd.read_csv("rfm_segments.csv")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        # Création de données factices pour la démo si le fichier manque
        df = pd.DataFrame({
            'Recency': np.random.randint(1, 365, 500),
            'Frequency': np.random.randint(1, 50, 500),
            'Monetary': np.random.uniform(10, 5000, 500),
            'Segment': np.random.choice(['Champions', 'Loyal Customers', 'At Risk', 'Lost'], 500)
        })
        return df

scaler, kmeans, mapping_segment = charger_modeles()
rfm = charger_donnees()

# Palette de couleurs professionnelle, cohérente sur tous les graphes
PALETTE = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B', '#9013FE', '#7ED321']
SEGMENT_COLOR_MAP = {
    seg: PALETTE[i % len(PALETTE)]
    for i, seg in enumerate(sorted(rfm['Segment'].unique()))
}

# Layout Plotly commun : fond transparent pour s'adapter au mode clair/sombre du thème Streamlit
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/dashboard.png", width=80)
    st.markdown("### CRM Analytics")
    st.caption("Segmentation client basée sur le modèle RFM")
    st.markdown("---")
    page = st.selectbox(
        "Navigation",
        ["📊 Tableau de Bord", "🔍 Prédiction Client"],
        index=0
    )
    st.markdown("---")
    st.info("💡 **Conseil** : Utilisez la vue 3D pour identifier les clusters de haute valeur.")
    st.caption("© 2026 CRM Analytics Dashboard")

# ===========================================================
# PAGE 1 : TABLEAU DE BORD (VUE D'ENSEMBLE)
# ===========================================================
if page == "📊 Tableau de Bord":
    st.markdown('<p class="header-style">📊 Vue d\'ensemble des segments clients</p>', unsafe_allow_html=True)

    # Métriques principales avec icônes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Clients", f"{len(rfm):,}")
    with col2:
        st.metric("📅 Récence Moyenne", f"{rfm['Recency'].mean():.0f} j")
    with col3:
        st.metric("🛒 Fréquence Moyenne", f"{rfm['Frequency'].mean():.1f}")
    with col4:
        st.metric("💰 Panier Moyen", f"{rfm['Monetary'].mean():,.0f} €")

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphiques principaux
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🥧 Répartition des Segments")
        repartition = rfm['Segment'].value_counts().reset_index()
        repartition.columns = ['Segment', 'Nombre']
        fig_pie = px.pie(
            repartition, values='Nombre', names='Segment',
            hole=0.55,
            color='Segment',
            color_discrete_map=SEGMENT_COLOR_MAP
        )
        fig_pie.update_traces(
            textposition='outside',
            textinfo='percent+label',
            marker=dict(line=dict(color='rgba(0,0,0,0.05)', width=1.5))
        )
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            margin=dict(t=10, b=10, l=10, r=10),
            height=380,
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("📈 Valeur Monétaire Moyenne par Segment")
        profil = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
        profil = profil.sort_values('Monetary', ascending=True)
        fig_bar = px.bar(
            profil, x='Monetary', y='Segment',
            color='Segment',
            orientation='h',
            text='Monetary',
            color_discrete_map=SEGMENT_COLOR_MAP
        )
        fig_bar.update_traces(
            texttemplate='%{text:,.0f} €',
            textposition='outside',
            marker_line_width=0
        )
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            showlegend=False,
            height=380,
            margin=dict(t=10, b=10, l=10, r=30),
            xaxis_title="Montant moyen (€)",
            yaxis_title=""
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Visualisation 3D avancée
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🌐 Visualisation 3D du Nuage de Clients (RFM)")
    fig_3d = px.scatter_3d(
        rfm, x='Recency', y='Frequency', z='Monetary',
        color='Segment',
        opacity=0.75,
        color_discrete_map=SEGMENT_COLOR_MAP,
        hover_data=['Recency', 'Frequency', 'Monetary']
    )
    fig_3d.update_traces(marker=dict(size=4, line=dict(width=0)))
    layout_3d = {**PLOTLY_LAYOUT}
    layout_3d["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    fig_3d.update_layout(
        **layout_3d,
        scene=dict(
            xaxis_title='Récence (Jours)',
            yaxis_title='Fréquence (Achats)',
            zaxis_title='Montant (€)',
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=680
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # ---------------------------------------------------------
    # Dendrogramme (Clustering Hiérarchique)
    # ---------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🌳 Dendrogramme du Clustering Hiérarchique (RFM)")
    st.caption(
        "Visualisation complémentaire à K-Means : regroupement progressif des clients "
        "selon leur proximité RFM (méthode de Ward). Un échantillon est utilisé pour la lisibilité."
    )

    col_dendro, col_param = st.columns([4, 1])
    with col_param:
        taille_echantillon = st.slider(
            "Taille de l'échantillon", min_value=20, max_value=150, value=60, step=10,
            help="Nombre de clients affichés dans le dendrogramme"
        )
        seuil_couleur = st.slider(
            "Seuil de coloration", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Position relative du seuil de coupe (couleur des branches)"
        )

    # Echantillon aleatoire (mais reproductible) des features RFM normalisees
    rfm_sample = rfm.sample(min(taille_echantillon, len(rfm)), random_state=42)
    X_sample = rfm_sample[['Recency', 'Frequency', 'Monetary']].values
    X_sample_scaled = (X_sample - X_sample.mean(axis=0)) / X_sample.std(axis=0)

    labels_dendro = [
        f"{i} • {seg}" for i, seg in zip(rfm_sample.index, rfm_sample['Segment'])
    ]

    dist_max = pdist(X_sample_scaled, metric='euclidean').max()

    fig_dendro = ff.create_dendrogram(
        X_sample_scaled,
        orientation='bottom',
        labels=labels_dendro,
        linkagefun=lambda x: scipy_linkage(x, method='ward'),
        color_threshold=seuil_couleur * dist_max
    )
    fig_dendro.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        margin=dict(t=20, b=120, l=40, r=20),
        xaxis=dict(title="Clients (échantillon)", tickangle=70, tickfont=dict(size=9)),
        yaxis=dict(title="Distance (Ward)"),
        showlegend=False
    )
    with col_dendro:
        st.plotly_chart(fig_dendro, use_container_width=True)

    # Tableau de données stylisé
    with st.expander("📋 Consulter la base de données clients"):
        st.dataframe(
            rfm.head(100).style.background_gradient(cmap='Blues', subset=['Monetary']),
            use_container_width=True
        )

# ===========================================================
# PAGE 2 : PRÉDICTION DU SEGMENT D'UN CLIENT
# ===========================================================
else:
    st.markdown('<p class="header-style">🔍 Analyse & Prédiction Individuelle</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("Saisissez les paramètres RFM pour positionner un client spécifique.")

    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input("📅 Récence (jours)", min_value=0, value=30, help="Jours depuis le dernier achat")
    with c2:
        frequency = st.number_input("🛒 Fréquence (achats)", min_value=1, value=5, help="Nombre total de commandes")
    with c3:
        monetary = st.number_input("💰 Montant (€)", min_value=0.0, value=500.0, step=50.0, help="Valeur totale dépensée")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Lancer l'Analyse")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        with st.spinner('Analyse des données en cours...'):
            time.sleep(0.6)  # Petit délai pour l'effet visuel

            # Préparation
            nouveau_client = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary]})

            if scaler is not None and kmeans is not None:
                nouveau_client_scaled = scaler.transform(nouveau_client)
                cluster_predit = kmeans.predict(nouveau_client_scaled)[0]
                segment_predit = mapping_segment[cluster_predit]
            else:
                # Fallback pour démo si modèles non chargés
                segment_predit = "Client Potentiel"
                cluster_predit = "N/A"

            segment_color = SEGMENT_COLOR_MAP.get(segment_predit, '#4A90E2')

            # Affichage du résultat
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.markdown(f"""
                <div class="pred-card">
                    <h3>Segment Prédit</h3>
                    <h1 style="color:{segment_color};">{segment_predit}</h1>
                    <span class="badge">Cluster ID : {cluster_predit}</span>
                </div>
                """, unsafe_allow_html=True)

                # Comparaison
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📊 Comparaison aux Moyennes")
                profil_segment = rfm[rfm['Segment'] == segment_predit][['Recency', 'Frequency', 'Monetary']].mean()

                comp_data = pd.DataFrame({
                    'Métrique': ['Récence (j)', 'Fréquence', 'Montant (€)'],
                    'Client': [recency, frequency, round(monetary, 1)],
                    'Moyenne Segment': profil_segment.round(1).values
                })
                st.dataframe(
                    comp_data.set_index('Métrique').style.background_gradient(cmap='Blues', axis=1),
                    use_container_width=True
                )

            with res_col2:
                st.subheader("📍 Positionnement 3D du Nouveau Client")
                fig_pred = go.Figure()

                # Points existants (échantillonnés pour la fluidité), un trace par segment
                sample_rfm = rfm.sample(min(1000, len(rfm)), random_state=42)
                for seg in sorted(sample_rfm['Segment'].unique()):
                    subset = sample_rfm[sample_rfm['Segment'] == seg]
                    fig_pred.add_trace(go.Scatter3d(
                        x=subset['Recency'], y=subset['Frequency'], z=subset['Monetary'],
                        mode='markers',
                        name=seg,
                        marker=dict(size=4, opacity=0.35, color=SEGMENT_COLOR_MAP.get(seg, '#999999'))
                    ))

                # Le nouveau client mis en evidence
                fig_pred.add_trace(go.Scatter3d(
                    x=[recency], y=[frequency], z=[monetary],
                    mode='markers+text',
                    name='NOUVEAU CLIENT',
                    text=['VOUS'],
                    textposition='top center',
                    marker=dict(
                        size=11,
                        color=segment_color,
                        symbol='diamond',
                        line=dict(color='white', width=2)
                    )
                ))

                layout_pred = {**PLOTLY_LAYOUT}
                layout_pred["legend"] = dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                fig_pred.update_layout(
                    **layout_pred,
                    scene=dict(
                        xaxis_title='Récence',
                        yaxis_title='Fréquence',
                        zaxis_title='Montant',
                        xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                        yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                        zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                    ),
                    margin=dict(l=0, r=0, b=0, t=10),
                    height=600
                )
                st.plotly_chart(fig_pred, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; opacity: 0.5; font-size: 13px;">© 2026 CRM Analytics Dashboard — Segmentation RFM</div>',
    unsafe_allow_html=True
)