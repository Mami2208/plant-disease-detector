import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import urllib.request

# ================================================================
# CONFIG PAGE
# ================================================================
st.set_page_config(
    page_title="🌿 Détection de Maladies des Plantes",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CLASSES & DONNÉES
# ================================================================
CLASS_NAMES = [
    'Bacterial Spot', 'Early Blight', 'Healthy Leaf', 'Late Blight',
    'Molds', 'Mosaic Virus', 'Septoria', 'Yellow Curl Virus'
]
CLASS_EMOJIS = ['🔴', '🟠', '🟢', '🟤', '🔵', '🟣', '⚫', '🟡']
CLASS_DESCRIPTIONS = {
    'Bacterial Spot'   : 'Maladie bactérienne causant des taches sombres sur les feuilles.',
    'Early Blight'     : 'Infection fongique précoce — taches brunes avec anneaux concentriques.',
    'Healthy Leaf'     : 'Feuille saine — aucune maladie détectée.',
    'Late Blight'      : 'Infection fongique tardive — très destructrice pour les cultures.',
    'Molds'            : 'Moisissures — champignons visibles sur la surface des feuilles.',
    'Mosaic Virus'     : 'Virus causant des motifs mosaïques sur les feuilles.',
    'Septoria'         : 'Maladie fongique — petites taches circulaires avec centre clair.',
    'Yellow Curl Virus': "Virus causant l'enroulement et le jaunissement des feuilles."
}
CLASS_COLORS = {
    'Bacterial Spot'   : '🔴',
    'Early Blight'     : '🟠',
    'Healthy Leaf'     : '🟢',
    'Late Blight'      : '🟤',
    'Molds'            : '🔵',
    'Mosaic Virus'     : '🟣',
    'Septoria'         : '⚫',
    'Yellow Curl Virus': '🟡'
}

# ================================================================
# CSS PERSONNALISÉ
# ================================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: #fff7ed;
        border-left: 4px solid #ea580c;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
    .healthy-card {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #15803d, #166534);
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# CHARGEMENT DU MODÈLE (mis en cache)
# ================================================================
@st.cache_resource
def load_model():
    """Charge le modèle YOLOv8 une seule fois et le garde en mémoire."""
    from ultralytics import YOLO

    # 📌 OPTION A — Si ton modèle est dans le repo GitHub (recommandé)
    model_path = "best.pt"

    # 📌 OPTION B — Télécharger depuis Google Drive (décommenter si besoin)
    # model_path = "best.pt"
    # if not os.path.exists(model_path):
    #     import gdown
    #     FILE_ID = "COLLE_TON_FILE_ID_ICI"
    #     gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", model_path, quiet=False)

    if not os.path.exists(model_path):
        st.error("❌ Modèle introuvable ! Lis le README pour savoir comment ajouter best.pt")
        st.stop()

    model = YOLO(model_path)
    return model

# ================================================================
# FONCTION DE PRÉDICTION
# ================================================================
def predict_disease(image_pil, confidence_threshold):
    """Lance l'inférence YOLOv8 et retourne l'image annotée + le rapport."""
    model = load_model()

    # Convertir PIL → numpy pour YOLO
    image_np = np.array(image_pil)

    results = model.predict(
        source  = image_np,
        conf    = confidence_threshold,
        iou     = 0.45,
        verbose = False
    )
    result = results[0]

    # Image annotée BGR → RGB
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    # Construire les détections
    detections = []
    if len(result.boxes) > 0:
        seen = {}
        for box in result.boxes:
            cls_id   = int(box.cls)
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'Classe {cls_id}'
            conf     = float(box.conf)
            if cls_name not in seen or seen[cls_name] < conf:
                seen[cls_name] = conf
        for cls_name, conf in seen.items():
            detections.append({"name": cls_name, "conf": conf})

    return annotated_pil, detections

# ================================================================
# EN-TÊTE
# ================================================================
st.markdown("""
<div class="main-header">
    <h1>🌿 Détection de Maladies des Plantes</h1>
    <p>Powered by YOLOv8 Fine-tuné — Master 2 GLSI, ESP UCAD</p>
</div>
""", unsafe_allow_html=True)

# ================================================================
# SIDEBAR — PARAMÈTRES & INFOS
# ================================================================
with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence_threshold = st.slider(
        "🎯 Seuil de confiance",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Plus le seuil est élevé → moins de détections mais plus précises"
    )

    st.markdown("---")
    st.header("📊 Infos Modèle")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("mAP@50", "45.1%")
        st.metric("Taille", "6.3 MB")
    with col2:
        st.metric("Classes", "8")
        st.metric("Vitesse", "~3ms")

    st.markdown("---")
    st.header("🏷️ Classes détectées")
    for name, emoji in CLASS_COLORS.items():
        st.write(f"{emoji} {name}")

# ================================================================
# CONTENU PRINCIPAL
# ================================================================
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("📤 Image d'entrée")
    uploaded_file = st.file_uploader(
        "Choisir une image de feuille",
        type=["jpg", "jpeg", "png", "webp"],
        help="Formats supportés : JPG, PNG, WEBP"
    )

    # Aperçu de l'image uploadée
    image_pil = None
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Image chargée", use_container_width=True)

    # Bouton d'analyse
    analyze_btn = st.button("🔍 Analyser la feuille", type="primary", disabled=(image_pil is None))

with col_right:
    st.subheader("📊 Résultats de la détection")

    if analyze_btn and image_pil is not None:
        with st.spinner("🔄 Analyse en cours..."):
            annotated_img, detections = predict_disease(image_pil, confidence_threshold)

        # Afficher l'image annotée
        st.image(annotated_img, caption="Image annotée avec détections", use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Rapport de diagnostic")

        if len(detections) == 0:
            st.markdown("""
            <div class="healthy-card">
                <h4>✅ Aucune maladie détectée</h4>
                <p>La feuille semble saine !</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"**🔍 {len(detections)} maladie(s) détectée(s)**")
            has_disease = False

            for det in detections:
                cls_name = det["name"]
                conf     = det["conf"]
                emoji    = CLASS_COLORS.get(cls_name, '🔵')
                desc     = CLASS_DESCRIPTIONS.get(cls_name, '')

                if cls_name != 'Healthy Leaf':
                    has_disease = True

                # Barre de progression pour la confiance
                st.markdown(f"""
                <div class="result-card">
                    <strong>{emoji} {cls_name}</strong> — Confiance : {conf*100:.1f}%<br>
                    <small><i>{desc}</i></small>
                </div>
                """, unsafe_allow_html=True)
                st.progress(conf)

            if has_disease:
                st.markdown("""
                <div class="warning-card">
                    ⚠️ <strong>Recommandation :</strong> Consultez un agronome pour un traitement adapté.
                </div>
                """, unsafe_allow_html=True)

    elif not analyze_btn:
        st.info("⬅️ Uploadez une image et cliquez sur **Analyser la feuille** pour commencer.")

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    "<center><small>🌿 Plant Disease Detector — Master 2 GLSI, ESP UCAD | YOLOv8 Fine-tuné</small></center>",
    unsafe_allow_html=True
)
