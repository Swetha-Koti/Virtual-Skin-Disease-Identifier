# =========================================================
# COMPLETE FINAL PREMIUM SKIN AI DASHBOARD
# =========================================================

# INSTALL:
# pip install streamlit tensorflow pillow numpy pandas
# pip install plotly gtts streamlit-option-menu fpdf

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime

from PIL import Image
from gtts import gTTS
from io import BytesIO
from fpdf import FPDF
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_option_menu import option_menu

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Virtual Skin Disease Identifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "show_all" not in st.session_state:
    st.session_state.show_all = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

.stApp{
    background:
    radial-gradient(circle at top left,
    #0f172a,
    #1e1b4b,
    #0f172a);
    color:white;
}

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

.hero{
    background:
    linear-gradient(
    135deg,
    rgba(91,33,182,0.45),
    rgba(30,64,175,0.25)
    );

    border-radius:30px;
    padding:40px;
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0 0 40px rgba(168,85,247,0.18);
}

.glass{

    background:
    rgba(255,255,255,0.06);

    backdrop-filter: blur(20px);

    border-radius:24px;

    border:
    1px solid rgba(255,255,255,0.08);

    padding:24px;

    margin-bottom:20px;
}

.feature{

    background:
    rgba(255,255,255,0.03);

    border-radius:20px;

    padding:22px;

    text-align:center;

    border:
    1px solid rgba(255,255,255,0.04);
}

.title{

    font-size:54px;

    font-weight:700;
}

.gradient{

    background:
    linear-gradient(
    90deg,
    #60a5fa,
    #c084fc
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;
}

.stButton>button{

    width:100%;

    border:none;

    border-radius:14px;

    background:
    linear-gradient(
    90deg,
    #2563eb,
    #9333ea
    );

    color:white;

    font-weight:600;

    padding:13px;
}

[data-testid="stFileUploader"]{

    background:
    rgba(255,255,255,0.03);

    border:
    1px dashed #8b5cf6;

    border-radius:20px;

    padding:18px;
}

h1,h2,h3,h4,h5,h6,p,span,label{
    color:white !important;
}

.metric-card{

    background:
    rgba(255,255,255,0.06);

    padding:20px;

    border-radius:20px;

    text-align:center;
}

.stProgress > div > div > div > div{
    background:
    linear-gradient(
    90deg,
    #3b82f6,
    #c084fc
    );
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_models():

    cnn_model = tf.keras.models.load_model(
        "skin_cancer_model.h5"
    )

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return cnn_model, label_encoder

cnn_model, label_encoder = load_models()

# =========================================================
# CLASS MAP
# =========================================================

class_name_mapping = {

    'bcc': 'Basal Cell Carcinoma',

    'nv': 'Melanocytic Nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign Keratosis-like Lesions',

    'vasc': 'Vascular Lesions',

    'df': 'Dermatofibroma',

    'akiec': 'Actinic Keratoses'
}

# =========================================================
# RISK
# =========================================================

def assess_risk(disease):

    if disease in [

        'Melanoma',

        'Basal Cell Carcinoma',

        'Actinic Keratoses'
    ]:

        return "HIGH"

    elif disease in [

        'Benign Keratosis-like Lesions',

        'Vascular Lesions',

        'Dermatofibroma'
    ]:

        return "MODERATE"

    else:

        return "LOW"

# =========================================================
# PRECAUTIONS
# =========================================================

precautions = {

    "Melanoma":[
        "Consult dermatologist immediately",
        "Avoid UV exposure",
        "Use SPF 50+ sunscreen",
        "Regular skin monitoring"
    ],

    "Basal Cell Carcinoma":[
        "Avoid sunlight",
        "Medical treatment required",
        "Use skin protection"
    ],

    "Melanocytic Nevi":[
        "Usually harmless",
        "Routine skin checks",
        "Monitor changes"
    ]
}

# =========================================================
# PREPROCESS
# =========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((128,128))

    img = img_to_array(image)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# =========================================================
# PREDICT
# =========================================================

def predict(image):

    processed = preprocess_image(image)

    predictions = cnn_model.predict(processed)

    pred_index = np.argmax(predictions)

    confidence = float(predictions[0][pred_index])

    class_short = label_encoder.inverse_transform(
        [pred_index]
    )[0]

    disease = class_name_mapping.get(
        class_short,
        class_short
    )

    risk = assess_risk(disease)

    probs = predictions[0]

    return disease, confidence, risk, probs

# =========================================================
# PDF
# =========================================================

def create_pdf(disease, confidence, risk):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=18)

    pdf.cell(200,10,txt="Skin AI Report",ln=True)

    pdf.set_font("Arial", size=14)

    pdf.cell(
        200,
        10,
        txt=f"Disease: {disease}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Confidence: {confidence:.2%}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Risk Level: {risk}",
        ln=True
    )

    filename = "report.pdf"

    pdf.output(filename)

    return filename

# =========================================================
# VOICE
# =========================================================

def text_to_speech(text):

    tts = gTTS(text)

    fp = BytesIO()

    tts.write_to_fp(fp)

    st.audio(fp)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    # 🩺 Skin AI
    ### Dermatology Assistant
    """)

    selected = option_menu(

        menu_title=None,

        options=[
            "Home",
            "Prediction",
            "History",
            "Knowledge Base",
            "Precautions",
            "Analytics",
            "About"
        ],

        icons=[
            "house",
            "activity",
            "clock-history",
            "book",
            "shield-check",
            "bar-chart",
            "info-circle"
        ],

        default_index=0
    )

# =========================================================
# HOME
# =========================================================

if selected == "Home":

    st.markdown("""
    <div class="hero">

    <div class="title">
    Welcome to
    <span class="gradient">
    Skin AI
    </span>
    👋
    </div>

    <p>
    AI Powered Skin Disease Detection & Risk Analysis
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)

    with c1:
        if st.button("🧠 Deep Learning"):
            st.info(
                "CNN Deep Learning model trained on HAM10000 dataset."
            )

    with c2:
        if st.button("⚡ Fast Prediction"):
            st.info(
                "Predicts diseases instantly using AI."
            )

    with c3:
        if st.button("🛡️ Risk Assessment"):
            st.info(
                "Classifies diseases into LOW/MODERATE/HIGH risk."
            )

    with c4:
        if st.button("📄 PDF Reports"):
            st.info(
                "Generate downloadable medical reports."
            )

    with c5:
        if st.button("🔊 Voice Assistant"):
            st.info(
                "Reads prediction results aloud."
            )

    st.markdown("<br>", unsafe_allow_html=True)

    q1,q2,q3,q4 = st.columns(4)

    stats = [

        ("7+","Skin Diseases"),

        ("73.6%","Accuracy"),

        ("10K+","Images"),

        ("24/7","AI Support")
    ]

    for col,stat in zip(
        [q1,q2,q3,q4],
        stats
    ):

        with col:

            st.markdown(f"""
            <div class="metric-card">
            <h1>{stat[0]}</h1>
            <p>{stat[1]}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">

    <h2>🩺 About The System</h2>

    <p>
    Virtual Skin Disease Identifier is an AI based
    healthcare assistant that helps users identify
    possible skin diseases using Deep Learning.
    </p>

    <p>
    The system uses Convolutional Neural Networks
    trained on HAM10000 dataset for prediction.
    </p>

    <p>
    Features include:
    </p>

    <ul>
    <li>Real-time Prediction</li>
    <li>Risk Assessment</li>
    <li>Voice Assistant</li>
    <li>Prediction Analytics</li>
    <li>Medical PDF Reports</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PREDICTION TAB
# =========================================================

elif selected == "Prediction":

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown("""
    <div class="hero">

    <div class="title">
    Skin Disease
    <span class="gradient">
    Prediction
    </span>
    </div>

    <p>
    Upload or Capture Skin Image for AI Analysis
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # CUSTOM CSS
    # =====================================================

    st.markdown("""
    <style>

    .upload-box{

        background:
        linear-gradient(
        135deg,
        rgba(91,33,182,0.25),
        rgba(30,64,175,0.12)
        );

        border:
        1px solid rgba(168,85,247,0.4);

        border-radius:25px;

        padding:35px;

        text-align:center;

        box-shadow:
        0 0 25px rgba(168,85,247,0.18);

        margin-bottom:20px;
    }

    .upload-title{

        font-size:26px;

        font-weight:600;

        margin-bottom:15px;
    }

    .upload-sub{

        color:#cbd5e1;

        font-size:14px;
    }

    .upload-icon{

        font-size:75px;

        margin-bottom:10px;
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # MAIN CONTAINER
    # =====================================================

    st.markdown("""
    <div class="glass">

    <h2>
    📤 Upload or Capture Skin Image
    </h2>

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2 = st.tabs([
        "📁 Upload Image",
        "📷 Camera Capture"
    ])

    image = None

    # =====================================================
    # UPLOAD
    # =====================================================

    with tab1:

        st.markdown("""
        <div class="upload-box">

        <div class="upload-icon">
        ☁️
        </div>

        <div class="upload-title">
        Drag & Drop Your Image Here
        </div>

        <div class="upload-sub">
        Supports JPG, JPEG, PNG (Max 10MB)
        </div>

        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "",
            type=["jpg","jpeg","png"],
            label_visibility="collapsed"
        )

        if uploaded_file:

            image = Image.open(uploaded_file)

    # =====================================================
    # CAMERA
    # =====================================================

    with tab2:

        camera_image = st.camera_input(
            "Capture Skin Image"
        )

        if camera_image:

            image = Image.open(camera_image)

    # =====================================================
    # IF IMAGE
    # =====================================================

    if image:

        disease, confidence, risk, probs = predict(image)

        current_prediction = f"{disease}_{confidence:.2f}"

        # STORE ONLY ONCE

        if st.session_state.last_prediction != current_prediction:

            st.session_state.history.append({

                "Disease": disease,

                "Confidence": f"{confidence:.2%}",

                "Risk": risk,

                "Time":
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                )
            })

            st.session_state.last_prediction = current_prediction

        st.markdown("<br>", unsafe_allow_html=True)

        # =================================================
        # RESULT SECTION
        # =================================================

        c1,c2,c3 = st.columns([1.1,1,1])

        # =================================================
        # LEFT RESULT
        # =================================================

        with c1:

            st.markdown("""
            <div class="glass">
            """, unsafe_allow_html=True)

            st.subheader("🧬 Prediction Result")

            st.write(f"# {disease}")

            st.write(
                f"## Confidence: {confidence:.2%}"
            )

            # RISK BADGE

            if risk == "HIGH":

                st.error("HIGH RISK")

            elif risk == "MODERATE":

                st.warning("MODERATE RISK")

            else:

                st.success("LOW RISK")

            # VOICE

            if st.button("🔊 Speak Result"):

                speech = f"""
                The predicted disease is {disease}.
                Risk level is {risk}
                """

                text_to_speech(speech)

            # PDF

            pdf_file = create_pdf(
                disease,
                confidence,
                risk
            )

            with open(pdf_file, "rb") as f:

                st.download_button(
                    "📄 Download PDF Report",
                    data=f,
                    file_name="skin_report.pdf",
                    mime="application/pdf"
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # =================================================
        # CENTER RISK METER
        # =================================================

        with c2:

            risk_value = {

                "LOW": 30,

                "MODERATE": 65,

                "HIGH": 95
            }

            fig2 = go.Figure(go.Indicator(

                mode="gauge+number",

                value=risk_value[risk],

                gauge={

                    'axis': {'range': [0,100]},

                    'bar': {'color': "#a855f7"}
                }
            ))

            fig2.update_layout(

                paper_bgcolor="rgba(0,0,0,0)",

                font={'color':'white'}
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        # =================================================
        # RIGHT IMAGE
        # =================================================

        with c3:

            st.markdown("""
            <div class="glass">
            """, unsafe_allow_html=True)

            st.image(
                image,
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # =================================================
        # TOP 3 + OVERVIEW
        # =================================================

        st.markdown("<br>", unsafe_allow_html=True)

        left,right = st.columns([1,1.3])

        # =================================================
        # TOP 3
        # =================================================

        with left:

            st.markdown("""
            <div class="glass">
            """, unsafe_allow_html=True)

            st.subheader("🏆 Top 3 Predictions")

            top3_idx = np.argsort(probs)[::-1][:3]

            for idx in top3_idx:

                name = class_name_mapping.get(
                    label_encoder.inverse_transform(
                        [idx]
                    )[0]
                )

                prob = probs[idx]

                st.write(f"### {name}")

                st.progress(float(prob))

                st.write(f"{prob:.2%}")

            # BUTTON

            if st.button("📊 View All Predictions"):

                st.session_state.show_all = (
                    not st.session_state.show_all
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # =================================================
        # OVERVIEW
        # =================================================

        with right:

            st.markdown("""
            <div class="glass">

            <h2>
            📈 Prediction Overview
            </h2>

            <p>
            Click 'View All Predictions'
            to display all 7 disease
            probability graph.
            </p>

            </div>
            """, unsafe_allow_html=True)

        # =================================================
        # SHOW ALL GRAPH ONLY AFTER CLICK
        # =================================================

        if st.session_state.show_all:

            class_labels = [

                class_name_mapping.get(

                    label_encoder.inverse_transform(
                        [i]
                    )[0]
                )

                for i in range(len(probs))
            ]

            all_df = pd.DataFrame({

                "Disease": class_labels,

                "Probability": [

                    round(float(p)*100,2)

                    for p in probs
                ]
            })

            st.markdown("""
            <div class="glass">
            """, unsafe_allow_html=True)

            st.subheader(
                "📊 All 7 Disease Probabilities"
            )

            fig_all = px.bar(

                all_df,

                x="Disease",

                y="Probability",

                text_auto=True
            )

            fig_all.update_layout(

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                font_color="white"
            )

            st.plotly_chart(
                fig_all,
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # =================================================
        # PRECAUTIONS
        # =================================================

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass">

        <h2>
        🩺 Precautions & Recommendations
        </h2>

        </div>
        """, unsafe_allow_html=True)

        p1,p2,p3,p4 = st.columns(4)

        with p1:

            st.markdown("""
            <div class="glass">

            <h3>👨‍⚕️ Consult Doctor</h3>

            <p>
            Seek professional medical advice regularly.
            </p>

            </div>
            """, unsafe_allow_html=True)

        with p2:

            st.markdown("""
            <div class="glass">

            <h3>☀️ Sun Protection</h3>

            <p>
            Use SPF 50+ sunscreen whenever outdoors.
            </p>

            </div>
            """, unsafe_allow_html=True)

        with p3:

            st.markdown("""
            <div class="glass">

            <h3>🧴 Avoid UV Exposure</h3>

            <p>
            Stay away from direct sunlight exposure.
            </p>

            </div>
            """, unsafe_allow_html=True)

        with p4:

            st.markdown("""
            <div class="glass">

            <h3>🔍 Regular Checkups</h3>

            <p>
            Monitor skin changes frequently.
            </p>

            </div>
            """, unsafe_allow_html=True)
# =========================================================
# HISTORY
# =========================================================

elif selected == "History":

    st.markdown("""
    # 📜 Prediction History
    """)

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

# =========================================================
# KNOWLEDGE BASE
# =========================================================

elif selected == "Knowledge Base":

    # =====================================================
    # CUSTOM CSS
    # =====================================================

    st.markdown("""
    <style>

    .disease-card{

        background:
        linear-gradient(
        135deg,
        rgba(91,33,182,0.25),
        rgba(30,64,175,0.15)
        );

        border:
        1px solid rgba(168,85,247,0.35);

        border-radius:22px;

        padding:22px;

        text-align:center;

        transition:0.4s;

        cursor:pointer;

        box-shadow:
        0 0 18px rgba(168,85,247,0.12);
    }

    .disease-card:hover{

        transform:
        translateY(-8px) scale(1.02);

        box-shadow:
        0 0 30px rgba(168,85,247,0.35);

        border:
        1px solid rgba(192,132,252,0.7);
    }

    .disease-title{

        font-size:22px;

        font-weight:700;

        margin-top:10px;
    }

    .info-box{

        animation: fadeIn 0.6s ease-in-out;

        background:
        rgba(255,255,255,0.06);

        backdrop-filter: blur(18px);

        border-radius:24px;

        padding:30px;

        border:
        1px solid rgba(255,255,255,0.08);

        margin-top:20px;
    }

    @keyframes fadeIn{

        from{
            opacity:0;
            transform:translateY(20px);
        }

        to{
            opacity:1;
            transform:translateY(0px);
        }
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown("""
    <div class="hero">

    <div class="title">
    Skin Disease
    <span class="gradient">
    Knowledge Base
    </span>
    </div>

    <p>
    Learn about different skin diseases,
    symptoms, causes, precautions and treatment.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # DISEASE DATA
    # =====================================================

    disease_data = {

        "Melanoma":{

            "emoji":"⚠️",

            "about":
            "Melanoma is one of the most dangerous forms of skin cancer that develops in melanocytes.",

            "symptoms":
            """
            • Irregular mole shape  
            • Dark black/brown spots  
            • Bleeding or itching  
            • Sudden mole growth
            """,

            "causes":
            """
            • Excess UV exposure  
            • Genetic factors  
            • Fair skin tone
            """,

            "precautions":
            """
            • Use sunscreen daily  
            • Avoid direct sunlight  
            • Monitor skin changes
            """,

            "treatment":
            """
            • Surgery  
            • Immunotherapy  
            • Radiation therapy
            """
        },

        "Basal Cell Carcinoma":{

            "emoji":"🧬",

            "about":
            "Basal Cell Carcinoma is the most common skin cancer caused by prolonged sunlight exposure.",

            "symptoms":
            """
            • Pearly bumps  
            • Red patches  
            • Open sores
            """,

            "causes":
            """
            • UV radiation  
            • Sunburn history  
            • Aging
            """,

            "precautions":
            """
            • Wear protective clothing  
            • Avoid tanning  
            • Regular skin exams
            """,

            "treatment":
            """
            • Cryotherapy  
            • Surgery  
            • Topical medication
            """
        },

        "Melanocytic Nevi":{

            "emoji":"🟤",

            "about":
            "Melanocytic Nevi are common moles usually harmless but should be monitored.",

            "symptoms":
            """
            • Brown circular spots  
            • Flat or raised moles  
            • Uniform color
            """,

            "causes":
            """
            • Genetics  
            • Sun exposure
            """,

            "precautions":
            """
            • Observe color changes  
            • Avoid scratching
            """,

            "treatment":
            """
            • Usually no treatment needed  
            • Mole removal if suspicious
            """
        },

        "Benign Keratosis-like Lesions":{

            "emoji":"🩹",

            "about":
            "Non-cancerous skin growths appearing with age.",

            "symptoms":
            """
            • Rough patches  
            • Waxy texture  
            • Brown lesions
            """,

            "causes":
            """
            • Aging  
            • Genetics
            """,

            "precautions":
            """
            • Avoid irritation  
            • Maintain skin hygiene
            """,

            "treatment":
            """
            • Laser therapy  
            • Cryotherapy
            """
        },

        "Vascular Lesions":{

            "emoji":"🩸",

            "about":
            "Abnormal blood vessel growths appearing red or purple.",

            "symptoms":
            """
            • Red spots  
            • Visible blood vessels
            """,

            "causes":
            """
            • Congenital conditions  
            • Skin trauma
            """,

            "precautions":
            """
            • Avoid injury  
            • Monitor bleeding
            """,

            "treatment":
            """
            • Laser treatment  
            • Surgery
            """
        },

        "Dermatofibroma":{

            "emoji":"🔵",

            "about":
            "Small benign skin nodules usually harmless.",

            "symptoms":
            """
            • Hard bumps  
            • Brown/red nodules
            """,

            "causes":
            """
            • Minor skin injuries  
            • Insect bites
            """,

            "precautions":
            """
            • Avoid scratching  
            • Monitor growth
            """,

            "treatment":
            """
            • Surgical removal if needed
            """
        },

        "Actinic Keratoses":{

            "emoji":"☀️",

            "about":
            "Precancerous rough skin patches caused by UV exposure.",

            "symptoms":
            """
            • Dry rough patches  
            • Scaly lesions
            """,

            "causes":
            """
            • Chronic sun exposure  
            • Aging
            """,

            "precautions":
            """
            • Use SPF sunscreen  
            • Avoid UV rays
            """,

            "treatment":
            """
            • Cryotherapy  
            • Topical creams
            """
        }
    }

    # =====================================================
    # BUTTON NAVIGATION
    # =====================================================

    if "selected_disease" not in st.session_state:
        st.session_state.selected_disease = "Melanoma"

    cols = st.columns(4)

    diseases = list(disease_data.keys())

    for i, disease in enumerate(diseases):

        with cols[i % 4]:

            if st.button(
                f"{disease_data[disease]['emoji']} {disease}"
            ):

                st.session_state.selected_disease = disease

    # =====================================================
    # DISPLAY INFO
    # =====================================================

    selected_disease = st.session_state.selected_disease

    data = disease_data[selected_disease]

    st.markdown(f"""
    <div class="info-box">

    <h1>
    {data['emoji']} {selected_disease}
    </h1>

    <hr>

    <h3>📖 About Disease</h3>

    <p>
    {data['about']}
    </p>

    <h3>⚠️ Symptoms</h3>

    <p>
    {data['symptoms']}
    </p>

    <h3>🧬 Causes</h3>

    <p>
    {data['causes']}
    </p>

    <h3>🩺 Precautions</h3>

    <p>
    {data['precautions']}
    </p>

    <h3>💊 Treatment</h3>

    <p>
    {data['treatment']}
    </p>

    </div>
    """, unsafe_allow_html=True)
# =========================================================
# PRECAUTIONS
# =========================================================

elif selected == "Precautions":

    st.markdown("""
    <div class="hero">

    <div class="title">
    Precautions &
    <span class="gradient">
    Recommendations
    </span>
    </div>

    <p>
    Important skincare precautions for healthy skin.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # ROW 1
    # =====================================================

    c1,c2,c3,c4 = st.columns(4)

    with c1:

        st.markdown("""
        <div class="glass">

        <h3>👨‍⚕️ Consult Doctor</h3>

        <p>
        Seek professional medical advice regularly.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="glass">

        <h3>☀️ Sun Protection</h3>

        <p>
        Use SPF 50+ sunscreen whenever outdoors.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="glass">

        <h3>🧴 Avoid UV Exposure</h3>

        <p>
        Stay away from direct sunlight exposure.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c4:

        st.markdown("""
        <div class="glass">

        <h3>🔍 Regular Checkups</h3>

        <p>
        Monitor skin changes frequently.
        </p>

        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # ROW 2
    # =====================================================

    c5,c6,c7,c8 = st.columns(4)

    with c5:

        st.markdown("""
        <div class="glass">

        <h3>💧 Hydration</h3>

        <p>
        Drink enough water for healthy skin.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c6:

        st.markdown("""
        <div class="glass">

        <h3>🥗 Healthy Diet</h3>

        <p>
        Eat fruits and vitamin-rich foods daily.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c7:

        st.markdown("""
        <div class="glass">

        <h3>🧼 Skin Hygiene</h3>

        <p>
        Maintain proper skincare hygiene routines.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c8:

        st.markdown("""
        <div class="glass">

        <h3>🛌 Proper Sleep</h3>

        <p>
        Get enough sleep for skin recovery.
        </p>

        </div>
        """, unsafe_allow_html=True)
# =========================================================
# ANALYTICS
# =========================================================

elif selected == "Analytics":

    st.markdown("""
    # 📊 AI Analytics
    """)

    col1,col2 = st.columns(2)

    with col1:

        fig = px.pie(

            values=[73.6,26.4],

            names=[
                "Correct",
                "Incorrect"
            ],

            title="Model Accuracy"
        )

        fig.update_layout(

            paper_bgcolor="rgba(0,0,0,0)",

            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig2 = px.bar(

            x=[
                "Melanoma",
                "BCC",
                "Nevi"
            ],

            y=[92,80,76],

            title="Top Predictions"
        )

        fig2.update_layout(

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            font_color="white"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# =========================================================
# ABOUT
# =========================================================

elif selected == "About":

    st.markdown("""
    # 🧠 About Project

    Virtual Skin Disease Identifier uses
    Deep Learning and CNN models trained
    on HAM10000 dataset.

    ### Features

    - Skin Disease Classification
    - Risk Analysis
    - PDF Reports
    - Voice Assistant
    - Camera Detection
    - Real-time AI Predictions

    ⚠️ This is not a substitute for
    professional medical advice.
    """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<hr>

<center>

Made with ❤️ using Streamlit & Deep Learning

</center>
""", unsafe_allow_html=True)