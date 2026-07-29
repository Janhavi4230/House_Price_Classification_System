import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Smart ML Prediction System",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:2rem;
max-width:900px;
}

.stApp{
background:#0E1117;
}

.title{
font-size:56px;
font-weight:800;
color:white;
margin-bottom:0px;
}

.subtitle{
font-size:18px;
color:#A7A7A7;
margin-bottom:35px;
}

hr{
border:1px solid #252525;
}

div[data-baseweb="select"] > div{
background:#1D1F2B;
border-radius:12px;
border:none;
}

div[data-baseweb="input"] > div{
background:#1D1F2B;
border-radius:12px;
}

.stNumberInput input{
background:#1D1F2B;
color:white;
}

.stTextInput input{
background:#1D1F2B;
color:white;
}

.stButton button{
width:100%;
height:58px;
background:#2563EB;
color:white;
font-size:18px;
font-weight:700;
border-radius:14px;
border:none;
transition:.3s;
}

.stButton button:hover{
background:#1D4ED8;
transform:scale(1.02);
}

.result{
background:#171923;
padding:25px;
border-radius:15px;
border-left:6px solid #2563EB;
margin-top:25px;
}

</style>
""",unsafe_allow_html=True)

st.markdown(
'<p class="title">🤖 Smart ML Prediction System</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Predict using Machine Learning Models with Professional Dashboard</p>',
unsafe_allow_html=True
)

model=st.selectbox(
"Select Machine Learning Model",
[
"Linear Regression",
"Logistic Regression",
"KNN",
"Naive Bayes"
]
)

st.markdown("---")
# ==============================
# LINEAR REGRESSION PAGE
# ==============================

if model == "Linear Regression":

    st.subheader("🏡 California House Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        medinc = st.number_input(
            "Median Income",
            min_value=0.0,
            value=3.5,
            format="%.2f"
        )

        houseage = st.number_input(
            "House Age",
            min_value=1,
            value=20
        )

        averooms = st.number_input(
            "Average Rooms",
            min_value=1.0,
            value=5.5,
            format="%.2f"
        )

        avebed = st.number_input(
            "Average Bedrooms",
            min_value=0.5,
            value=1.0,
            format="%.2f"
        )

    with col2:

        population = st.number_input(
            "Population",
            min_value=1,
            value=1000
        )

        aveoccup = st.number_input(
            "Average Occupancy",
            min_value=1.0,
            value=3.0,
            format="%.2f"
        )

        latitude = st.number_input(
            "Latitude",
            value=34.00,
            format="%.2f"
        )

        longitude = st.number_input(
            "Longitude",
            value=-118.00,
            format="%.2f"
        )

    st.write("")

    predict = st.button("🚀 Predict House Price")
# ============================
# LOAD MODEL & PREDICT
# ============================

if predict:

    try:
        # Load trained model
        model_lr = joblib.load("models/linear_regression_model.pkl")

        # Prepare input
        input_data = np.array([[
            medinc,
            houseage,
            averooms,
            avebed,
            population,
            aveoccup,
            latitude,
            longitude
        ]])

        # Prediction
        prediction = model_lr.predict(input_data)[0]

        st.markdown(f"""
        <div class="result">
            <h2 style="color:#4CAF50;">✅ Prediction Successful</h2>
            <hr>
            <h3 style="color:white;">
                Estimated House Price:
            </h3>

            <h1 style="
                color:#00E676;
                font-size:42px;
                font-weight:bold;">
                ${prediction*100000:,.2f}
            </h1>

            <p style="color:#CFCFCF;">
                Prediction generated using Linear Regression Model.
            </p>

        </div>
        """, unsafe_allow_html=True)

    except Exception as e:

        st.error(f"Error : {e}")
        