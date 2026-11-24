
import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Mumbai House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL AND DATASET OPTIONS
# --------------------------------------------------

MODEL_PATH = "models/mumbai_house_price_knn.pkl"
DATA_PATH = "Mumbai House Prices.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset_options():
    data = pd.read_csv(DATA_PATH)

    return {
        "localities": sorted(
            data["locality"].dropna().astype(str).unique().tolist()
        ),
        "regions": sorted(
            data["region"].dropna().astype(str).unique().tolist()
        )
    }


try:
    model = load_model()
    options = load_dataset_options()

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏠 Mumbai House Price Prediction")

st.markdown("""
Predict the estimated price of a residential property
in Mumbai using Machine Learning.

Enter the property details below to get a price prediction.
""")

st.divider()


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.subheader("Enter Property Details")

with st.form("house_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        bhk = st.selectbox(
            "Number of Bedrooms (BHK)",
            options=list(range(1, 11))
        )

        property_type = st.selectbox(
            "Property Type",
            options=[
                "Apartment",
                "Villa",
                "Studio Apartment",
                "Independent House",
                "Penthouse"
            ]
        )

        locality = st.selectbox(
            "Locality",
            options=options["localities"]
        )

        area = st.number_input(
            "Area",
            min_value=127,
            max_value=16000,
            value=685,
            step=10,
            help="Enter area in the same units used in the training dataset."
        )

    with col2:

        region = st.selectbox(
            "Region",
            options=options["regions"]
        )

        status = st.selectbox(
            "Construction Status",
            options=[
                "Ready to move",
                "Under Construction"
            ]
        )

        age = st.selectbox(
            "Property Age",
            options=[
                "New",
                "Resale",
                "Unknown"
            ]
        )

    predict_button = st.form_submit_button(
        "Predict House Price",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    input_data = pd.DataFrame(
        [[
            bhk,
            property_type,
            locality,
            area,
            region,
            status,
            age
        ]],
        columns=[
            "bhk",
            "type",
            "locality",
            "area",
            "region",
            "status",
            "age"
        ]
    )

    try:

        prediction = model.predict(input_data)[0]

        price_lakhs = float(prediction)
        price_crores = price_lakhs / 100

        st.divider()

        st.subheader("Predicted House Price")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Estimated Price (Lakhs)",
                f"₹ {price_lakhs:,.2f} Lakhs"
            )

        with col2:
            st.metric(
                "Estimated Price (Crores)",
                f"₹ {price_crores:,.2f} Crores"
            )

        st.success("House price prediction generated successfully!")

        # Property summary

        st.subheader("Property Summary")

        summary = pd.DataFrame({
            "Feature": [
                "BHK",
                "Property Type",
                "Locality",
                "Area",
                "Region",
                "Construction Status",
                "Property Age"
            ],
            "Selected Value": [
                bhk,
                property_type,
                locality,
                area,
                region,
                status,
                age
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "This is an estimated price based on historical "
            "dataset patterns, not an official property valuation."
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("About the Machine Learning Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algorithm", "KNN Regression")

with col2:
    st.metric("Optimal K", "3")

with col3:
    st.metric("Test R² Score", "81.09%")

st.markdown("""
### Model Evaluation Results

- MAE: 31.73 Lakhs
- MSE: 9593.19
- RMSE: 97.94 Lakhs
- R² Score: 0.8109
""")

st.divider()

st.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)
