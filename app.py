import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "Medical Insurance.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def build_model():
    return Pipeline([
        (
            "preprocessor",
            ColumnTransformer(
                [
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        ["sex", "smoker", "region"],
                    )
                ],
                remainder="passthrough",
            ),
        ),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ])


def main():
    st.title("Medical Insurance Cost Predictor")
    st.write("Enter patient details and click Predict to estimate insurance charges.")

    df = load_data()
    age = st.slider("Age", int(df["age"].min()), int(df["age"].max()), 34)
    bmi = st.slider("BMI", float(df["bmi"].min()), float(df["bmi"].max()), 28.0, step=0.1)
    children = st.slider("Children", int(df["children"].min()), int(df["children"].max()), 0)
    sex = st.selectbox("Sex", sorted(df["sex"].unique()))
    smoker = st.selectbox("Smoker", sorted(df["smoker"].unique()))
    region = st.selectbox("Region", sorted(df["region"].unique()))

    if st.button("Predict"):
        model = build_model()
        X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
        y = df["charges"]
        model.fit(X, y)

        input_data = pd.DataFrame([
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ])

        prediction = model.predict(input_data)[0]
        st.success(f"Estimated annual insurance charge: ${prediction:,.2f}")

    st.markdown("---")
    st.write("### Dataset preview")
    st.dataframe(df.head(10))


if __name__ == "__main__":
    main()
