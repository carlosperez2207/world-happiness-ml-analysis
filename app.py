# app.py
# Dashboard interactivo en Streamlit para el World Happiness Report 2019
# Incluye regiones como filtro y en las visualizaciones.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# -----------------------------------------------------------
# Funciones auxiliares para REGIONES (mismas que en análisis)
# -----------------------------------------------------------

def _assign_region(country: str) -> str:
    europe = {
        "Finland", "Denmark", "Norway", "Iceland", "Netherlands", "Switzerland",
        "Sweden", "Luxembourg", "Ireland", "Germany", "Belgium", "United Kingdom",
        "UK", "France", "Austria", "Spain", "Italy", "Portugal", "Czech Republic",
        "Czechia", "Slovakia", "Slovenia", "Poland", "Hungary", "Greece",
        "Estonia", "Latvia", "Lithuania", "Croatia", "Romania", "Bulgaria",
        "Serbia", "Montenegro", "North Macedonia", "Albania", "Bosnia and Herzegovina",
        "Russia", "Ukraine", "Belarus", "Moldova", "Cyprus", "Malta"
    }

    north_america = {
        "United States", "United States of America", "USA",
        "Canada"
    }

    latin_america = {
        "Mexico", "Costa Rica", "Panama", "Guatemala", "El Salvador", "Honduras",
        "Nicaragua", "Belize", "Dominican Republic", "Haiti", "Cuba", "Jamaica",
        "Trinidad and Tobago", "Bahamas", "Barbados",
        "Brazil", "Argentina", "Chile", "Uruguay", "Paraguay", "Bolivia",
        "Peru", "Colombia", "Ecuador", "Venezuela"
    }

    asia_pacific = {
        "New Zealand", "Australia", "Singapore", "Japan", "South Korea",
        "Taiwan", "Hong Kong", "China", "India", "Pakistan", "Bangladesh",
        "Sri Lanka", "Thailand", "Vietnam", "Philippines", "Malaysia",
        "Indonesia", "Laos", "Cambodia", "Myanmar", "Mongolia", "Nepal"
    }

    mena = {  # Middle East and North Africa
        "Israel", "United Arab Emirates", "UAE", "Saudi Arabia", "Qatar",
        "Kuwait", "Bahrain", "Oman", "Jordan", "Lebanon",
        "Morocco", "Tunisia", "Algeria", "Libya", "Egypt", "Iran", "Iraq",
    }

    subsaharan_africa = {
        "South Africa", "Namibia", "Botswana", "Zimbabwe", "Zambia",
        "Mozambique", "Angola", "Nigeria", "Ghana", "Kenya", "Tanzania",
        "Uganda", "Rwanda", "Burundi", "Ethiopia", "Somalia", "Madagascar",
        "Senegal", "Mali", "Niger", "Chad", "Cameroon", "Benin",
        "Burkina Faso", "Sierra Leone", "Liberia", "Togo", "Guinea",
        "Mauritania", "Malawi", "Lesotho", "Eswatini", "Gabon",
        "Congo (Brazzaville)", "Congo (Kinshasa)", "Central African Republic"
    }

    c = (country or "").strip()

    if c in europe:
        return "Europe"
    if c in north_america:
        return "North America"
    if c in latin_america:
        return "Latin America & Caribbean"
    if c in asia_pacific:
        return "Asia-Pacific"
    if c in mena:
        return "Middle East & North Africa"
    if c in subsaharan_africa:
        return "Sub-Saharan Africa"

    return "Other"


def add_region_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Region" in df.columns:
        df["Region"] = df["Region"].fillna("Other")
        return df

    if "Country" not in df.columns:
        raise ValueError("No se encuentra la columna 'Country' para generar 'Region'.")

    df["Region"] = df["Country"].apply(_assign_region)
    return df


# -----------------------------------------------------------
# Datos y modelos
# -----------------------------------------------------------

@st.cache_data
def load_data(path="2019.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Country or region": "Country",
        "Score": "HappinessScore",
        "GDP per capita": "GDP_per_capita",
        "Social support": "Social_support",
        "Healthy life expectancy": "Healthy_life_expectancy",
        "Freedom to make life choices": "Freedom",
        "Generosity": "Generosity",
        "Perceptions of corruption": "Corruption"
    })
    df = add_region_column(df)
    return df


@st.cache_data
def prepare_data(df):
    features = [
        "GDP_per_capita",
        "Social_support",
        "Healthy_life_expectancy",
        "Freedom",
        "Generosity",
        "Corruption"
    ]
    X = df[features].copy()
    y = df["HappinessScore"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (X, y,
            X_train, X_test,
            y_train, y_test,
            X_train_scaled, X_test_scaled,
            scaler, features)


def train_all_models(X_train_scaled, y_train,
                     n_estimators_rf=100,
                     max_depth_rf=None,
                     n_estimators_gb=100,
                     learning_rate_gb=0.1):
    models = {}

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models["Linear Regression"] = lr

    rf = RandomForestRegressor(
        n_estimators=n_estimators_rf,
        max_depth=max_depth_rf,
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    models["Random Forest"] = rf

    gb = GradientBoostingRegressor(
        n_estimators=n_estimators_gb,
        learning_rate=learning_rate_gb,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    models["Gradient Boosting"] = gb

    return models


def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return r2, rmse, mae, y_pred


def evaluate_all_models(models, X_test_scaled, y_test):
    results = {}
    for name, model in models.items():
        r2, rmse, mae, _ = evaluate_model(model, X_test_scaled, y_test)
        results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        return results


# -----------------------------------------------------------
# Funciones de gráficas (usan región y filtro)
# -----------------------------------------------------------

def plot_corr_heatmap(df, features):
    plt.figure(figsize=(9, 7))
    cols = ["HappinessScore"] + features
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlación")
    st.pyplot(plt.gcf())
    plt.close()


def plot_gdp_vs_happiness(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="GDP_per_capita",
        y="HappinessScore",
        hue="Region"
    )
    plt.xlabel("PIB per cápita")
    plt.ylabel("Índice de felicidad")
    plt.title("PIB per cápita vs Felicidad (por región)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(plt.gcf())
    plt.close()


def plot_top_countries(df, top_n=10):
    top = df.sort_values("HappinessScore", ascending=False).head(top_n)
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=top,
        x="HappinessScore",
        y="Country",
        hue="Region",
        dodge=False
    )
    plt.xlabel("Índice de felicidad")
    plt.ylabel("País")
    plt.title(f"Top {top_n} países más felices")
    plt.legend(title="Región", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(plt.gcf())
    plt.close()


def plot_happiness_by_region(df):
    plt.figure(figsize=(10, 6))
    order = df.groupby("Region")["HappinessScore"].mean().sort_values(ascending=False).index
    sns.boxplot(
        data=df,
        x="Region",
        y="HappinessScore",
        order=order
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Región")
    plt.ylabel("Índice de felicidad")
    plt.title("Distribución del índice de felicidad por región")
    st.pyplot(plt.gcf())
    plt.close()


def plot_models_comparison(results):
    models = list(results.keys())
    r2_values = [results[m]["R2"] for m in models]

    plt.figure(figsize=(7, 5))
    sns.barplot(x=models, y=r2_values)
    plt.ylabel("R²")
    plt.title("Comparación de modelos (R²)")
    plt.ylim(0, 1)
    st.pyplot(plt.gcf())
    plt.close()


# -----------------------------------------------------------
# App principal
# -----------------------------------------------------------

def main():
    st.set_page_config(
        page_title="World Happiness Report 2019",
        layout="wide"
    )

    st.title("Factores socioeconómicos y felicidad de los países")
    st.caption("Dashboard interactivo - World Happiness Report 2019")

    df = load_data()
    (X, y,
     X_train, X_test,
     y_train, y_test,
     X_train_scaled, X_test_scaled,
     scaler, features) = prepare_data(df)

    # -------- Sidebar: modelo, hiperparámetros y REGIONES --------
    st.sidebar.header("Configuración")

    modelo_seleccionado = st.sidebar.selectbox(
        "Modelo",
        ("Linear Regression", "Random Forest", "Gradient Boosting")
    )

    st.sidebar.subheader("Hiperparámetros")

    n_estimators_rf = st.sidebar.slider(
        "n_estimators (Random Forest)",
        min_value=50, max_value=300, value=100, step=10
    )
    max_depth_rf = st.sidebar.slider(
        "max_depth (Random Forest)",
        min_value=1, max_value=15, value=5, step=1
    )
    max_depth_rf = int(max_depth_rf)

    n_estimators_gb = st.sidebar.slider(
        "n_estimators (Gradient Boosting)",
        min_value=50, max_value=300, value=100, step=10
    )
    learning_rate_gb = st.sidebar.slider(
        "learning_rate (Gradient Boosting)",
        min_value=0.01, max_value=0.5, value=0.1, step=0.01
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtro por región")
    regiones_disponibles = sorted(df["Region"].unique())
    regiones_seleccionadas = st.sidebar.multiselect(
        "Regiones a mostrar",
        regiones_disponibles,
        default=regiones_disponibles
    )

    # Dataset filtrado solo para visualizaciones y selección de país
    df_filtered = df[df["Region"].isin(regiones_seleccionadas)].copy()
    if df_filtered.empty:
        st.warning("No hay países en las regiones seleccionadas. Ajusta el filtro de región.")
        df_filtered = df.copy()

    st.sidebar.markdown("---")
    modo_prediccion = st.sidebar.radio(
        "Modo de predicción",
        ("Seleccionar país", "Valores personalizados")
    )

    # Entrenar todos los modelos con los hiperparámetros actuales
    models = train_all_models(
        X_train_scaled, y_train,
        n_estimators_rf=n_estimators_rf,
        max_depth_rf=max_depth_rf,
        n_estimators_gb=n_estimators_gb,
        learning_rate_gb=learning_rate_gb
    )

    # Evaluar todos los modelos
    results = {}
    for name, model in models.items():
        r2, rmse, mae, _ = evaluate_model(model, X_test_scaled, y_test)
        results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}

    # Modelo actual
    current_model = models[modelo_seleccionado]
    r2, rmse, mae, y_pred = evaluate_model(current_model, X_test_scaled, y_test)

    # -------- Layout principal --------
    tab1, tab2, tab3 = st.tabs([
        "Vista general y KPIs",
        "Visualizaciones",
        "Predicciones"
    ])

    # ---- Tab 1: KPIs + dataset ----
    with tab1:
        st.subheader("KPIs del modelo seleccionado")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Modelo", modelo_seleccionado)
        with col2:
            st.metric("R²", f"{r2:.3f}")
        with col3:
            st.metric("RMSE", f"{rmse:.3f}")
        with col4:
            st.metric("MAE", f"{mae:.3f}")

        st.markdown("---")
        st.subheader("Vista del dataset (filtrado por región)")
        st.dataframe(df_filtered)

    # ---- Tab 2: Visualizaciones ----
    with tab2:
        st.subheader("Análisis exploratorio de datos (EDA)")

        st.markdown("### 1. Matriz de correlación")
        plot_corr_heatmap(df_filtered, features)

        st.markdown("### 2. PIB per cápita vs felicidad (por región)")
        plot_gdp_vs_happiness(df_filtered)

        st.markdown("### 3. Top países más felices")
        top_n = st.slider("Número de países a mostrar", 5, 20, 10)
        plot_top_countries(df_filtered, top_n=top_n)

        st.markdown("### 4. Felicidad por región")
        plot_happiness_by_region(df_filtered)

        st.markdown("### 5. Comparación de modelos (R²)")
        plot_models_comparison(results)

    # ---- Tab 3: Predicciones ----
    with tab3:
        st.subheader("Predicción del índice de felicidad")

        if modo_prediccion == "Seleccionar país":
            paises_opcion = df_filtered["Country"].values
            pais = st.selectbox("Selecciona un país", paises_opcion)
            fila = df[df["Country"] == pais].iloc[0]

            input_values = np.array([[
                fila["GDP_per_capita"],
                fila["Social_support"],
                fila["Healthy_life_expectancy"],
                fila["Freedom"],
                fila["Generosity"],
                fila["Corruption"]
            ]])

            scaler = StandardScaler()
            scaler.fit(X)  # entrenamos sobre todos los datos para escalar
            input_scaled = scaler.transform(input_values)
            pred = current_model.predict(input_scaled)[0]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("País", pais)
            with col2:
                st.metric("Región", fila["Region"])
            with col3:
                st.metric("Felicidad real", f"{fila['HappinessScore']:.3f}")
            with col4:
                st.metric("Predicción modelo", f"{pred:.3f}")

            error = float(pred - fila["HappinessScore"])
            st.write(f"Error (predicción - real): {error:.3f}")

        else:
            st.markdown("Ajusta los valores de las variables para obtener una predicción:")

            col1, col2 = st.columns(2)

            with col1:
                gdp = st.slider(
                    "PIB per cápita",
                    float(df["GDP_per_capita"].min()),
                    float(df["GDP_per_capita"].max()),
                    float(df["GDP_per_capita"].mean())
                )
                social = st.slider(
                    "Apoyo social",
                    float(df["Social_support"].min()),
                    float(df["Social_support"].max()),
                    float(df["Social_support"].mean())
                )
                health = st.slider(
                    "Esperanza de vida saludable",
                    float(df["Healthy_life_expectancy"].min()),
                    float(df["Healthy_life_expectancy"].max()),
                    float(df["Healthy_life_expectancy"].mean())
                )

            with col2:
                freedom = st.slider(
                    "Libertad para tomar decisiones",
                    float(df["Freedom"].min()),
                    float(df["Freedom"].max()),
                    float(df["Freedom"].mean())
                )
                generosity = st.slider(
                    "Generosidad",
                    float(df["Generosity"].min()),
                    float(df["Generosity"].max()),
                    float(df["Generosity"].mean())
                )
                corruption = st.slider(
                    "Percepción de corrupción",
                    float(df["Corruption"].min()),
                    float(df["Corruption"].max()),
                    float(df["Corruption"].mean())
                )

            input_values = np.array([[gdp, social, health, freedom, generosity, corruption]])

            scaler = StandardScaler()
            scaler.fit(X)
            input_scaled = scaler.transform(input_values)
            pred = current_model.predict(input_scaled)[0]

            st.markdown("### Resultado de la predicción")
            st.metric("Índice de felicidad estimado", f"{pred:.3f}")

            st.info(
                "Este valor es una estimación basada en los datos de 2019 "
                "y en las relaciones aprendidas por el modelo."
            )


if __name__ == "__main__":
    main()
