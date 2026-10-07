# world_happiness_analysis.py
# Análisis del World Happiness Report 2019
# Incluye macro-regiones y las usa en el EDA.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------
# 1. Carga de datos y creación de regiones
# -------------------------------------------------------------------

def _assign_region(country: str) -> str:
    """
    Asigna una macro-región a un país.
    Si no se encuentra el país, regresa 'Other'.
    Puedes agregar o mover países según necesites.
    """
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

    # Si no lo encontramos, lo mandamos a Other
    return "Other"


def add_region_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si el dataset ya tiene columna 'Region', la respeta.
    Si no, crea una nueva columna 'Region' a partir del país.
    """
    if "Region" in df.columns:
        df["Region"] = df["Region"].fillna("Other")
        return df

    if "Country" not in df.columns:
        raise ValueError("No se encuentra la columna 'Country' para generar 'Region'.")

    df["Region"] = df["Country"].apply(_assign_region)
    return df


def load_data(path: str = "2019.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Renombrar columnas a nombres más manejables
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

    # Agregar región (o respetar la que ya venga)
    df = add_region_column(df)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Manejo básico de nulos
    if df.isnull().sum().sum() > 0:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype != "object":
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
    return df


# -------------------------------------------------------------------
# 2. Preparación de features (modelos solo con variables numéricas)
# -------------------------------------------------------------------

def prepare_features(df: pd.DataFrame):
    # Variables numéricas (sin incluir Region en el modelo, si quieres
    # usarla podrías hacer dummies aquí)
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

    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, features


# -------------------------------------------------------------------
# 3. EDA y visualización (aquí sí usamos Region)
# -------------------------------------------------------------------

def plot_correlation(df: pd.DataFrame):
    plt.figure(figsize=(9, 7))
    cols = [
        "HappinessScore",
        "GDP_per_capita",
        "Social_support",
        "Healthy_life_expectancy",
        "Freedom",
        "Generosity",
        "Corruption"
    ]
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.show()


def plot_gdp_vs_happiness(df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="GDP_per_capita",
        y="HappinessScore",
        hue="Region"
    )
    plt.xlabel("PIB per cápita")
    plt.ylabel("Índice de felicidad")
    plt.title("PIB per cápita vs Felicidad (coloreado por región)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_happiness_hist(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["HappinessScore"], kde=True)
    plt.xlabel("Índice de felicidad")
    plt.title("Distribución del índice de felicidad")
    plt.tight_layout()
    plt.show()


def plot_top10_countries(df: pd.DataFrame):
    top10 = df.sort_values("HappinessScore", ascending=False).head(10)
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=top10,
        x="HappinessScore",
        y="Country",
        hue="Region",
        dodge=False
    )
    plt.xlabel("Índice de felicidad")
    plt.ylabel("País")
    plt.title("Top 10 países más felices")
    plt.legend(title="Región", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_happiness_by_region(df: pd.DataFrame):
    """
    Distribución del índice de felicidad por región.
    """
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
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 4. Modelado y evaluación
# -------------------------------------------------------------------

def train_models(X_train_scaled, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        models[name] = model

    return models


def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return r2, rmse, mae, y_pred


def print_model_results(models, X_test_scaled, y_test):
    results = {}
    print("\n=== RESULTADOS DE LOS MODELOS ===")
    for name, model in models.items():
        r2, rmse, mae, _ = evaluate_model(model, X_test_scaled, y_test)
        results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        print(f"\nModelo: {name}")
        print(f"  R²   : {r2:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  MAE  : {mae:.4f}")
    return results


def plot_feature_importances(model, features, title):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        plt.figure(figsize=(8, 5))
        sns.barplot(
            x=importances[sorted_idx],
            y=np.array(features)[sorted_idx]
        )
        plt.title(f"Importancia de variables - {title}")
        plt.xlabel("Importancia")
        plt.tight_layout()
        plt.show()
    else:
        print(f"El modelo {title} no tiene atributo 'feature_importances_'.")


# -------------------------------------------------------------------
# 5. Main
# -------------------------------------------------------------------

def main():
    print("Cargando datos...")
    df = load_data()
    df = clean_data(df)

    print("\nPrimeras filas del dataset:")
    print(df.head())

    print("\nDescripción estadística:")
    print(df.describe())

    print("\nDistribución de países por región:")
    print(df["Region"].value_counts())

    # EDA visual (descomenta para ver las gráficas)
    # plot_correlation(df)
    # plot_gdp_vs_happiness(df)
    # plot_happiness_hist(df)
    # plot_top10_countries(df)
    # plot_happiness_by_region(df)

    # Preparar features para los modelos (solo numéricas)
    (
        X, y,
        X_train, X_test,
        y_train, y_test,
        X_train_scaled, X_test_scaled,
        scaler, features
    ) = prepare_features(df)

    # Entrenar modelos
    models = train_models(X_train_scaled, y_train)

    # Evaluar y mostrar resultados
    results = print_model_results(models, X_test_scaled, y_test)

    # Importancia de variables para modelos de árbol
    print("\nGráfica de importancia de variables (Random Forest):")
    plot_feature_importances(models["Random Forest"], features, "Random Forest")

    print("\nGráfica de importancia de variables (Gradient Boosting):")
    plot_feature_importances(models["Gradient Boosting"], features, "Gradient Boosting")

    # Estadísticas finales
    print("\n=== INFORMACIÓN DEL PROYECTO ===")
    print(f"Dataset original: {df.shape}")
    print(f"Variables predictoras numéricas: {len(features)}")
    print(f"Observaciones de entrenamiento: {X_train.shape[0]}")
    print(f"Observaciones de prueba: {X_test.shape[0]}")
    print("Variable objetivo: HappinessScore")
    print(f"Países analizados: {len(df)}")

    print("\n=== ESTADÍSTICAS CLAVE ===")
    print(f"Score de felicidad promedio: {df['HappinessScore'].mean():.2f}")
    print(f"Score de felicidad máximo: {df['HappinessScore'].max():.2f}")
    print(f"Score de felicidad mínimo: {df['HappinessScore'].min():.2f}")
    print(f"Rango de scores: {df['HappinessScore'].max() - df['HappinessScore'].min():.2f}")


if __name__ == "__main__":
    main()
