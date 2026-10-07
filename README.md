# World Happiness 2019 - Análisis y Predicción con Machine Learning

Proyecto de análisis de datos y machine learning basado en el dataset **World Happiness Report 2019**.

El objetivo del proyecto es analizar qué factores socioeconómicos tienen mayor relación con el índice de felicidad de los países y construir modelos capaces de predecir dicho índice.

## Objetivo

Predecir el índice de felicidad de un país utilizando variables como:

- PIB per cápita
- Apoyo social
- Esperanza de vida saludable
- Libertad para tomar decisiones
- Generosidad
- Percepción de corrupción

También se busca identificar cuáles de estas variables tienen mayor influencia en el nivel de felicidad.

## Dataset

Se utiliza el archivo:

`2019.csv`

correspondiente al World Happiness Report 2019.

El dataset contiene información de aproximadamente 156 países y variables relacionadas con bienestar, economía y calidad de vida.

## Variables utilizadas

Las variables predictoras utilizadas en los modelos son:

- GDP_per_capita
- Social_support
- Healthy_life_expectancy
- Freedom
- Generosity
- Corruption

La variable objetivo es:

- HappinessScore

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Análisis exploratorio de datos

Durante el análisis se generaron diferentes visualizaciones para comprender mejor el comportamiento del dataset.

Entre ellas:

- Matriz de correlación
- PIB per cápita vs índice de felicidad
- Distribución del índice de felicidad
- Top 10 países más felices
- Distribución de felicidad por región
- Comparación entre modelos

El análisis muestra que variables como el PIB per cápita, el apoyo social y la esperanza de vida saludable presentan una relación importante con el índice de felicidad.

## Modelos utilizados

Se entrenaron y compararon tres modelos de regresión:

### Regresión Lineal

Se utilizó como modelo base para observar la relación lineal entre las variables socioeconómicas y el índice de felicidad.

### Random Forest Regressor

Modelo basado en múltiples árboles de decisión, capaz de capturar relaciones no lineales entre las variables.

### Gradient Boosting Regressor

Modelo que construye árboles de manera secuencial para corregir progresivamente los errores de predicción.

## Evaluación

Los modelos se evaluaron utilizando las siguientes métricas:

- R²
- RMSE
- MAE

Los tres modelos lograron explicar una parte importante de la variación del índice de felicidad.

Random Forest y Gradient Boosting presentaron un desempeño competitivo frente a la regresión lineal.

## Dashboard interactivo

Además del análisis en Python, se desarrolló un dashboard utilizando **Streamlit**.

El dashboard permite:

- Seleccionar entre distintos modelos de regresión
- Ajustar hiperparámetros
- Filtrar países por región
- Visualizar métricas de desempeño
- Explorar el dataset
- Comparar modelos
- Seleccionar un país y obtener una predicción
- Introducir valores personalizados para estimar el índice de felicidad

Para ejecutar el dashboard:

```bash
streamlit run app.py
