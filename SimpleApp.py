import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Page configuration
st.set_page_config(
  page_title='Iris prediccion',
  layout='wide',
  initial_sidebar_state='expanded'
)

# Title of the app
st.title('Prediccion clase de orquidea')

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Exploratory Data Analysis (EDA)
st.header("Análisis Exploratorio de Datos")

# Summary statistics
st.write("**Resumen estadístico:**")
st.dataframe(df.describe())

# Feature distribution visualizations
st.write("**Distribución de las características:**")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df['sepal length (cm)'], ax=axes[0, 0])
sns.histplot(df['sepal width (cm)'], ax=axes[0, 1])
sns.histplot(df['petal length (cm)'], ax=axes[1, 0])
sns.histplot(df['petal width (cm)'], ax=axes[1, 1])
st.pyplot(fig)


# User input sliders
# Sepal Length
sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    min_value=df['sepal length (cm)'].min(),
    max_value=df['sepal length (cm)'].max(),
    value=df['sepal length (cm)'].mean()
)

# Sepal Width
sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    min_value=df['sepal width (cm)'].min(),
    max_value=df['sepal width (cm)'].max(),
    value=df['sepal width (cm)'].mean()
)

# Petal Length
petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    min_value=df['petal length (cm)'].min(),
    max_value=df['petal length (cm)'].max(),
    value=df['petal length (cm)'].mean()
)

# Petal Width
petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    min_value=df['petal width (cm)'].min(),
    max_value=df['petal width (cm)'].max(),
    value=df['petal width (cm)'].mean()
)

# Data preparation and model training
df['target'] = iris.target

groupby_species_mean = df.groupby('target').mean()
print(groupby_species_mean)

X = df.drop(['target'], axis='columns')
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=40,max_depth=4)
model.fit(X_train, y_train)

# Prediction
new_data = np.array([sepal_length, sepal_width, petal_length, petal_width])
predicted_label = model.predict(new_data.reshape(1, -1))[0]

st.write("**Datos seleccionados:**")
selected_data = pd.DataFrame({
    'Característica': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
    'Valor': [sepal_length, sepal_width, petal_length, petal_width]
})
st.table(selected_data)

st.write(f"La especie de iris predicha es: {iris.target_names[predicted_label]}")