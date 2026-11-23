import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Codigo de Raúl Emiliano Sotelo Delgado
# Parte 2: Limpieza, Analisis Exploratorio (EDA), division y escalado de datos
# Este script toma todo lo que detectamos en 01_sondeo_errores.py y realiza la preparacion
# completa usando los procedimientos que practicamos en "Clase 3-4 EDA/Untitled.ipynb"
# y "Tratamiento de complejidad de datos/04_PreparacionDatos.ipynb".

# Fijamos una semilla aleatoria para favorecer la reproducibilidad de las divisiones
# y mantener resultados consistentes al ejecutarlo en otro entorno
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Rutas relativas para mantener ordenado el proyecto
ruta_datos = os.path.join("data", "breast+cancer+wisconsin+diagnostic", "wdbc.data")
ruta_figuras = os.path.join("resultados", "figuras")
ruta_procesados = os.path.join("data", "procesados")

os.makedirs(ruta_figuras, exist_ok=True)
os.makedirs(ruta_procesados, exist_ok=True)

# 1. CARGA Y LIMPIEZA DE DATOS

caracteristicas_base = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
]
columnas = ["id", "diagnosis"] + (
    [f"mean_{c}" for c in caracteristicas_base] +
    [f"se_{c}" for c in caracteristicas_base] +
    [f"worst_{c}" for c in caracteristicas_base]
)

df = pd.read_csv(ruta_datos, header=None, names=columnas)

# Eliminamos la columna 'id' usando drop() como lo vimos en 04_PreparacionDatos.ipynb
# porque los numeros de expediente no sirven para predecir el diagnostico
df = df.drop(columns=["id"])

# Codificamos la variable objetivo de texto (M y B) a numeros:
# Maligno (M) = 1 porque es la clase positiva que nos interesa detectar clinicamente
# Benigno (B) = 0
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Separamos variables predictoras (X) y variable objetivo (y)
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]


# 2. GENERACION DE LAS 3 GRAFICAS DEL EDA (GUARDADAS EN resultados/figuras/)

# Grafica 1: Proporcion de clases para revisar si hay desbalance
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="Set2")
plt.xticks([0, 1], ["Benigno (0)", "Maligno (1)"])
plt.title("Distribucion de clases en Breast Cancer Wisconsin")
plt.xlabel("Diagnostico")
plt.ylabel("Cantidad de pacientes")
plt.tight_layout()
plt.savefig(os.path.join(ruta_figuras, "01_distribucion_clases.png"), dpi=300)
plt.close()

# Grafica 2: Distribucion (curvas de densidad / histogramas suavizados) de 4 variables clave
# Esto nos sirve para ver donde se traslapan los tumores benignos grandes con malignos pequenos
variables_clave = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.ravel(), variables_clave):
    sns.kdeplot(data=df, x=col, hue="diagnosis", fill=True, ax=ax, palette="Set1")
    ax.set_title(f"Distribucion de {col} por clase")
    ax.legend(title="Diagnostico", labels=["Maligno (1)", "Benigno (0)"])
plt.tight_layout()
plt.savefig(os.path.join(ruta_figuras, "02_distribucion_variables.png"), dpi=300)
plt.close()

# Grafica 3: Mapa de calor de correlaciones de las 10 variables 'mean'
# Con esto comprobamos visualmente la multicolinealidad entre radio, perimetro y area
plt.figure(figsize=(10, 8))
corr_mean = X.iloc[:, :10].corr()
sns.heatmap(corr_mean, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Mapa de calor de correlaciones (Variables mean)")
plt.tight_layout()
plt.savefig(os.path.join(ruta_figuras, "03_correlacion_mean.png"), dpi=300)
plt.close()

print("Las 3 graficas del EDA se guardaron correctamente en:", ruta_figuras)


# 3. DIVISION EN ENTRENAMIENTO (70%), VALIDACION (15%) Y PRUEBA (15%)

# Usamos stratify=y para que en los 3 conjuntos se respete la misma proporcion de 62.7% benignos y 37.3% malignos
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

print("\nResumen de la division de datos:")
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Validacion:    {X_val.shape[0]} muestras")
print(f"Prueba:        {X_test.shape[0]} muestras")


# 4. ESCALADO SIN FUGA DE DATOS (DATA LEAKAGE)

# Aqui es donde evitamos un error muy comun: NUNCA se escala toda la tabla antes de dividir.
# Ajustamos (fit_transform) StandardScaler SOLO con X_train y aplicamos transform a val y test
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

# Guardamos los archivos procesados igual que en la carpeta "Tratamiento de complejidad de datos" de clase
X_train_scaled.to_csv(os.path.join(ruta_procesados, "x_train_scaled.csv"), index=True)
X_val_scaled.to_csv(os.path.join(ruta_procesados, "x_val_scaled.csv"), index=True)
X_test_scaled.to_csv(os.path.join(ruta_procesados, "x_test_scaled.csv"), index=True)
y_train.to_csv(os.path.join(ruta_procesados, "y_train.csv"), index=True)
y_val.to_csv(os.path.join(ruta_procesados, "y_val.csv"), index=True)
y_test.to_csv(os.path.join(ruta_procesados, "y_test.csv"), index=True)

print("Conjuntos divididos y escalados guardados en:", ruta_procesados)