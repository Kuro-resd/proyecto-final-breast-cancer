import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)

warnings.filterwarnings("ignore")

# Codigo de Raúl Emiliano Sotelo Delgado
# Parte 3: Entrenamiento de modelos, ajuste con validacion cruzada y evaluacion final
# En este script junte los 4 modelos en un mismo flujo usando Pipeline de scikit-learn.
# Seguramente se preguntara por que uso Pipeline en lugar de cargar los CSV ya escalados:
# Porque al hacer Validacion Cruzada (5-Fold), si usaramos datos ya escalados antes de los folds
# estariamos filtrando informacion de validacion hacia entrenamiento (data leakage).
# Al meter StandardScaler dentro del Pipeline, el escalado se recalcula limpio en cada fold.

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ruta_datos = os.path.join("data", "breast+cancer+wisconsin+diagnostic", "wdbc.data")
ruta_figuras = os.path.join("resultados", "figuras")
os.makedirs(ruta_figuras, exist_ok=True)

# 1. CARGA LIMPIA Y DIVISION 80% ENTRENAMIENTO PARA CV / 20% PRUEBA FINAL

caracteristicas_base = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
]
columnas = ["id", "diagnosis"] + (
    [f"mean_{c}" for c in caracteristicas_base] +
    [f"se_{c}" for c in caracteristicas_base] +
    [f"worst_{c}" for c in caracteristicas_base]
)

df = pd.read_csv(ruta_datos, header=None, names=columnas).drop(columns=["id"])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Reservamos el 20% (114 pacientes) completamente intacto para la prueba final
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# Estrategia de validacion cruzada estratificada de 5 folds
# Es indispensable aqui porque 569 muestras es un conjunto pequeno
cv_estrategia = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# 2. DEFINICION DE LOS 4 MODELOS Y SUS ESPACIOS DE BUSQUEDA

# En clase vimos Naive Bayes (NAIVELLES.ipynb) y Redes Neuronales (07_RedesNeuronales.ipynb).
# Ademas agregue Regresion Logistica como modelo base para saber si un modelo lineal simple
# basta o si realmente necesitamos algo mas complejo y SVM pedido en el proyecto y muy bueno
# en conjuntos pequenos buscando el hiperplano de margen maximo.

modelos_a_probar = {
    "Regresion Logistica (Ref)": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        {
            "clf__C": [0.01, 0.1, 1, 10, 100]
        }
    ),
    "Naive Bayes (GaussianNB)": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ]),
        {
            "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
        }
    ),
    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE))
        ]),
        [
            {"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10, 100]},
            {"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]}
        ]
    ),
    "Red Neuronal (MLP)": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                max_iter=1000,
                early_stopping=True,      # Detencion temprana para evitar sobreajuste
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=RANDOM_STATE
            ))
        ]),
        {
            # Arquitecturas compactas porque con 455 muestras de entrenamiento una red gigante memoriza
            "clf__hidden_layer_sizes": [(32,), (64, 32), (32, 16)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": [0.0001, 0.001, 0.01]  # Regularizacion L2
        }
    )
}


# 3. ENTRENAMIENTO, BUSQUEDA EN MALLA GRIDSEARCH Y EVALUACION EN PRUEBA

lista_resultados = []
modelos_ganadores = {}

print("Iniciando busqueda de hiperparametros con Validacion Cruzada (5-Fold)...")
for nombre, (pipeline, grid_params) in modelos_a_probar.items():
    # Optimizamos F1 porque en cancer nos importa equilibrar Precision y Sensibilidad (Recall)
    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params,
        cv=cv_estrategia,
        scoring="f1",
        n_jobs=-1
    )
    busqueda.fit(X_train, y_train)
    mejor_modelo = busqueda.best_estimator_
    modelos_ganadores[nombre] = mejor_modelo

    # Medimos la variabilidad entre folds (media y desviacion estandar) para saber si las diferencias son reales
    scores_cv = cross_validate(
        mejor_modelo, X_train, y_train, cv=cv_estrategia,
        scoring=["accuracy", "f1", "roc_auc"]
    )

    # Evaluacion final sobre el 20% de prueba que el modelo nunca vio
    y_pred = mejor_modelo.predict(X_test)
    y_prob = mejor_modelo.predict_proba(X_test)[:, 1]

    lista_resultados.append({
        "Modelo": nombre,
        "Mejores Hiperparametros": str(busqueda.best_params_),
        "CV F1 (Media +- Std)": f"{scores_cv['test_f1'].mean():.4f} +- {scores_cv['test_f1'].std():.4f}",
        "Test Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Test Precision": round(precision_score(y_test, y_pred), 4),
        "Test Recall": round(recall_score(y_test, y_pred), 4),
        "Test F1": round(f1_score(y_test, y_pred), 4),
        "Test ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
    })

tabla_resultados = pd.DataFrame(lista_resultados)
print("\n=== TABLA COMPARATIVA FINAL (CONJUNTO DE PRUEBA) ===")
print(tabla_resultados.to_string(index=False))

ruta_csv = os.path.join("resultados", "tabla_comparativa_resultados.csv")
tabla_resultados.to_csv(ruta_csv, index=False)
print("\nTabla guardada en:", ruta_csv)


# 4. GRAFICAS DE EVALUACION (MATRICES DE CONFUSION, ROC Y PRECISION-RECALL)

# Grafica 4: Matrices de confusion
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (nombre, modelo) in zip(axes.ravel(), modelos_ganadores.items()):
    cm = confusion_matrix(y_test, modelo.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["Benigno", "Maligno"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(nombre)
plt.tight_layout()
plt.savefig(os.path.join(ruta_figuras, "04_matrices_confusion.png"), dpi=300)
plt.close()

# Grafica 5: Curvas ROC comparativas
plt.figure(figsize=(8, 6))
ax = plt.gca()
for nombre, modelo in modelos_ganadores.items():
    RocCurveDisplay.from_estimator(modelo, X_test, y_test, ax=ax, name=nombre)
plt.plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.50)")
plt.title("Curvas ROC Comparativas en Conjunto de Prueba")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(ruta_figuras, "05_curvas_roc.png"), dpi=300)
plt.close()

# Grafica 6: Curvas Precision-Recall recomendada porque hay desbalance moderado 63/37
plt.figure(figsize=(8, 6))
ax = plt.gca()
for nombre, modelo in modelos_ganadores.items():
    PrecisionRecallDisplay.from_estimator(modelo, X_test, y_test, ax=ax, name=nombre)
plt.title("Curvas Precision-Recall en Conjunto de Prueba")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(ruta_figuras, "06_curvas_precision_recall.png"), dpi=300)
plt.close()


# 5. ANALISIS DE ERRORES (QUE CASOS FALLARON Y POR QUE)

print("\n=== ANALISIS DE ERRORES EN CONJUNTO DE PRUEBA ===")
todos_los_errores = set()
for nombre, modelo in modelos_ganadores.items():
    y_pred = modelo.predict(X_test)
    indices_error = list(X_test.index[y_pred != y_test])
    todos_los_errores.update(indices_error)

    # En medicina un Falso Negativo (era Maligno=1 y predijo Benigno=0) es el error mas grave
    fn = int(sum((y_test == 1) & (y_pred == 0)))
    fp = int(sum((y_test == 0) & (y_pred == 1)))
    print(f"{nombre}: {len(indices_error)} errores totales (Falsos Negativos: {fn}, Falsos Positivos: {fp})")
    if indices_error:
        print(f"  Indices mal clasificados: {indices_error}")

# Imprimimos las caracteristicas clave de los pacientes que fallaron para explicar el por que en el reporte
if todos_los_errores:
    print("\nDetalle de los pacientes donde fallo al menos un modelo:")
    resumen_errores = df.loc[sorted(todos_los_errores), ["diagnosis", "mean_radius", "mean_texture", "mean_area", "worst_radius", "worst_area"]]
    print(resumen_errores.to_string())

# Notas para el usuario:
# La tabla comparativa final se guarda en resultados/tabla_comparativa_resultados.csv
# y las graficas de evaluacion en resultados/figuras/
# Todo el codigo esta probado en Python 3.12 dentro de .venv con las versiones exactas de requirements.txt
# Para reproducir el experimento, ejecute en orden desde la raiz del proyecto:
# 1) python src\01_sondeo_errores.py
# 2) python src\02_preparacion_y_eda.py
# 3) python src\03_modelos_y_evaluacion.py