import os
import pandas as pd
import numpy as np

# Codigo de Raúl Emiliano Sotelo Delgado
# Parte 1: Sondeo y deteccion de errores en wdbc.data
# Aunque en el archivo wdbc.names los autores dicen que la base de datos esta limpia
# y no tiene valores faltantes, decidi no confiarme y hacer este script independiente
# para revisar la base de datos por mi cuenta, basandome en las revisiones que vimos
# en clase en "Clase 3-4 EDA/Untitled.ipynb" y "Tratamiento de complejidad de datos/04_PreparacionDatos.ipynb".

# Ruta relativa para que el codigo corra en cualquier computadora sin importar en que disco este
ruta_datos = os.path.join("data", "breast+cancer+wisconsin+diagnostic", "wdbc.data")

# Primer problema que encontre: el archivo wdbc.data empieza directo con los datos del primer paciente
# Si usara pd.read_csv(ruta_datos) normal, python se comeria la primera fila como encabezados.
# Por eso use header=None y arme la lista de nombres guiandome con la seccion 7 de wdbc.names
caracteristicas_base = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
]

# Cada una de las 10 caracteristicas se repite 3 veces: promedio (mean), error estandar (se) y peor valor (worst)
columnas_variables = (
    [f"mean_{c}" for c in caracteristicas_base] +
    [f"se_{c}" for c in caracteristicas_base] +
    [f"worst_{c}" for c in caracteristicas_base]
)
columnas = ["id", "diagnosis"] + columnas_variables

df = pd.read_csv(ruta_datos, header=None, names=columnas)


print("1. REVISION GENERAL DE LA BASE DE DATOS")

print(f"Total de filas (pacientes): {df.shape[0]}")
print(f"Total de columnas: {df.shape[1]}")
print(f"Valores faltantes (nulos) totales: {df.isnull().sum().sum()}")
print(f"Filas duplicadas exactas: {df.duplicated().sum()}")

# Revision de la columna ID: aqui podemos ver que tienen longitudes diferentes (de 4 a 9 digitos)
# y no sirven para predecir si un tumor es maligno o benigno, por eso en el siguiente script la vamos a borrar
longitudes_id = df["id"].astype(str).str.len()
print(f"\nLongitud minima de ID: {longitudes_id.min()} digitos | Maxima: {longitudes_id.max()} digitos")


print("2. REVISION DE CLASES (M = Maligno, B = Benigno)")

print(df["diagnosis"].value_counts())
print("\nEn porcentajes:")
print((df["diagnosis"].value_counts(normalize=True) * 100).round(2))

print("3. REVISION DE CEROS EXACTOS EN CONCAVIDAD Y PUNTOS CONCAVOS")

# Al revisar el archivo note varios ceros en estas columnas. Aqui cuento cuantos hay y de que clase son
# para saber si son errores o si pertenecen a tumores benignos con bordes lisos
cols_concavidad = ["mean_concavity", "mean_concave_points", "worst_concavity", "worst_concave_points"]
for col in cols_concavidad:
    ceros = df[df[col] == 0]
    conteo_clases = ceros["diagnosis"].value_counts().to_dict()
    print(f"Columna {col}: {len(ceros)} ceros exactos,  Distribucion por clase: {conteo_clases}")

print("4. REVISION DE ESCALAS (MINIMOS Y MAXIMOS)")

# Aqui comprobamos la diferencia extrema de escalas entre variables.
# Esto justifica por que es obligatorio estandarizar antes de usar SVM y la Red Neuronal
solo_numericas = df.drop(columns=["id", "diagnosis"])
min_global = solo_numericas.min().min()
max_global = solo_numericas.max().max()
print(f"Valor minimo en variables numericas: {min_global}")
print(f"Valor maximo en variables numericas: {max_global}")
print("\nEjemplos de diferencia de escala:")
for col in ["se_smoothness", "mean_fractal_dimension", "mean_area", "worst_area"]:
    print(f"  {col:25s} -> min: {df[col].min():.6f} | max: {df[col].max():.4f}")

print("5. DETECCION DE MULTICOLINEALIDAD (CORRELACIONES > 0.90)")

# Si dos variables tienen correlacion mayor a 0.90 practicamente miden lo mismo.
# Esto afecta directo al supuesto de independencia de Naive Bayes
matriz_corr = solo_numericas.corr().abs()
pares_altos = []
for i in range(len(matriz_corr.columns)):
    for j in range(i + 1, len(matriz_corr.columns)):
        valor = matriz_corr.iloc[i, j]
        if valor > 0.90:
            pares_altos.append((matriz_corr.columns[i], matriz_corr.columns[j], round(valor, 4)))

print(f"Se encontraron {len(pares_altos)} pares de variables con correlacion mayor a 0.90.")
print("Primeros 10 pares mas correlacionados:")
for v1, v2, val in sorted(pares_altos, key=lambda x: x[2], reverse=True)[:10]:
    print(f"  {v1} <-> {v2}: {val}")


print("6. CASOS ATIPICOS Y CASOS FRONTERA")

# Casos donde el valor promedio y el peor valor son exactamente iguales
iguales = df[df["mean_radius"] == df["worst_radius"]]
print(f"Pacientes donde mean_radius es igual a worst_radius: {len(iguales)}")
print(iguales[["id", "diagnosis", "mean_radius", "worst_radius", "mean_area", "worst_area"]].to_string(index=False))

# Casos frontera: tumores malignos pequenos vs benignos grandes
malignos_chicos = df[(df["diagnosis"] == "M") & (df["mean_radius"] < 12.0)]
benignos_grandes = df[(df["diagnosis"] == "B") & (df["mean_radius"] > 16.0)]
print(f"\nTumores Malignos con radio pequeno (< 12): {len(malignos_chicos)}")
print(f"Tumores Benignos con radio grande (> 16):  {len(benignos_grandes)}")
print("\nSondeo terminado. Con esta informacion ya podemos limpiar y preparar los datos correctamente.")