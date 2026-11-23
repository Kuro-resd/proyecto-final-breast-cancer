================================================================================

PROYECTO FINAL DE CIENCIA DE DATOS

Clasificacion Supervisada para el Diagnostico de Cancer de Mama

(Breast Cancer Wisconsin - Diagnostic)

================================================================================



Autor:       Raul Emiliano Sotelo Delgado

Carrera:     Ingenieria en Computacion

Institucion: Universidad Autonoma del Estado de Mexico, Campus Texcoco

Correo:      rsotelod001@alumno.uaemex.mx

Instructor:  M. en C. Benito Samuel Lopez Razo



--------------------------------------------------------------------------------

1. DESCRIPCION DEL PROYECTO

--------------------------------------------------------------------------------

Este repositorio contiene el desarrollo completo del proyecto final de Ciencia

de Datos sobre el conjunto Breast Cancer Wisconsin (Diagnostic) del repositorio

UCI (569 muestras y 30 variables numericas continuas).



El objetivo es clasificar tumores mamarios en Benignos (0) y Malignos (1)

comparando cuatro modelos supervisados:

  - Regresion Logistica (LogisticRegression): modelo de referencia (linea base).

  - Clasificador Bayesiano (GaussianNB): variante de Naive Bayes adecuada para

    variables continuas, evaluando el efecto de la multicolinealidad sobre su

    supuesto de independencia condicional.

  - Maquina de Soporte Vectorial (SVC): se evaluaron configuraciones de SVM con

    kernels lineal y RBF (ajustando C y gamma), seleccionando la configuracion

    final (RBF) mediante busqueda en malla y validacion cruzada.

  - Red Neuronal Artificial (MLPClassifier): perceptron multicapa con

    arquitecturas compactas, funciones de activacion relu y tanh, regularizacion

    L2 (alpha) y detencion temprana (early_stopping=True) para evitar sobreajuste.



--------------------------------------------------------------------------------

2. ESTRUCTURA DEL REPOSITORIO

--------------------------------------------------------------------------------

Proyecto Final/

  data/

    breast+cancer+wisconsin+diagnostic/   -> Dataset original (wdbc.data y wdbc.names)

    procesados/                           -> Conjuntos divididos y escalados (script 02)

      x_train_scaled.csv

      x_val_scaled.csv

      x_test_scaled.csv

      y_train.csv

      y_val.csv

      y_test.csv

  presentacion/                           -> Material de presentacion de resultados

  reporte/

    articulo_ieee.tex                     -> Codigo fuente LaTeX (plantilla IEEE)

    referencias.bib                       -> Referencias bibliograficas en formato IEEE

    articulo_ieee.pdf                     -> Reporte final compilado (6 paginas)

  resultados/

    figuras/                              -> 6 graficas en 300 DPI para el articulo

      01_distribucion_clases.png

      02_distribucion_variables.png

      03_correlacion_mean.png

      04_matrices_confusion.png

      05_curvas_roc.png

      06_curvas_precision_recall.png

    tabla_comparativa_resultados.csv      -> Metricas finales en CV (5-Fold) y prueba

  src/

    01_sondeo_errores.py                  -> Auditoria de nulos, ceros, escalas y correlaciones

    02_preparacion_y_eda.py               -> Limpieza, 3 graficas EDA, division 70/15/15 y escalado

    03_modelos_y_evaluacion.py            -> Pipelines, GridSearchCV (5-Fold), metricas y errores

  .gitignore

  README.txt (o README.md)

  requirements.txt



--------------------------------------------------------------------------------

3. REQUISITOS E INSTALACION

--------------------------------------------------------------------------------

Se desarrollo y probo con Python 3.12.



Paso 1. Abrir una terminal en la carpeta raiz del proyecto y crear un entorno

virtual:



    python -m venv .venv



Paso 2. Activar el entorno virtual:



  En Windows (PowerShell):

    .\.venv\Scripts\Activate.ps1



  En Linux / macOS:

    source .venv/bin/activate



Paso 3. Instalar las dependencias exactas del proyecto:



    pip install -r requirements.txt



--------------------------------------------------------------------------------

4. INSTRUCCIONES DE EJECUCION

--------------------------------------------------------------------------------

Con el entorno virtual (.venv) activo y ubicado en la carpeta raiz del proyecto,

ejecute los tres scripts en el siguiente orden:



    python src/01_sondeo_errores.py

    python src/02_preparacion_y_eda.py

    python src/03_modelos_y_evaluacion.py



Resumen de cada script:

  - 01_sondeo_errores.py: verifica nulos, duplicados, longitudes de ID, ceros

    exactos en concavidad, diferencias de escala, multicolinealidad (r > 0.90)

    y casos frontera.

  - 02_preparacion_y_eda.py: limpia y codifica la variable objetivo (M=1, B=0),

    genera las 3 graficas de EDA en resultados/figuras/ y exporta las

    particiones estratificadas y escaladas (70/15/15) en data/procesados/.

  - 03_modelos_y_evaluacion.py: reserva el 20% (n = 114) para prueba final y

    ejecuta GridSearchCV con validacion cruzada estratificada de 5 folds sobre

    el 80% restante (n = 455), encapsulando StandardScaler dentro de un Pipeline

    para evitar fuga de datos. Genera tabla_comparativa_resultados.csv, las

    graficas de evaluacion (confusion, ROC y Precision-Recall) e imprime el

    analisis de errores.



--------------------------------------------------------------------------------

5. REPRODUCIBILIDAD

--------------------------------------------------------------------------------

Todos los scripts utilizan rutas relativas y establecen la semilla aleatoria

RANDOM_STATE = 42 para favorecer la reproducibilidad de la division de datos,

la validacion cruzada y el entrenamiento de los modelos.



Las dependencias directas del proyecto se especifican con sus versiones exactas

en requirements.txt. Para obtener resultados consistentes, se recomienda

utilizar las versiones de Python y de las bibliotecas indicadas en la

documentacion del proyecto.



Los resultados pueden presentar pequenas variaciones entre distintos entornos

de ejecucion debido a diferencias en las versiones de las bibliotecas y en las

plataformas de computo.