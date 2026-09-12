# Algoritmos Genéticos aplicados a Machine Learning


**Objetivo del trabajo:** comprender y demostrar, con ejemplos ejecutables, la aplicación de Algoritmos Genéticos (AG) en tres etapas distintas del ciclo de vida de un modelo de Machine Learning:

1. **Feature Selection** — selección de las mejores características.
2. **Hyperparameter Optimization** — búsqueda de los mejores hiperparámetros.
3. **Neuroevolution** — búsqueda de la mejor arquitectura de una red neuronal.

---

## 1. Estructura del repositorio

```
├── 01_Feature_Selection_AG.ipynb          # AG para selección de características
├── 02_Hyperparameter_Optimization_AG.ipynb # AG para optimización de hiperparámetros
├── 03_Neuroevolution_AG.ipynb              # AG para neuroevolución (arquitectura de red neuronal)
├── Resumen_Ejecutivo_AG_ML.pdf             # Resumen ejecutivo (máx. 2 hojas)
└── README.md                               # Este archivo
```

Cada notebook es **autocontenido**: carga el dataset internamente desde scikit-learn (no depende de archivos externos ni de conexión a internet) y puede abrirse y ejecutarse directamente en **Google Colab** sin instalar nada adicional (usa librerías ya preinstaladas: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tensorflow`, `scipy`).

## 2. Dataset utilizado 

Para dar continuidad y comparabilidad entre los tres experimentos, se usa el **mismo dataset real** en los tres notebooks:

> **Breast Cancer Wisconsin (Diagnostic)** — `sklearn.datasets.load_breast_cancer()`

- **569 pacientes** (registros) × **30 variables predictoras reales**, calculadas a partir de imágenes digitalizadas de biopsias por aspiración con aguja fina (FNA) de masas mamarias.
- Las 30 variables corresponden a **10 medidas físicas del núcleo celular** (radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría, dimensión fractal), cada una reportada en **3 versiones**: valor medio (*mean*), error estándar (*error*) y peor caso (*worst*) — de ahí que existan variables naturalmente **redundantes** entre sí (p. ej. `mean radius`, `radius error`, `worst radius`).
- **Variable objetivo:** diagnóstico — **1 = maligno** (212 casos), **0 = benigno** (357 casos).
- Es uno de los datasets reales más usados en la literatura de Machine Learning para clasificación binaria (UCI / Wisconsin Diagnostic Breast Cancer), y viene incluido directamente en scikit-learn, por lo que no requiere descarga externa.

Este dataset es especialmente adecuado para el ejercicio de **feature selection** por su redundancia natural (mean/error/worst de las mismas 10 medidas), y suficientemente liviano para permitir búsquedas evolutivas rápidas en **hyperparameter optimization** y **neuroevolution**.

## 3. ¿Qué es un Algoritmo Genético y qué elementos tiene el ciclo?

Un Algoritmo Genético es una metaheurística de optimización inspirada en la selección natural. Mantiene una **población** de soluciones candidatas ("cromosomas") que evoluciona a lo largo de **generaciones**, mediante:

| Etapa | Descripción general |
|---|---|
| **Representación (cromosoma)** | Cómo se codifica una solución candidata (vector binario, vector numérico, diccionario de parámetros, etc.). |
| **Inicialización** | Cómo se genera la población inicial (usualmente al azar). |
| **Función de aptitud (fitness)** | Cómo se mide qué tan buena es una solución candidata (en ML: desempeño de un modelo entrenado con esa configuración). |
| **Selección** | Cómo se eligen los "padres" que se reproducirán (aquí: selección por torneo). |
| **Cruce (crossover)** | Cómo se combinan dos padres para producir hijos. |
| **Mutación** | Cómo se introduce variación aleatoria para explorar nuevas soluciones y evitar estancamiento. |
| **Elitismo** | Los mejores individuos de una generación pasan intactos a la siguiente, para no perder el mejor resultado encontrado. |
| **Terminación** | Cuándo se detiene la búsqueda (número máximo de generaciones o falta de mejora durante varias generaciones consecutivas). |

Los tres notebooks de este proyecto implementan **el mismo ciclo genérico**, adaptando únicamente la representación del cromosoma y la función de aptitud a cada problema.

## 4. Resumen de cada notebook

### 4.1 `01_Feature_Selection_AG.ipynb`

- **Cromosoma:** vector binario de 30 genes (1 = variable incluida, 0 = excluida).
- **Fitness:** F1-score (validación cruzada 5-fold) de una Regresión Logística entrenada solo con las variables activas, con una leve penalización por número de variables.
- **Selección / Cruce / Mutación:** torneo (k=3) / cruce uniforme / bit-flip (2%).
- **Terminación:** 25 generaciones máx., o 6 sin mejora.
- **Resultado obtenido:** el AG redujo el conjunto de **30 a 14 variables** (descartando variables redundantes como varias combinaciones de radio/perímetro/área que aportaban información repetida), mejorando el F1-score de **0.9690** (todas las variables) a **0.9784** (subconjunto óptimo).

### 4.2 `02_Hyperparameter_Optimization_AG.ipynb`

- **Cromosoma:** vector mixto entero/real con 5 hiperparámetros de un `RandomForestClassifier`: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- **Fitness:** F1-score (validación cruzada 3-fold), con **caché** de resultados para no reevaluar combinaciones repetidas.
- **Selección / Cruce / Mutación:** torneo (k=3) / cruce aritmético / mutación gaussiana.
- **Terminación:** 10 generaciones máx., o 4 sin mejora.
- **Resultado obtenido:** con el mismo presupuesto de evaluaciones, el AG (F1 = **0.9383**) resultó **muy cercano** a los hiperparámetros por defecto de scikit-learn (F1 = 0.9424) y a Random Search con igual número de iteraciones (F1 = 0.9424). El hallazgo honesto es que, para este dataset (relativamente pequeño y bien separable), el Random Forest ya es bastante robusto a su configuración de hiperparámetros — un resultado empírico legítimo y también valioso de reportar.

### 4.3 `03_Neuroevolution_AG.ipynb`

- **Cromosoma:** diccionario que codifica una arquitectura de red neuronal (Keras/TensorFlow): número de capas ocultas (1 a 3), neuronas por capa (8–64), función de activación (`relu`/`tanh`), tasa de dropout y tasa de aprendizaje.
- **Fitness:** accuracy de validación tras entrenar 8 épocas (búsqueda rápida), con penalización leve por número de parámetros del modelo.
- **Selección / Cruce / Mutación:** torneo (k=3) / cruce uniforme / mutación específica por tipo de gen.
- **Terminación:** 6 generaciones máx., o 3 sin mejora.
- **Resultado obtenido:** la arquitectura hallada por el AG (3 capas: 43-64-21 neuronas, activación ReLU, sin dropout) alcanzó una accuracy de validación de **0.979** tras reentrenar 25 épocas, superando claramente a una arquitectura definida manualmente (2 capas de 32 y 16 neuronas), que alcanzó **0.958**.

## 5. Comparación entre los tres enfoques

| | Feature Selection | Hyperparameter Optimization | Neuroevolution |
|---|---|---|---|
| **Qué codifica el cromosoma** | Subconjunto de variables | Configuración de un modelo clásico (RF) | Arquitectura de una red neuronal |
| **Costo de evaluar un individuo** | Bajo (varios folds de CV con un modelo simple) | Medio (varios folds de CV con un ensemble) | Alto (entrenar una red neuronal, aunque sea pocas épocas) |
| **Alternativas típicas** | Forward/Backward selection, RFE, filtros estadísticos | Grid Search, Random Search, Bayesian Optimization | Búsqueda manual, Random Search de arquitecturas, NAS/ENAS |
| **Ventaja principal del AG** | Explora combinaciones no obvias de variables redundantes sin necesitar 2^30 evaluaciones | Aprovecha información de combinaciones previas (vs. Random Search "a ciegas") | Automatiza decisiones de diseño (capas, neuronas) sin descenso de gradiente |
| **Resultado en este trabajo** | F1: 0.9690 → 0.9784 (30→14 variables) | AG ≈ default ≈ Random Search (~0.94) | Accuracy: 0.958 → 0.979 |

## 6. Cómo ejecutar los notebooks

1. Abrir cada `.ipynb` directamente en [Google Colab](https://colab.research.google.com/) (Archivo → Subir notebook, o arrastrar el archivo).
2. Ejecutar las celdas en orden (Entorno de ejecución → Ejecutar todas). No requieren archivos externos ni GPU.
3. Tiempos de ejecución aproximados en un entorno estándar de Colab (CPU), usando el dataset real (569 filas):
   - Notebook 1 (Feature Selection): ~6 segundos.
   - Notebook 2 (Hyperparameter Optimization): ~20-30 segundos.
   - Notebook 3 (Neuroevolution): ~1-2 minutos.

## 7. Requisitos / librerías usadas

`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`, `tensorflow` (Keras) — todas preinstaladas en Google Colab. El dataset se carga con `sklearn.datasets.load_breast_cancer()`, incluido en scikit-learn (no requiere descarga externa).

## 8. Conclusión general del trabajo

Los tres notebooks comparten la misma estructura conceptual de un Algoritmo Genético (representación → inicialización → fitness → selección → cruce → mutación → terminación), lo que demuestra que se trata de una **metaheurística de propósito general**: basta con redefinir qué representa el cromosoma y cómo se mide su aptitud para aplicarla a problemas de optimización muy distintos dentro del ciclo de vida de un modelo de Machine Learning. Sobre el dataset real Breast Cancer Wisconsin, el AG mostró mejoras claras en selección de características (F1 +0.94 pp con menos de la mitad de las variables) y en neuroevolución (+2.1 pp de accuracy), y un resultado competitivo (aunque no claramente superior) en optimización de hiperparámetros frente a Random Search — lo cual es también una conclusión válida: **el AG no siempre gana**, pero ofrece una alternativa sistemática y explicable a la búsqueda manual o exhaustiva.

Para una síntesis más breve, ver `Resumen_Ejecutivo_AG_ML.pdf` (máximo 2 hojas).
