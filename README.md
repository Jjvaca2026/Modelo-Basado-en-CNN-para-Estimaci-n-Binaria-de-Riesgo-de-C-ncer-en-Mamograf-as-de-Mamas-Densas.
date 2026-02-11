# Modelo Basado en CNN para Estimación Binaria de Riesgo de Cáncer en Mamografías Digitales de Mamas Densas, Utilizando Criterios BI-RADS como Referencia

Este repositorio contiene el código, scripts y recursos técnicos asociados al trabajo de grado de maestría:

**“Modelo Basado en CNN para Estimación Binaria de Riesgo de Cáncer en Mamografías Digitales de Mamas Densas, Utilizando Criterios BI-RADS como Referencia”**

El objetivo del repositorio es respaldar la **reproducibilidad metodológica** del trabajo, no servir como sistema clínico ni como repositorio de datos médicos.

## Alcance del repositorio

- Uso **exclusivamente académico**
- Enfoque **metodológico y experimental**
- No contiene datos clínicos sensibles
- No incluye modelos entrenados finales listos para despliegue

Los resultados aquí presentados corresponden a los experimentos descritos y discutidos en el documento de tesis.

## Estructura del repositorio

├── src/

│ ├── preprocessing/ # Scripts de preprocesamiento de mamografías

│ ├── datasets/ # Construcción y normalización de datasets

│ ├── models/ # Definición de arquitecturas y wrappers

│ ├── evaluation/ # Métricas y evaluación experimental

│ └── utils/ # Funciones auxiliares

│

├── experiments/

│ ├── baseline/ # Experimentos base

│ ├── finetuning/ # Ajuste fino de modelos

│ └── external_validation/ # Validación externa

│

├── notebooks/

│ ├── exploratory_analysis.ipynb

│ └── experiments_summary.ipynb

│

├── assets/

│ ├── figures/ # Figuras utilizadas en la tesis

│ └── saliency_examples/ # Ejemplos de mapas de saliencia

│

├── requirements.txt

└── README.md


## Datos y modelos

Por razones **éticas, legales y de licenciamiento**, este repositorio **no incluye**:

- Mamografías originales
- Datos clínicos o metadatos sensibles
- Modelos entrenados finales (`.h5`, checkpoints completos)

Los experimentos fueron realizados utilizando **datasets públicos**, siguiendo las licencias correspondientes.  
Los scripts incluidos permiten reproducir el pipeline metodológico empleando datos equivalentes.

---

## Metodología general

El trabajo se centra en una **clasificación binaria de riesgo** en mamografías de mamas densas, priorizando métricas clínicamente relevantes como la sensibilidad y la tasa de falsos negativos.

El pipeline general incluye:

1. Preprocesamiento y normalización de imágenes
2. Selección objetiva de técnicas de preprocesamiento
3. Entrenamiento y ajuste fino de modelos de aprendizaje profundo
4. Evaluación mediante validación cruzada
5. Análisis de resultados y mapas de saliencia

Los detalles completos se encuentran documentados en la tesis.

---

## Reproducibilidad

El repositorio está diseñado para facilitar la **reproducibilidad experimental** a nivel metodológico.  
Los parámetros principales, configuraciones y métricas de evaluación se encuentran documentados en los scripts y archivos de resultados resumidos.

No se garantiza reproducibilidad numérica exacta debido a:
- Naturaleza estocástica del entrenamiento
- Diferencias en hardware y versiones de librerías

---

## Consideraciones éticas

Este trabajo **no constituye una herramienta clínica**, no reemplaza la evaluación médica especializada y no ha sido validado para uso diagnóstico.

El propósito del estudio es **académico e investigativo**.

---

## Licencia

Uso académico y educativo.  
Cualquier reutilización debe citar el trabajo original de tesis.

