## Requerimientos de Software y Dependencias

El desarrollo experimental de este proyecto se realizó utilizando Python y
librerías de código abierto ampliamente adoptadas en el análisis de imágenes
médicas y aprendizaje profundo. Debido a la naturaleza computacional del
entrenamiento y procesamiento de imágenes, se emplearon dos entornos de ejecución
diferenciados: un entorno local para el preprocesamiento inicial y un entorno
en la nube (Google Colab) para el procesamiento intensivo y entrenamiento de modelos.

### Entorno Local (Preprocesamiento de imágenes DICOM)

Este entorno fue utilizado para la carga de metadatos clínicos, filtrado de casos,
conversión de imágenes DICOM a formato PNG normalizado y generación de datasets
estructurados.

- Python ≥ 3.8
- Jupyter
- numpy ≥ 1.20
- pandas ≥ 1.3
- pydicom ≥ 2.3
- pillow ≥ 8.0
- matplotlib ≥ 3.4
- seaborn ≥ 0.11


### Entorno en la Nube (Google Colab – GPU/TPU)

Este entorno fue utilizado para el preprocesamiento avanzado de imágenes,
cálculo de métricas objetivas de calidad, análisis de clustering no supervisado
y entrenamiento de modelos de aprendizaje profundo mediante transferencia de aprendizaje.

#### Procesamiento de Imágenes
- opencv-python-headless ≥ 4.5
- scikit-image ≥ 0.19
- pywavelets ≥ 1.3

#### Análisis de Datos y Métricas
- numpy ≥ 1.20
- pandas ≥ 1.3
- scipy ≥ 1.6
- tqdm ≥ 4.60

#### Aprendizaje Automático y Clustering
- scikit-learn ≥ 1.0

#### Aprendizaje Profundo
- tensorflow ≥ 2.8,<2.13
- keras ≥ 2.8

#### Visualización
- matplotlib ≥ 3.4
- seaborn ≥ 0.11


### Consideraciones de Reproducibilidad

Las versiones exactas de las librerías no se fijan de forma estricta debido a:
- la evolución constante de los entornos en la nube,
- la naturaleza estocástica del entrenamiento de modelos de aprendizaje profundo,
- y la dependencia del hardware acelerado (GPU/TPU).

No obstante, las dependencias listadas corresponden a las librerías efectivamente
utilizadas durante el desarrollo experimental y permiten reproducir el pipeline
metodológico descrito en la tesis.
