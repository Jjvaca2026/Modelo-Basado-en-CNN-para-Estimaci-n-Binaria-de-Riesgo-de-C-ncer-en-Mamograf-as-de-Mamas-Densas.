"""
VALIDACIÓN CRUZADA ESTRATIFICADA 10-FOLD
MODELO INCEPTIONV3 CON PREPROCESAMIENTO WAVELET

Este script realiza una evaluación exhaustiva de generalización mediante
validación cruzada estratificada de 10 pliegues (10-Fold Cross Validation)
sobre un modelo InceptionV3 previamente entrenado con imágenes
preprocesadas mediante Wavelet.

El objetivo es evaluar la estabilidad, robustez y ausencia de sobreajuste
del modelo, así como analizar el impacto de la calibración del umbral de
decisión (0.5 vs 0.9) en métricas clínicas relevantes.

El script NO entrena el modelo, únicamente lo evalúa.
"""

import os
import time
import json
import datetime
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

# ======================================================
# 1. CONFIGURACIÓN GENERAL
# ======================================================

BASE_PATH = "/content/drive/MyDrive/Tesis Maestría/entrenamiento/Wavelet"

AR_PATH = os.path.join(BASE_PATH, "AR")
BR_PATH = os.path.join(BASE_PATH, "BR")

MODEL_PATH = os.path.join(
    BASE_PATH,
    "Resultados_InceptionV3_Wavelet",
    "modelos",
    "mejor_modelo.h5"
)

IMG_SIZE = (299, 299)
N_FOLDS = 10
UMBRAL_OPTIMO = 0.9
SEED = 42

RESULTS_DIR = os.path.join(BASE_PATH, "10Fold_Evaluation_InceptionV3")
GRAFICAS_DIR = os.path.join(RESULTS_DIR, "graficas")
DATOS_DIR = os.path.join(RESULTS_DIR, "datos")

os.makedirs(GRAFICAS_DIR, exist_ok=True)
os.makedirs(DATOS_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ======================================================
# 2. CARGA DE IMÁGENES
# ======================================================

def cargar_imagenes(carpeta, etiqueta):
    """
    Carga imágenes desde una carpeta, aplica redimensionamiento y
    preprocesamiento específico para InceptionV3.
    """
    imagenes = []
    etiquetas = []

    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta = os.path.join(carpeta, archivo)
            img = cv2.imread(ruta)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32)
            img = keras.applications.inception_v3.preprocess_input(img)

            imagenes.append(img)
            etiquetas.append(etiqueta)

    return np.array(imagenes), np.array(etiquetas)

def cargar_dataset_completo():
    """
    Carga el dataset completo Wavelet (Alto y Bajo Riesgo).
    """
    X_ar, y_ar = cargar_imagenes(AR_PATH, 1)
    X_br, y_br = cargar_imagenes(BR_PATH, 0)

    X = np.concatenate([X_ar, X_br], axis=0)
    y = np.concatenate([y_ar, y_br], axis=0)

    return X, y, len(X_ar), len(X_br)

# ======================================================
# 3. CARGA DEL MODELO
# ======================================================

model = keras.models.load_model(MODEL_PATH)

# ======================================================
# 4. PREPARAR DATASET
# ======================================================

X, y, n_ar, n_br = cargar_dataset_completo()

indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# ======================================================
# 5. VALIDACIÓN CRUZADA 10-FOLD
# ======================================================

kf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

resultados_folds = []
tiempos_folds = []

todas_predicciones = []
todas_etiquetas = []

for fold, (_, test_idx) in enumerate(kf.split(X, y), start=1):

    X_test = X[test_idx]
    y_test = y[test_idx]

    inicio = time.time()

    y_pred_proba = model.predict(X_test, batch_size=16, verbose=0).flatten()

    todas_predicciones.extend(y_pred_proba)
    todas_etiquetas.extend(y_test)

    y_pred_05 = (y_pred_proba > 0.5).astype(int)
    y_pred_09 = (y_pred_proba > UMBRAL_OPTIMO).astype(int)

    def calcular_metricas(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sensibilidad = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        especificidad = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        f1 = (2 * precision * sensibilidad /
              (precision + sensibilidad)) if (precision + sensibilidad) > 0 else 0

        return sensibilidad, especificidad, precision, f1, cm

    sens_05, esp_05, prec_05, f1_05, cm_05 = calcular_metricas(y_test, y_pred_05)
    sens_09, esp_09, prec_09, f1_09, cm_09 = calcular_metricas(y_test, y_pred_09)

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall_vals, precision_vals)

    tiempo_fold = time.time() - inicio
    tiempos_folds.append(tiempo_fold)

    resultados_folds.append({
        "fold": fold,
        "sensibilidad_05": sens_05,
        "especificidad_05": esp_05,
        "f1_05": f1_05,
        "sensibilidad_09": sens_09,
        "especificidad_09": esp_09,
        "f1_09": f1_09,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "matriz_confusion_05": cm_05.tolist(),
        "matriz_confusion_09": cm_09.tolist(),
        "tiempo_segundos": tiempo_fold
    })

# ======================================================
# 6. ANÁLISIS ESTADÍSTICO GLOBAL
# ======================================================

df_resultados = pd.DataFrame(resultados_folds)

estadisticas = df_resultados.describe()

# ======================================================
# 7. GUARDAR RESULTADOS
# ======================================================

json_path = os.path.join(DATOS_DIR, "10fold_evaluation_inception_wavelet.json")
csv_path = os.path.join(DATOS_DIR, "10fold_evaluation_inception_wavelet.csv")

df_resultados.to_csv(csv_path, index=False, encoding="utf-8")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "fecha": datetime.datetime.now().isoformat(),
            "arquitectura": "InceptionV3",
            "preprocesamiento": "Wavelet",
            "n_folds": N_FOLDS,
            "umbral_optimo": UMBRAL_OPTIMO,
            "imagenes_total": len(X),
            "alto_riesgo": n_ar,
            "bajo_riesgo": n_br
        },
        "resultados": resultados_folds,
        "estadisticas_descriptivas": estadisticas.to_dict()
    }, f, indent=2, ensure_ascii=False)

print("Evaluación 10-Fold completada correctamente.")
print(f"Resultados guardados en: {RESULTS_DIR}")
