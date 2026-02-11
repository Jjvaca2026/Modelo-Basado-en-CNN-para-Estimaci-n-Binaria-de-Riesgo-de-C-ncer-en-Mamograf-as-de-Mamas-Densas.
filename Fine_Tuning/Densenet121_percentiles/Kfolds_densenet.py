"""
10-FOLD CROSS VALIDATION
Evaluación robusta de generalización del modelo entrenado

Autor: Jhon Vaca
Objetivo:
Realizar una validación cruzada estratificada de 10 folds sobre un modelo
entrenado previamente, con el fin de evaluar su estabilidad, capacidad
de generalización y posible sobreajuste.
"""

# ====================
# IMPORTACIÓN DE LIBRERÍAS
# ====================
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    precision_recall_curve, auc
)
import time
import os
import cv2
import json
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("=" * 80)
print("10-FOLD CROSS VALIDATION")
print("Evaluación de generalización mediante validación cruzada estratificada")
print("=" * 80)

# ====================
# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ====================
# Ruta del modelo entrenado (ajustar según la estructura del proyecto)
MODEL_PATH = (
    "/content/drive/MyDrive/Tesis Maestría/"
    "entrenamiento/Percentiles/Resultados_FineTuning_Final/mejor_modelo.h5"
)

# Rutas de los datasets preprocesados
AR_PATH = (
    "/content/drive/MyDrive/Tesis Maestría/"
    "entrenamiento/Percentiles/AR"
)
BR_PATH = (
    "/content/drive/MyDrive/Tesis Maestría/"
    "entrenamiento/Percentiles/BR"
)

IMG_SIZE = (224, 224)
UMBRAL_OPTIMO = 0.90
N_FOLDS = 10

RESULTS_DIR = (
    "/content/drive/MyDrive/Tesis Maestría/"
    "entrenamiento/Percentiles/10Fold_Evaluation"
)
os.makedirs(os.path.join(RESULTS_DIR, "graficas"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "datos"), exist_ok=True)

# ====================
# 2. CARGA DE DATOS
# ====================
def cargar_imagenes(carpeta, label):
    """Carga y preprocesa imágenes desde una carpeta."""
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
            img = img.astype(np.float32) / 255.0
            imagenes.append(img)
            etiquetas.append(label)

    return np.array(imagenes), np.array(etiquetas)

print("\nCargando dataset completo...")
X_ar, y_ar = cargar_imagenes(AR_PATH, 1)
X_br, y_br = cargar_imagenes(BR_PATH, 0)

X = np.concatenate([X_ar, X_br], axis=0)
y = np.concatenate([y_ar, y_br], axis=0)

indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

print(f"Total de imágenes: {len(X)}")

# ====================
# 3. CARGA DEL MODELO
# ====================
print("\nCargando modelo entrenado...")
model = keras.models.load_model(MODEL_PATH)

# ====================
# 4. VALIDACIÓN CRUZADA 10-FOLD
# ====================
kf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=42
)

resultados_folds = []
tiempos_folds = []

for fold, (_, test_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nEvaluando fold {fold}/{N_FOLDS}")

    X_test = X[test_idx]
    y_test = y[test_idx]

    start_time = time.time()

    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred_09 = (y_pred_proba > UMBRAL_OPTIMO).astype(int)

    cm = confusion_matrix(y_test, y_pred_09)
    tn, fp, fn, tp = cm.ravel()

    sensibilidad = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1 = (
        2 * precision * sensibilidad / (precision + sensibilidad)
        if (precision + sensibilidad) > 0 else 0
    )

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(rec_vals, prec_vals)

    tiempo = time.time() - start_time
    tiempos_folds.append(tiempo)

    resultados_folds.append({
        "fold": fold,
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "precision": precision,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "tiempo_segundos": tiempo
    })

# ====================
# 5. ANÁLISIS ESTADÍSTICO
# ====================
df_resultados = pd.DataFrame(resultados_folds)

estadisticas = df_resultados.describe().loc[
    ['mean', 'std', 'min', 'max']
]

csv_path = os.path.join(
    RESULTS_DIR, "datos", "resultados_10fold.csv"
)
df_resultados.to_csv(csv_path, index=False, encoding="utf-8")

json_path = os.path.join(
    RESULTS_DIR, "datos", "resultados_10fold.json"
)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(resultados_folds, f, indent=2)

print("\nEvaluación 10-Fold completada.")
print(f"Resultados guardados en: {RESULTS_DIR}")
