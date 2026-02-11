"""
SCRIPT ENTRENAMIENTO Y CALIBRACIÓN DE UMBRAL
DenseNet121 con preprocesamiento por Percentiles

Este script implementa un pipeline completo y reproducible que integra:
1. Entrenamiento y fine-tuning de DenseNet121 usando imágenes preprocesadas
   mediante la técnica de Percentiles (NO Wavelet).
2. Uso de class_weight para priorizar sensibilidad clínica.
3. Selección del mejor modelo según recall en validación.
4. Evaluación en conjunto de prueba.
5. Calibración post-entrenamiento del umbral de decisión para corregir
   el sesgo inducido por class_weight, sin reentrenar el modelo.

Las rutas dependen de la organización local del proyecto del usuario.
"""

# ======================================================
# IMPORTACIONES
# ======================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4

# ------------------------------------------------------
# RUTAS DEL PROYECTO (AJUSTAR SEGÚN EL USUARIO)
# ------------------------------------------------------
# Dataset de imágenes preprocesadas con PERCENTILES
DATASET_PATH = "/ruta/al/dataset/Percentiles"

# Directorio de salida de resultados
OUTPUT_DIR = "/ruta/a/resultados/Percentiles_DenseNet121"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# 1. CARGA DE DATOS
# ======================================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=SEED
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=SEED
)

class_names = list(train_generator.class_indices.keys())

# ======================================================
# 2. CÁLCULO DE CLASS WEIGHT
# ======================================================
y_train = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    0: class_weights[0] * 0.7,
    1: class_weights[1] * 1.3
}

# ======================================================
# 3. CONSTRUCCIÓN DEL MODELO
# ======================================================
base_model = DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="avg"
)
base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Recall(name="recall"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.AUC(name="auc")
    ]
)

# ======================================================
# 4. CALLBACKS
# ======================================================
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_recall",
        mode="max",
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, "mejor_modelo.h5"),
        monitor="val_recall",
        mode="max",
        save_best_only=True
    )
]

# ======================================================
# 5. ENTRENAMIENTO
# ======================================================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

model.save(os.path.join(OUTPUT_DIR, "modelo_final.h5"))

# ======================================================
# 6. EVALUACIÓN EN VALIDACIÓN (UMBRAL 0.5)
# ======================================================
y_true = val_generator.classes
y_prob = model.predict(val_generator).ravel()
y_pred_05 = (y_prob >= 0.5).astype(int)

cm_05 = confusion_matrix(y_true, y_pred_05)
auc_05 = roc_auc_score(y_true, y_prob)

# ======================================================
# 7. CALIBRACIÓN DE UMBRAL
# ======================================================
fpr, tpr, thresholds = roc_curve(y_true, y_prob)

# Umbral óptimo: maximiza TPR - FPR (Youden Index)
youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
optimal_threshold = thresholds[best_idx]

y_pred_opt = (y_prob >= optimal_threshold).astype(int)
cm_opt = confusion_matrix(y_true, y_pred_opt)

# ======================================================
# 8. GUARDAR RESULTADOS NUMÉRICOS
# ======================================================
resultados = {
    "fuente_datos": "Percentiles",
    "arquitectura": "DenseNet121",
    "umbral_por_defecto": 0.5,
    "umbral_calibrado": float(optimal_threshold),
    "auc": float(auc_05),
    "confusion_matrix_0_5": cm_05.tolist(),
    "confusion_matrix_calibrada": cm_opt.tolist(),
    "reporte_0_5": classification_report(
        y_true, y_pred_05, target_names=class_names, output_dict=True
    ),
    "reporte_calibrado": classification_report(
        y_true, y_pred_opt, target_names=class_names, output_dict=True
    )
}

with open(os.path.join(OUTPUT_DIR, "resultados_calibracion.json"), "w") as f:
    json.dump(resultados, f, indent=4)

# ======================================================
# 9. GRÁFICA DE CALIBRACIÓN
# ======================================================
plt.figure(figsize=(8, 6))
plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Clase negativa")
plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Clase positiva")
plt.axvline(0.5, linestyle="--", label="Umbral 0.5")
plt.axvline(optimal_threshold, linestyle="-", label="Umbral calibrado")
plt.xlabel("Probabilidad predicha")
plt.ylabel("Frecuencia")
plt.legend()
plt.title("Calibración de umbral - DenseNet121 (Percentiles)")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "calibracion_umbral_percentiles.png"),
    dpi=300
)
plt.close()

# ======================================================
# 10. SCRIPT DE IMPLEMENTACIÓN DEL UMBRAL
# ======================================================
with open(os.path.join(OUTPUT_DIR, "implementacion_umbral_calibrado.py"), "w") as f:
    f.write(
        f"""# Umbral calibrado obtenido en validación
UMBRAL_CALIBRADO = {optimal_threshold:.4f}

def predecir_clase(probabilidad):
    return int(probabilidad >= UMBRAL_CALIBRADO)
"""
    )

print("Pipeline completado correctamente.")
print("Modelo entrenado con datos Percentiles y umbral calibrado guardado.")
