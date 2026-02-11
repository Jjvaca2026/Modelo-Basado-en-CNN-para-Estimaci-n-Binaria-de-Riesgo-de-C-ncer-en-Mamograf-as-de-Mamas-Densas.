"""
SCRIPT UNIFICADO DE ENTRENAMIENTO, EVALUACIÓN Y RECUPERACIÓN
DE EXPERIMENTOS CNN CON IMÁGENES PREPROCESADAS (WAVELET)

Autor: Jhon Vaca

Descripción:
Este script implementa un flujo completo y reproducible para la evaluación
de múltiples arquitecturas CNN sobre imágenes mamográficas preprocesadas
mediante Wavelet.

El script integra en un solo archivo:
1. Ejecución completa de experimentos desde cero
2. Reanudación automática de experimentos interrumpidos
3. Reconstrucción del progreso a partir de resultados parciales

Las rutas de entrada y salida dependen de la estructura local del usuario.
"""

# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================
import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import (
    DenseNet121, ResNet50, InceptionV3, EfficientNetB0
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    precision_recall_curve,
    auc
)
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# 2. CONFIGURACIÓN GENERAL DEL USUARIO
# ============================================================

# Modo de ejecución:
# "normal"       -> Ejecuta todos los experimentos desde cero
# "reanudar"     -> Continúa desde progreso.pkl
# "reconstruir"  -> Reconstruye progreso a partir de JSON existentes
MODO_EJECUCION = "reanudar"

# Ruta base donde están las imágenes preprocesadas
BASE_DATASET = "/content/drive/MyDrive/Tesis Maestría/PP/Wavelet"

# Ruta donde se guardarán resultados, modelos y métricas
BASE_RESULTADOS = "/content/drive/MyDrive/Tesis Maestría/Resultados_Wavelet"

# Clases del problema
CLASES = ["Bajo_Riesgo", "Alto_Riesgo"]

# Tamaño de imagen
IMG_SIZE = (224, 224)

# Semilla para reproducibilidad
SEED = 42

# Archivos de control
PROGRESO_PATH = os.path.join(BASE_RESULTADOS, "progreso.pkl")

os.makedirs(BASE_RESULTADOS, exist_ok=True)

# ============================================================
# 3. DEFINICIÓN DE ARQUITECTURAS Y CONFIGURACIONES
# ============================================================

ARQUITECTURAS = {
    "DenseNet121": DenseNet121,
    "ResNet50": ResNet50,
    "InceptionV3": InceptionV3,
    "EfficientNetB0": EfficientNetB0
}

CONFIGURACIONES = {
    "BASELINE": {
        "batch_size": 16,
        "epochs": 10,
        "learning_rate": 1e-4,
        "dropout": 0.5,
        "monitor": "val_recall",
        "patience": 4,
        "class_weight": "balanced"
    },
    "OPTIMIZADA": {
        "batch_size": 8,
        "epochs": 12,
        "learning_rate": 5e-5,
        "dropout": 0.4,
        "monitor": "val_auc",
        "patience": 5,
        "class_weight": "custom"
    }
}

# ============================================================
# 4. FUNCIONES DE SOPORTE
# ============================================================

def fijar_semillas(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def cargar_imagenes():
    imagenes = []
    etiquetas = []

    for idx, clase in enumerate(CLASES):
        carpeta = os.path.join(BASE_DATASET, clase)
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
                imagenes.append(os.path.join(carpeta, archivo))
                etiquetas.append(idx)

    return imagenes, np.array(etiquetas)

def crear_dataset(rutas, etiquetas, batch_size, entrenamiento=True):
    def _parse(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.image.grayscale_to_rgb(img)
        img = img / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if entrenamiento:
        ds = ds.shuffle(1000, seed=SEED)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def construir_modelo(nombre_arq, dropout, lr):
    base_model = ARQUITECTURAS[nombre_arq](
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(base_model.input, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model

def calcular_metricas(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recall, precision)

    return cm.tolist(), mcc, auc_roc, auc_pr

# ============================================================
# 5. MANEJO DE PROGRESO
# ============================================================

def cargar_progreso():
    if os.path.exists(PROGRESO_PATH):
        with open(PROGRESO_PATH, "rb") as f:
            return pickle.load(f)
    return {"completados": set(), "resultados": []}

def guardar_progreso(progreso):
    with open(PROGRESO_PATH, "wb") as f:
        pickle.dump(progreso, f)

def reconstruir_progreso():
    progreso = {"completados": set(), "resultados": []}
    for archivo in os.listdir(BASE_RESULTADOS):
        if archivo.endswith(".json"):
            with open(os.path.join(BASE_RESULTADOS, archivo)) as f:
                data = json.load(f)
            progreso["resultados"].append(data)
            progreso["completados"].add(data["experimento"])
    guardar_progreso(progreso)
    return progreso

# ============================================================
# 6. EJECUCIÓN PRINCIPAL
# ============================================================

fijar_semillas(SEED)

if MODO_EJECUCION == "reconstruir":
    progreso = reconstruir_progreso()
elif MODO_EJECUCION == "reanudar":
    progreso = cargar_progreso()
else:
    progreso = {"completados": set(), "resultados": []}

imagenes, etiquetas = cargar_imagenes()

X_train, X_temp, y_train, y_temp = train_test_split(
    imagenes, etiquetas, test_size=0.3, stratify=etiquetas, random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)

# ============================================================
# 7. BUCLE DE EXPERIMENTOS
# ============================================================

for nombre_arq in ARQUITECTURAS:
    for nombre_cfg, cfg in CONFIGURACIONES.items():

        exp_id = f"{nombre_arq}_{nombre_cfg}"

        if exp_id in progreso["completados"]:
            continue

        train_ds = crear_dataset(X_train, y_train, cfg["batch_size"], True)
        val_ds = crear_dataset(X_val, y_val, cfg["batch_size"], False)
        test_ds = crear_dataset(X_test, y_test, cfg["batch_size"], False)

        model = construir_modelo(
            nombre_arq,
            cfg["dropout"],
            cfg["learning_rate"]
        )

        cb = callbacks.EarlyStopping(
            monitor=cfg["monitor"],
            patience=cfg["patience"],
            restore_best_weights=True
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg["epochs"],
            callbacks=[cb],
            verbose=1
        )

        y_prob = model.predict(test_ds).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        cm, mcc, auc_roc, auc_pr = calcular_metricas(y_test, y_pred, y_prob)

        resultado = {
            "experimento": exp_id,
            "arquitectura": nombre_arq,
            "configuracion": nombre_cfg,
            "mcc": mcc,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "matriz_confusion": cm
        }

        progreso["resultados"].append(resultado)
        progreso["completados"].add(exp_id)

        with open(os.path.join(BASE_RESULTADOS, f"{exp_id}.json"), "w") as f:
            json.dump(resultado, f, indent=4)

        guardar_progreso(progreso)

# ============================================================
# 8. EXPORTACIÓN FINAL
# ============================================================

df_final = pd.DataFrame(progreso["resultados"])
df_final.to_csv(
    os.path.join(BASE_RESULTADOS, "resultados_finales_wavelet.csv"),
    index=False
)
