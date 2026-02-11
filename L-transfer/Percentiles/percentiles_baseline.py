"""

Este script implementa la Fase 2 del estudio experimental, evaluando el
preprocesamiento por percentiles aplicado a imágenes médicas, utilizando
exclusivamente la configuración BASELINE de diferentes arquitecturas CNN.

La decisión de evaluar únicamente la configuración BASELINE se fundamenta
en los resultados obtenidos en la Fase 1 (Wavelet), donde se evidenció que
configuraciones más complejas no presentaban mejoras consistentes y
mostraban menor estabilidad.

El script:
- Verifica la existencia del dataset
- Carga y cachea las imágenes
- Entrena y evalúa modelos CNN preentrenados
- Calcula métricas de desempeño clínicamente relevantes
- Guarda resultados individuales y consolidados

IMPORTANTE:
Las rutas de acceso al dataset y de almacenamiento de resultados deben
ajustarse según la estructura de carpetas del usuario.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
import gc
import time
from pathlib import Path

# ====================
# MONTAJE DE GOOGLE DRIVE
# ====================
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ====================
# CONFIGURACIÓN GENERAL
# ====================

# Ruta base del dataset preprocesado por percentiles
# Debe contener una carpeta por clase
BASE_PATH = "/content/drive/MyDrive/Tesis Maestría/PP/Percentiles"

# Nombres de las clases (una carpeta por clase)
CLASES = ["Bajo_Riesgo", "Alto_Riesgo"]

# Tamaño de las imágenes de entrada
IMG_SIZE = (224, 224)

# Semilla para reproducibilidad
SEED = 42

# Directorio donde se almacenan todos los resultados
RESULTS_DIR = "/content/drive/MyDrive/Tesis Maestría/PP/Resultados_Percentiles_Optimizado"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Archivo de control de progreso (no obligatorio para análisis)
PROGRESS_FILE = os.path.join(RESULTS_DIR, "progreso_baseline.pkl")

# ====================
# 1. VERIFICACIÓN DEL DATASET
# ====================

def verificar_dataset():
    """
    Verifica que existan las carpetas de cada clase
    y que contengan imágenes válidas.
    """
    for clase in CLASES:
        ruta_clase = os.path.join(BASE_PATH, clase)

        if not os.path.exists(ruta_clase):
            print(f"ERROR: No existe la ruta {ruta_clase}")
            return False

        extensiones = ('.png', '.jpg', '.jpeg')
        archivos = [f for f in os.listdir(ruta_clase)
                    if f.lower().endswith(extensiones)]

        print(f"{clase}: {len(archivos)} imágenes encontradas")

        if len(archivos) == 0:
            print(f"Advertencia: La clase {clase} no contiene imágenes")

    return True


if not verificar_dataset():
    print("El dataset no cumple las condiciones mínimas.")
    sys.exit(1)

# ====================
# 2. DEFINICIÓN DE EXPERIMENTOS (SOLO BASELINE)
# ====================

EXPERIMENTOS = [
    ("DenseNet121", "BASELINE"),
    ("ResNet50", "BASELINE"),
    ("InceptionV3", "BASELINE"),
    ("EfficientNetB0", "BASELINE")
]

# Hiperparámetros BASELINE (consistentes con Fase Wavelet)
CONFIG_BASELINE = {
    "batch_size": 16,
    "learning_rate": 1e-4,
    "dropout_rate": 0.5,
    "epochs": 10,
    "patience": 4,
    "monitor": "val_recall"
}

# ====================
# 3. FUNCIÓN DE LIMPIEZA DE MEMORIA
# ====================

def limpiar_memoria_completa():
    """
    Realiza una limpieza agresiva de memoria entre experimentos,
    incluyendo sesiones de TensorFlow y recolector de basura.
    """
    gc.collect()

    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.keras.backend.clear_session()

        try:
            from numba import cuda
            cuda.select_device(0)
            cuda.close()
            cuda.select_device(0)
        except:
            pass

    for _ in range(5):
        gc.collect()

    time.sleep(2)

# ====================
# 4. CARGA Y CACHEO DEL DATASET
# ====================

def cargar_datos_percentiles():
    """
    Carga las imágenes desde disco, las normaliza,
    divide el dataset y guarda una versión cacheada.
    """
    cache_file = os.path.join(RESULTS_DIR, "dataset_cache.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    from PIL import Image
    from sklearn.model_selection import train_test_split

    imagenes = []
    etiquetas = []

    for idx_clase, clase in enumerate(CLASES):
        ruta_clase = os.path.join(BASE_PATH, clase)
        archivos = [f for f in os.listdir(ruta_clase)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Límite de imágenes por clase (ajustable según memoria)
        for archivo in archivos[:300]:
            try:
                img_path = os.path.join(ruta_clase, archivo)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img, dtype=np.float32) / 255.0

                imagenes.append(img_array)
                etiquetas.append(idx_clase)
            except:
                continue

    X = np.array(imagenes, dtype=np.float32)
    y = np.array(etiquetas, dtype=np.int32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    datos = (X_train, X_val, X_test, y_train, y_val, y_test)

    with open(cache_file, 'wb') as f:
        pickle.dump(datos, f)

    return datos


X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos_percentiles()

# ====================
# 5. EJECUCIÓN DE EXPERIMENTOS
# ====================

resultados_totales = []

for arch_name, config_name in EXPERIMENTOS:

    inicio = time.time()
    limpiar_memoria_completa()

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef

    if arch_name == "DenseNet121":
        from tensorflow.keras.applications import DenseNet121 as ModeloBase
    elif arch_name == "ResNet50":
        from tensorflow.keras.applications import ResNet50 as ModeloBase
    elif arch_name == "InceptionV3":
        from tensorflow.keras.applications import InceptionV3 as ModeloBase
    elif arch_name == "EfficientNetB0":
        from tensorflow.keras.applications import EfficientNetB0 as ModeloBase

    base_model = ModeloBase(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.Dropout(CONFIG_BASELINE["dropout_rate"]),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(CONFIG_BASELINE["dropout_rate"] * 0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG_BASELINE["learning_rate"]),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Recall(name='recall'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.AUC(name='auc_roc')
        ]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=CONFIG_BASELINE["monitor"],
            patience=CONFIG_BASELINE["patience"],
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    train_gen = keras.preprocessing.image.ImageDataGenerator().flow(
        X_train, y_train,
        batch_size=CONFIG_BASELINE["batch_size"],
        shuffle=True,
        seed=SEED
    )

    val_gen = keras.preprocessing.image.ImageDataGenerator().flow(
        X_val, y_val,
        batch_size=CONFIG_BASELINE["batch_size"],
        shuffle=False
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG_BASELINE["epochs"],
        callbacks=callbacks,
        verbose=1
    )

    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensibilidad = (tp / (tp + fn)) * 100
    especificidad = (tn / (tn + fp)) * 100
    fnr = (fn / (fn + tp)) * 100
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    f1 = (2 * precision * (sensibilidad / 100)) / (precision + (sensibilidad / 100)) if precision > 0 else 0

    resultado = {
        "arquitectura": arch_name,
        "configuracion": config_name,
        "preprocesamiento": "Percentiles",
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "fnr": fnr,
        "auc_roc": auc_roc,
        "mcc": mcc,
        "f1_score": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "epocas": len(history.history["loss"]),
        "tiempo_segundos": time.time() - inicio
    }

    resultados_totales.append(resultado)

    with open(os.path.join(RESULTS_DIR, f"resultado_{arch_name}_{config_name}.json"), "w") as f:
        json.dump(resultado, f, indent=2)

    keras.backend.clear_session()

# ====================
# 6. GUARDADO CONSOLIDADO
# ====================

df_resultados = pd.DataFrame(resultados_totales)
csv_path = os.path.join(RESULTS_DIR, "resultados_percentiles_baseline.csv")
df_resultados.to_csv(csv_path, index=False, encoding="utf-8")
