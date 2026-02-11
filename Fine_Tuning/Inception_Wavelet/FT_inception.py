"""
ENTRENAMIENTO COMPLETO INCEPTIONV3 CON IMÁGENES WAVELET

Descripción:
Este script implementa el entrenamiento completo de un modelo InceptionV3
para clasificación binaria (Alto Riesgo / Bajo Riesgo) utilizando imágenes
preprocesadas mediante Wavelet.

El flujo metodológico incluye:
1. Preparación del dataset y división estratificada (80/10/10).
2. Entrenamiento inicial mediante transferencia de aprendizaje (feature extraction).
3. Ajuste fino (fine-tuning) del modelo utilizando pesos de clase corregidos.
4. Evaluación con métricas estándar de clasificación.
5. Almacenamiento reproducible de modelos y resultados.

Las rutas de acceso a imágenes y almacenamiento de resultados deben ser
ajustadas según la estructura local del proyecto del usuario.
"""

# ======================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ======================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 2. CONFIGURACIÓN GENERAL
# ======================================================
# RUTAS (AJUSTAR SEGÚN EL PROYECTO)
BASE_PATH = "/content/drive/MyDrive/Tesis Maestría/entrenamiento/Wavelet"
AR_PATH = os.path.join(BASE_PATH, "AR")  # Imágenes de Alto Riesgo
BR_PATH = os.path.join(BASE_PATH, "BR")  # Imágenes de Bajo Riesgo

RESULTS_PATH = os.path.join(BASE_PATH, "Resultados_InceptionV3_Wavelet")
MODEL_PATH = os.path.join(RESULTS_PATH, "modelos")
os.makedirs(MODEL_PATH, exist_ok=True)

# PARÁMETROS EXPERIMENTALES
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
SEED = 42

LR_BASE = 1e-4
LR_FINETUNE = 1e-5

EPOCHS_BASE = 6
EPOCHS_FINETUNE = 12
PATIENCE = 4

CLASS_WEIGHTS = {0: 1.0, 1: 2.5}

# ======================================================
# 3. GENERADOR DE DATOS
# ======================================================
class WaveletDataGenerator(keras.utils.Sequence):
    """
    Generador de datos para imágenes Wavelet compatible con InceptionV3.
    """

    def __init__(self, image_paths, labels, batch_size, img_size,
                 augment=False, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle

        self.preprocess = keras.applications.inception_v3.preprocess_input

        if augment:
            self.datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.05,
                height_shift_range=0.05,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode="reflect"
            )
        else:
            self.datagen = None

        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []

        for i in batch_idx:
            img = keras.preprocessing.image.load_img(
                self.image_paths[i], target_size=self.img_size
            )
            img = keras.preprocessing.image.img_to_array(img)

            if self.augment and self.datagen:
                img = self.datagen.random_transform(img)

            img = self.preprocess(img)
            images.append(img)
            labels.append(self.labels[i])

        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ======================================================
# 4. PREPARACIÓN DEL DATASET
# ======================================================
def preparar_datos_wavelet():
    """
    Carga imágenes Wavelet y genera una división estratificada 80/10/10.
    """

    def cargar_imagenes(path):
        return [
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    ar_imgs = cargar_imagenes(AR_PATH)
    br_imgs = cargar_imagenes(BR_PATH)

    all_paths = ar_imgs + br_imgs
    all_labels = [1] * len(ar_imgs) + [0] * len(br_imgs)

    idx = np.arange(len(all_paths))

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, all_labels, test_size=0.2,
        stratify=all_labels, random_state=SEED
    )

    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.5,
        stratify=y_temp, random_state=SEED
    )

    train_gen = WaveletDataGenerator(
        [all_paths[i] for i in train_idx], y_train,
        BATCH_SIZE, IMG_SIZE, augment=True
    )

    val_gen = WaveletDataGenerator(
        [all_paths[i] for i in val_idx], y_val,
        BATCH_SIZE, IMG_SIZE, augment=False, shuffle=False
    )

    test_gen = WaveletDataGenerator(
        [all_paths[i] for i in test_idx], y_test,
        BATCH_SIZE, IMG_SIZE, augment=False, shuffle=False
    )

    return train_gen, val_gen, test_gen, y_test

# ======================================================
# 5. CONSTRUCCIÓN DEL MODELO
# ======================================================
def construir_modelo():
    """
    Construye el modelo InceptionV3 con capas superiores personalizadas.
    """

    base_model = keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(299, 299, 3),
        pooling="avg"
    )
    base_model.trainable = False

    x = base_model.output
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(LR_BASE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.AUC(name="auc_roc")
        ]
    )

    return model, base_model

# ======================================================
# 6. ENTRENAMIENTO COMPLETO
# ======================================================
def entrenar_modelo():
    train_gen, val_gen, test_gen, y_test = preparar_datos_wavelet()
    model, base_model = construir_modelo()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_recall",
            patience=PATIENCE,
            restore_best_weights=True,
            mode="max"
        )
    ]

    # ETAPA 1: TRANSFERENCIA DE APRENDIZAJE
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_BASE,
        class_weight=CLASS_WEIGHTS,
        callbacks=callbacks
    )

    # ETAPA 2: AJUSTE FINO
    base_model.trainable = True
    for layer in base_model.layers[:120]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(LR_FINETUNE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.AUC(name="auc_roc")
        ]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_BASE + EPOCHS_FINETUNE,
        initial_epoch=EPOCHS_BASE,
        class_weight=CLASS_WEIGHTS,
        callbacks=callbacks
    )

    return model, test_gen, y_test

# ======================================================
# 7. EVALUACIÓN
# ======================================================
def evaluar_modelo(model, test_gen, y_test):
    y_proba = model.predict(test_gen).flatten()
    y_pred = (y_proba > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    resultados = {
        "preprocesamiento": "Wavelet",
        "sensibilidad": tp / (tp + fn),
        "especificidad": tn / (tn + fp),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(RESULTS_PATH, "resultados_finales_wavelet.json"), "w") as f:
        json.dump(resultados, f, indent=2)

    model.save(os.path.join(MODEL_PATH, "modelo_final_inceptionv3_wavelet.h5"))

    return resultados

# ======================================================
# 8. EJECUCIÓN
# ======================================================
if __name__ == "__main__":
    modelo, test_gen, y_test = entrenar_modelo()
    resultados = evaluar_modelo(modelo, test_gen, y_test)
    print("Entrenamiento y evaluación completados.")
