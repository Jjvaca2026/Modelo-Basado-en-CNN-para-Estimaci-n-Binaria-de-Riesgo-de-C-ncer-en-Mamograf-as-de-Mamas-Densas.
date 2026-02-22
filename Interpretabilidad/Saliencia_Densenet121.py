"""
ANÁLISIS DE MAPAS DE SALIENCIA (SALIENCY MAPS)
Visualización interpretativa de verdaderos positivos (TP),
verdaderos negativos (TN), falsos positivos (FP) y falsos negativos (FN).

Descripción:
Este script utiliza un modelo de clasificación previamente entrenado para:
1. Evaluar imágenes de validación externa (AR / BR)
2. Identificar ejemplos representativos de TP, TN, FP y FN
3. Calcular mapas de saliencia mediante gradientes (Gradient-based Saliency)
4. Generar visualizaciones limpias y comparables para análisis interpretativo
5. Exportar imágenes individuales, un resumen comparativo y métricas asociadas

Salida:
- Carpetas TP, TN, FP, FN con imágenes interpretables
- Imagen resumen comparativa
- Archivo CSV con métricas de saliencia y predicción

Uso recomendado:
Análisis cualitativo de explicabilidad del modelo en contexto de tesis.
"""

# ======================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ======================================================================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("INICIO DEL ANÁLISIS DE MAPAS DE SALIENCIA")
print("=" * 60)

# ======================================================================
# CONFIGURACIÓN GENERAL
# ======================================================================

MODEL_PATH = "/content/drive/MyDrive/Tesis Maestría/entrenamiento/Wavelet/Resultados_FineTuning_Final/mejor_modelo.h5"
DATASET_PATH = "/content/drive/MyDrive/Tesis Maestría/Validacion Externa"

AR_PATH = os.path.join(DATASET_PATH, "AR")  # Alto riesgo
BR_PATH = os.path.join(DATASET_PATH, "BR")  # Bajo riesgo

OUTPUT_BASE = "/content/drive/MyDrive/Tesis Maestría/SALIENCY_MAPS_LIMPIOS"

IMG_SIZE = (224, 224)
THRESHOLD = 0.5
IMAGES_PER_CATEGORY = 3

print(f"Modelo cargado desde: {MODEL_PATH}")

# ======================================================================
# CARGA DEL MODELO
# ======================================================================

print("\nCargando modelo...")
model = keras.models.load_model(MODEL_PATH)
print("Modelo cargado correctamente")

# ======================================================================
# FUNCIONES AUXILIARES
# ======================================================================

def load_image(img_path, target_size=IMG_SIZE):
    """Carga y normaliza una imagen RGB."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img


def predict_image(model, img):
    """Obtiene la probabilidad de clase positiva."""
    img_batch = np.expand_dims(img, axis=0)
    return float(model.predict(img_batch, verbose=0)[0][0])


def compute_saliency_map(img_array, model):
    """
    Calcula el mapa de saliencia usando gradientes absolutos promedio.
    """
    img_tensor = tf.convert_to_tensor(img_array[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor, training=False)
        loss = prediction[:, 0]

    grads = tape.gradient(loss, img_tensor)
    grads = tf.abs(grads)
    saliency = tf.reduce_mean(grads, axis=-1)[0]

    if tf.reduce_max(saliency) > 0:
        saliency /= tf.reduce_max(saliency)

    return saliency.numpy()


# ======================================================================
# BÚSQUEDA DE IMÁGENES POR CATEGORÍA
# ======================================================================

def buscar_imagenes(folder_path, true_label, max_images=IMAGES_PER_CATEGORY):
    """
    Identifica imágenes TP, TN, FP y FN dentro de una carpeta.
    """
    resultados = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    archivos = [f for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for archivo in archivos:
        img_path = os.path.join(folder_path, archivo)
        img = load_image(img_path)

        if img is None:
            continue

        pred = predict_image(model, img)
        pred_label = 1 if pred > THRESHOLD else 0

        if true_label == 1 and pred_label == 1:
            categoria = 'TP'
        elif true_label == 1 and pred_label == 0:
            categoria = 'FN'
        elif true_label == 0 and pred_label == 1:
            categoria = 'FP'
        else:
            categoria = 'TN'

        if len(resultados[categoria]) < max_images:
            saliency = compute_saliency_map(img, model)
            resultados[categoria].append({
                'image': img,
                'filename': archivo,
                'prediction': pred,
                'true_label': true_label,
                'category': categoria,
                'saliency': saliency,
                'confidence': abs(pred - 0.5) * 2
            })

        if all(len(resultados[c]) >= max_images for c in resultados):
            break

    return resultados


print("\nBuscando imágenes representativas...")
resultados_ar = buscar_imagenes(AR_PATH, true_label=1)
resultados_br = buscar_imagenes(BR_PATH, true_label=0)

resultados = {
    'TP': resultados_ar['TP'],
    'FN': resultados_ar['FN'],
    'FP': resultados_br['FP'],
    'TN': resultados_br['TN']
}

# ======================================================================
# CREACIÓN DE DIRECTORIOS DE SALIDA
# ======================================================================

for categoria in resultados:
    os.makedirs(os.path.join(OUTPUT_BASE, categoria), exist_ok=True)

# ======================================================================
# FUNCIÓN DE VISUALIZACIÓN
# ======================================================================

def generar_imagen(resultado, output_path):
    """Genera una visualización limpia de saliencia."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(resultado['image'])
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(resultado['saliency'], cmap='hot')
    axes[1].set_title("Mapa de saliencia")
    axes[1].axis("off")

    heatmap = cv2.applyColorMap(np.uint8(255 * resultado['saliency']), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.uint8(resultado['image'] * 255), 0.6, heatmap, 0.4, 0)

    axes[2].imshow(overlay)
    axes[2].set_title(f"{resultado['category']} | Pred: {resultado['prediction']:.3f}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================================
# GENERACIÓN DE IMÁGENES
# ======================================================================

print("\nGenerando visualizaciones...")
registros = []

for categoria, items in resultados.items():
    for item in items:
        nombre = f"{categoria}_{os.path.splitext(item['filename'])[0]}.png"
        ruta = os.path.join(OUTPUT_BASE, categoria, nombre)
        generar_imagen(item, ruta)

        registros.append({
            'categoria': categoria,
            'archivo': item['filename'],
            'etiqueta_real': 'AR' if item['true_label'] == 1 else 'BR',
            'prediccion': item['prediction'],
            'confianza': item['confidence'],
            'saliency_media': float(np.mean(item['saliency'])),
            'saliency_maxima': float(np.max(item['saliency']))
        })

# ======================================================================
# EXPORTACIÓN DE RESULTADOS
# ======================================================================

df = pd.DataFrame(registros)
csv_path = os.path.join(OUTPUT_BASE, "resultados_saliency.csv")
df.to_csv(csv_path, index=False, encoding='utf-8')

print("\nANÁLISIS COMPLETADO")
print(f"Resultados guardados en: {OUTPUT_BASE}")
print(f"Archivo CSV: {csv_path}")
print("=" * 60)
