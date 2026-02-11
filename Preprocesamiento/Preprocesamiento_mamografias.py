# -*- coding: utf-8 -*-
"""
Pipeline completo de preprocesamiento para tesis de maestría

Este script implementa el pipeline completo de preprocesamiento utilizado en la tesis
para el análisis de mamografías. Incluye la limpieza de imágenes, la aplicación de
cuatro técnicas de preprocesamiento (Ecualización de Histograma, CLAHE, ajuste por
percentiles y realce mediante transformada wavelet), el cálculo de métricas objetivas
de calidad de imagen (PSNR, SSIM y entropía), así como la generación de resultados
cuantitativos utilizados posteriormente en el análisis de clustering.

El script está diseñado para ejecutarse en Google Colab y asume que las imágenes
crudas y los resultados se almacenan en rutas definidas por el usuario.

IMPORTANTE:
Las rutas definidas en la Sección 4 deben ser modificadas según el entorno local
o en la nube (por ejemplo, Google Drive).
"""

# ============================================================
# 1. INSTALACIÓN DE LIBRERÍAS (GOOGLE COLAB)
# ============================================================
!pip install pywavelets scikit-image opencv-python-headless tqdm > /dev/null 2>&1

# ============================================================
# 2. IMPORTACIÓN DE LIBRERÍAS
# ============================================================
import os
import cv2
import numpy as np
import pandas as pd
from skimage import exposure
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import pywt
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from google.colab import drive

warnings.filterwarnings('ignore')

# ============================================================
# 3. MONTAJE DE GOOGLE DRIVE
# ============================================================
drive.mount('/content/drive', force_remount=True)

# ============================================================
# 4. DEFINICIÓN DE RUTAS DE ENTRADA Y SALIDA
# ============================================================
"""
BASE_PATH debe apuntar al directorio raíz del proyecto.

Estructura esperada dentro de BASE_PATH:

RAW/
 ├── Alto_Riesgo/      -> Imágenes crudas de mamografías (alto riesgo)
 └── Bajo_Riesgo/      -> Imágenes crudas de mamografías (bajo riesgo)

PP/
 ├── HE/
 ├── Clahe/
 ├── Percentiles/
 └── Wavelet/

Archivos generados por el script:
- metricas_preprocesamiento.csv
- analisis_estadistico.csv
- metricas_completas.pkl
- Figuras_Tesis/
"""

BASE_PATH = '/content/drive/My Drive/Tesis Maestría'  # <-- MODIFICAR SEGÚN EL ENTORNO

# Rutas de entrada (imágenes crudas)
RAW_ALTO = os.path.join(BASE_PATH, 'RAW/Alto_Riesgo')
RAW_BAJO = os.path.join(BASE_PATH, 'RAW/Bajo_Riesgo')

# Rutas de salida para cada técnica de preprocesamiento
TECNICAS = {
    'HE': 'PP/HE',
    'Clahe': 'PP/Clahe',
    'Percentiles': 'PP/Percentiles',
    'Wavelet': 'PP/Wavelet'
}

# Creación de carpetas de salida
for tecnica, ruta_base in TECNICAS.items():
    for clase in ['Alto_Riesgo', 'Bajo_Riesgo']:
        os.makedirs(os.path.join(BASE_PATH, ruta_base, clase), exist_ok=True)

# Carpeta para figuras de la tesis
FIGURAS_PATH = os.path.join(BASE_PATH, 'Figuras_Tesis')
os.makedirs(FIGURAS_PATH, exist_ok=True)

# ============================================================
# 5. FUNCIONES DE LIMPIEZA DE IMÁGENES
# ============================================================
def asegurar_fondo_negro(img):
    """
    Asegura que el fondo de la imagen sea negro y la región de la mama tenga
    intensidades altas. Si la imagen está invertida, se corrige automáticamente.
    """
    if img is None or img.size == 0:
        return img

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    if np.sum(hist[200:]) > np.sum(hist[:50]):
        img = 255 - img

    return img


def eliminar_anotaciones(img, umbral_fondo=20):
    """
    Elimina anotaciones médicas y artefactos del fondo mediante la detección
    del contorno principal correspondiente a la mama.
    """
    if img is None:
        return img

    _, mask_fondo = cv2.threshold(img, umbral_fondo, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_fondo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    largest_contour = max(contours, key=cv2.contourArea)

    mask_mama = np.zeros_like(img)
    cv2.drawContours(mask_mama, [largest_contour], -1, 255, -1)

    kernel = np.ones((5, 5), np.uint8)
    mask_mama = cv2.morphologyEx(mask_mama, cv2.MORPH_CLOSE, kernel)
    mask_mama = cv2.morphologyEx(mask_mama, cv2.MORPH_OPEN, kernel)

    img_limpia = cv2.bitwise_and(img, img, mask=mask_mama)
    img_limpia[mask_mama == 0] = 0

    return img_limpia


def limpiar_imagen(img):
    """
    Pipeline completo de limpieza de la imagen:
    - Corrección de orientación de intensidades
    - Eliminación de anotaciones y fondo
    """
    if img is None:
        return None

    img = asegurar_fondo_negro(img)
    img = eliminar_anotaciones(img)
    return img

# ============================================================
# 6. TÉCNICAS DE PREPROCESAMIENTO
# ============================================================
def preprocesar_HE(img):
    """Ecualización global del histograma."""
    img_clean = limpiar_imagen(img)
    if img_clean is None:
        return None

    img_eq = exposure.equalize_hist(img_clean) * 255
    return img_eq.astype(np.uint8)


def preprocesar_CLAHE(img, clip_limit=2.0, grid_size=(8, 8)):
    """Ecualización adaptativa limitada por contraste (CLAHE)."""
    img_clean = limpiar_imagen(img)
    if img_clean is None:
        return None

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img_clean)


def preprocesar_Percentiles(img, low_perc=1, high_perc=99):
    """Ajuste de contraste basado en percentiles."""
    img_clean = limpiar_imagen(img)
    if img_clean is None:
        return None

    non_zero_pixels = img_clean[img_clean > 10]
    if len(non_zero_pixels) == 0:
        return img_clean

    low_val = np.percentile(non_zero_pixels, low_perc)
    high_val = np.percentile(non_zero_pixels, high_perc)

    img_rescaled = exposure.rescale_intensity(
        img_clean,
        in_range=(low_val, high_val),
        out_range=(0, 255)
    )

    return img_rescaled.astype(np.uint8)


def preprocesar_Wavelet(img, wavelet='db1', level=2):
    """Realce de detalles mediante transformada wavelet."""
    img_clean = limpiar_imagen(img)
    if img_clean is None:
        return None

    try:
        rows, cols = img_clean.shape
        max_level = pywt.dwt_max_level(min(rows, cols), pywt.Wavelet(wavelet).dec_len)
        level = min(level, max_level)

        if level < 1:
            return img_clean

        coeffs = pywt.wavedec2(img_clean, wavelet, level=level)
        coeffs_list = list(coeffs)

        for i in range(1, len(coeffs_list)):
            if isinstance(coeffs_list[i], tuple) and len(coeffs_list[i]) == 3:
                LH, HL, HH = coeffs_list[i]
                coeffs_list[i] = (LH * 1.3, HL * 1.3, HH * 1.3)

        img_wavelet = pywt.waverec2(coeffs_list, wavelet)
        img_wavelet = np.clip(img_wavelet, 0, 255).astype(np.uint8)

        if img_wavelet.shape != img_clean.shape:
            img_wavelet = cv2.resize(img_wavelet, (cols, rows))

        return img_wavelet

    except Exception:
        return preprocesar_CLAHE(img_clean)

# ============================================================
# 7. MÉTRICAS DE CALIDAD DE IMAGEN
# ============================================================
def calcular_metricas(img_original, img_procesada):
    """
    Calcula PSNR, SSIM y entropía de Shannon entre la imagen original
    y la imagen preprocesada.
    """
    if img_original is None or img_procesada is None:
        return 0.0, 0.0, 0.0

    if img_original.shape != img_procesada.shape:
        img_procesada = cv2.resize(
            img_procesada,
            (img_original.shape[1], img_original.shape[0])
        )

    try:
        return (
            psnr(img_original, img_procesada, data_range=255),
            ssim(img_original, img_procesada, data_range=255),
            shannon_entropy(img_procesada)
        )
    except Exception:
        return 0.0, 0.0, 0.0

# ============================================================
# 8. PROCESAMIENTO DE UNA IMAGEN
# ============================================================
def procesar_imagen(ruta_original, clase):
    """
    Aplica todas las técnicas de preprocesamiento a una imagen individual
    y calcula las métricas correspondientes.
    """
    nombre_archivo = os.path.basename(ruta_original)
    img_original = cv2.imread(ruta_original, cv2.IMREAD_GRAYSCALE)

    if img_original is None:
        return None

    img_clean = limpiar_imagen(img_original.copy())

    funciones = {
        'HE': preprocesar_HE,
        'Clahe': preprocesar_CLAHE,
        'Percentiles': preprocesar_Percentiles,
        'Wavelet': preprocesar_Wavelet
    }

    resultados = {}

    for nombre_tecnica, funcion in funciones.items():
        img_proc = funcion(img_original.copy())
        if img_proc is None:
            continue

        ruta_salida = os.path.join(
            BASE_PATH, TECNICAS[nombre_tecnica], clase, nombre_archivo
        )
        cv2.imwrite(ruta_salida, img_proc)

        p, s, e = calcular_metricas(img_clean, img_proc)
        resultados[nombre_tecnica] = {
            'psnr': p,
            'ssim': s,
            'entropia': e
        }

    return {
        'imagen': nombre_archivo,
        'clase': clase,
        'resultados': resultados
    }

# ============================================================
# 9. PROCESAMIENTO EN LOTE
# ============================================================
def procesar_lote(limite=None):
    """
    Procesa todas las imágenes disponibles y almacena las métricas
    en un archivo CSV.
    """
    todas_metricas = []

    for clase, ruta_raw in [('Alto_Riesgo', RAW_ALTO), ('Bajo_Riesgo', RAW_BAJO)]:
        if not os.path.exists(ruta_raw):
            continue

        imagenes = [
            f for f in os.listdir(ruta_raw)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if limite:
            imagenes = imagenes[:limite]

        for nombre_img in tqdm(imagenes, desc=f'Procesando {clase}'):
            resultado = procesar_imagen(os.path.join(ruta_raw, nombre_img), clase)
            if resultado:
                for tecnica, m in resultado['resultados'].items():
                    todas_metricas.append({
                        'imagen': resultado['imagen'],
                        'clase': resultado['clase'],
                        'tecnica': tecnica,
                        'psnr': m['psnr'],
                        'ssim': m['ssim'],
                        'entropia': m['entropia']
                    })

    if not todas_metricas:
        return None

    df = pd.DataFrame(todas_metricas)
    df.to_csv(os.path.join(BASE_PATH, 'metricas_preprocesamiento.csv'), index=False)

    return df

# ============================================================
# 10. ANÁLISIS ESTADÍSTICO
# ============================================================
def analizar_metricas(df):
    """
    Calcula estadísticas descriptivas por técnica de preprocesamiento.
    """
    stats = df.groupby('tecnica').agg({
        'psnr': ['mean', 'std', 'min', 'max'],
        'ssim': ['mean', 'std', 'min', 'max'],
        'entropia': ['mean', 'std', 'min', 'max']
    }).round(3)

    stats.to_csv(os.path.join(BASE_PATH, 'analisis_estadistico.csv'))

# ============================================================
# 11. EJECUCIÓN PRINCIPAL
# ============================================================
if __name__ == '__main__':
    df_resultados = procesar_lote(limite=None)
    if df_resultados is not None:
        analizar_metricas(df_resultados)
        df_resultados.to_pickle(
            os.path.join(BASE_PATH, 'metricas_completas.pkl')
        )
