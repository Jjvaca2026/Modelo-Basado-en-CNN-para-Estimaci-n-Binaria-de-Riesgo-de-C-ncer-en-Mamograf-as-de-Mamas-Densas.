"""
Autor: Jhon Jaime Vaca Hincapié
Maestría en Ingeniería
Fundación Universitaria Los Libertadores
2025

Script: conversion_dicom_png_polaridad.py
Descripción: Convierte archivos DICOM de mamografías a formato PNG, corrigiendo 
             automáticamente la polaridad (fondo negro/blanco) utilizando los 
             metadatos DICOM (PhotometricInterpretation). Filtra imágenes de 
             alto riesgo (densidad C/D y BI-RADS 4/5) y organiza la salida en 
             estructura de carpetas por categoría.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
BASE_DATASET_DIR = Path(r"E:\DrMammo")
IMAGES_DIR = BASE_DATASET_DIR / "images"
OUTPUT_DIR = Path(r"E:\filtered_mammograms_png_corrected")
MAPEO_CSV = Path(r"E:\filtered_mammograms\mapeo_completo.csv")

print("=" * 80)
print("CONVERSIÓN DICOM → PNG (POLARIDAD DESDE METADATOS DICOM)")
print("=" * 80)

# Verificar que existe el archivo de mapeo
if not MAPEO_CSV.exists():
    print(f"ERROR: No se encontró el archivo de mapeo: {MAPEO_CSV}")
    print("Ejecuta primero el script de descubrimiento para crear el mapeo.")
    exit()

# Crear carpeta de salida si no existe
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
print(f"INFO: Carpeta de salida: {OUTPUT_DIR}")

# ============================================================================
# FUNCIONES DE CONVERSIÓN CON POLARIDAD DICOM
# ============================================================================
def get_dicom_polarity(ds):
    """
    Determina la polaridad de la imagen DICOM.
    
    PhotometricInterpretation puede ser:
    - 'MONOCHROME1': Negro es brillante (fondo blanco, mama oscura) - requiere invertir
    - 'MONOCHROME2': Blanco es brillante (fondo negro, mama clara) - correcto
    
    Returns:
        bool: True si necesita invertir (MONOCHROME1), False si está bien (MONOCHROME2)
    """
    photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    
    print(f"    INFO: PhotometricInterpretation: {photometric}")
    
    if photometric == 'MONOCHROME1':
        # Negro es brillante → fondo blanco, mama oscura → necesita invertir
        return True
    elif photometric == 'MONOCHROME2':
        # Blanco es brillante → fondo negro, mama clara → correcto
        return False
    else:
        # Por defecto, asumir MONOCHROME2
        print(f"    ADVERTENCIA: PhotometricInterpretation desconocido: {photometric}, asumiendo MONOCHROME2")
        return False

def apply_dicom_windowing(img_array, ds):
    """
    Aplica windowing DICOM si está disponible.
    Muchas mamografías usan ventanas específicas para mejor visualización.
    
    Args:
        img_array (numpy.ndarray): Array de la imagen
        ds (pydicom.dataset.FileDataset): Objeto DICOM
        
    Returns:
        numpy.ndarray: Array con windowing aplicado
    """
    # Verificar si hay valores de windowing
    has_windowing = (
        hasattr(ds, 'WindowCenter') and 
        hasattr(ds, 'WindowWidth') and
        ds.WindowCenter is not None and
        ds.WindowWidth is not None
    )
    
    if has_windowing:
        # Puede ser un solo valor o una lista
        if isinstance(ds.WindowCenter, pydicom.multival.MultiValue):
            window_center = float(ds.WindowCenter[0])
            window_width = float(ds.WindowWidth[0])
        else:
            window_center = float(ds.WindowCenter)
            window_width = float(ds.WindowWidth)
        
        print(f"    INFO: Windowing DICOM: Center={window_center}, Width={window_width}")
        
        # Aplicar windowing lineal
        img_min = window_center - window_width/2
        img_max = window_center + window_width/2
        
        # Clip y normalizar
        img_array = np.clip(img_array, img_min, img_max)
        img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
    return img_array

def convert_dicom_to_png_pro(dicom_path, output_path):
    """
    Conversión profesional DICOM → PNG respetando metadatos DICOM.
    
    Args:
        dicom_path (Path): Ruta al archivo DICOM
        output_path (Path): Ruta de salida para el PNG
        
    Returns:
        bool: True si la conversión fue exitosa, False en caso contrario
    """
    try:
        # 1. Leer archivo DICOM completo
        ds = pydicom.dcmread(dicom_path, force=True)
        
        # Verificar que tiene pixel array
        if not hasattr(ds, 'pixel_array'):
            print(f"    ERROR: No tiene pixel_array: {dicom_path.name}")
            return False
        
        # 2. Obtener array de píxeles en su formato original
        img_array = ds.pixel_array
        
        # 3. Mostrar información DICOM útil
        print(f"    INFO: DICOM Info:")
        print(f"      - Bits Stored: {getattr(ds, 'BitsStored', 'N/A')}")
        print(f"      - Bits Allocated: {getattr(ds, 'BitsAllocated', 'N/A')}")
        print(f"      - High Bit: {getattr(ds, 'HighBit', 'N/A')}")
        print(f"      - Pixel Representation: {getattr(ds, 'PixelRepresentation', 'N/A')}")
        
        # 4. Manejar diferentes profundidades de bits
        bits_stored = getattr(ds, 'BitsStored', 16)
        pixel_representation = getattr(ds, 'PixelRepresentation', 0)  # 0=unsigned, 1=signed
        
        if bits_stored == 16:
            if pixel_representation == 0:  # Unsigned 16-bit
                # Rango típico de mamografía: 0-65535
                img_array = img_array.astype(np.float32)
                
                # Usar Rescale Slope/Intercept si están disponibles
                rescale_slope = getattr(ds, 'RescaleSlope', 1.0)
                rescale_intercept = getattr(ds, 'RescaleIntercept', 0.0)
                
                if rescale_slope != 1.0 or rescale_intercept != 0.0:
                    print(f"    INFO: Rescale: Slope={rescale_slope}, Intercept={rescale_intercept}")
                    img_array = img_array * rescale_slope + rescale_intercept
                
                # Aplicar windowing DICOM si está disponible
                img_array = apply_dicom_windowing(img_array, ds)
                
                # Si aún no se aplicó windowing, usar rango típico mamográfico
                if img_array.dtype != np.uint8:
                    # Rango típico para mamografías (ajustable)
                    min_val = np.percentile(img_array, 1)
                    max_val = np.percentile(img_array, 99)
                    
                    if max_val > min_val:
                        img_array = np.clip(img_array, min_val, max_val)
                        img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        # Fallback a normalización simple
                        img_array = ((img_array - img_array.min()) / 
                                    (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            
            elif pixel_representation == 1:  # Signed 16-bit
                # Convertir a unsigned
                img_array = img_array.astype(np.float32)
                img_array = ((img_array - img_array.min()) / 
                            (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        
        elif bits_stored == 12 or bits_stored == 14:
            # 12-bit o 14-bit mammography (común)
            img_array = img_array.astype(np.float32)
            
            # Normalizar a 0-255 basado en bits stored
            max_val = 2**bits_stored - 1
            img_array = (img_array / max_val * 255).astype(np.uint8)
        
        elif img_array.dtype != np.uint8:
            # Normalización genérica como fallback
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        # 5. CORREGIR POLARIDAD basada en metadatos DICOM
        need_invert = get_dicom_polarity(ds)
        if need_invert:
            # Invertir: MONOCHROME1 → MONOCHROME2
            img_array = 255 - img_array
            print(f"    INFO: Polaridad corregida (MONOCHROME1 → MONOCHROME2)")
        
        # 6. Opcional: Ajuste de contraste para mamografías
        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Solo si la imagen tiene bajo contraste
        contrast = np.std(img_array)
        if contrast < 40:  # Bajo contraste
            print(f"    INFO: Aplicando CLAHE (contraste bajo: {contrast:.1f})")
            
            # Usar CLAHE de OpenCV si está disponible, o implementación simple
            try:
                import cv2
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_array = clahe.apply(img_array)
            except:
                # Implementación simple de ecualización
                hist, bins = np.histogram(img_array.flatten(), 256, [0,256])
                cdf = hist.cumsum()
                cdf_normalized = cdf * 255 / cdf[-1]
                img_array = np.interp(img_array.flatten(), bins[:-1], cdf_normalized)
                img_array = img_array.reshape(img_array.shape).astype(np.uint8)
        
        # 7. Convertir a PIL Image
        img = Image.fromarray(img_array, mode='L')
        
        # 8. Redimensionar a 224x224 manteniendo relación de aspecto
        target_size = (224, 224)
        
        # Calcular nuevo tamaño manteniendo relación
        ratio = min(target_size[0] / img.width, target_size[1] / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        
        # Usar LANCZOS para mejor calidad en reducción
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 9. Crear imagen 224x224 con fondo negro
        img_final = Image.new("L", target_size, 0)  # Fondo NEGRO
        
        # Centrar la imagen redimensionada
        offset = ((target_size[0] - new_size[0]) // 2, 
                  (target_size[1] - new_size[1]) // 2)
        img_final.paste(img_resized, offset)
        
        # 10. Guardar como PNG con compresión
        img_final.save(output_path, "PNG", optimize=True, compress_level=6)
        
        # 11. Verificar resultado
        img_verification = Image.open(output_path)
        img_array_final = np.array(img_verification)
        
        # Calcular estadísticas de la imagen final
        mean_val = np.mean(img_array_final)
        std_val = np.std(img_array_final)
        black_pixels = np.sum(img_array_final < 20) / img_array_final.size * 100
        
        print(f"    INFO: PNG creado: media={mean_val:.1f}, desv={std_val:.1f}, {black_pixels:.1f}% pixeles negros")
        
        return True
        
    except Exception as e:
        print(f"    ERROR: Error en {dicom_path.name}: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return False

def crear_estructura_organizada(base_dir, birads, density, laterality):
    """
    Crea estructura de carpetas organizada por BI-RADS, densidad y lateralidad.
    
    Args:
        base_dir (Path): Directorio base de salida
        birads (str): Categoría BI-RADS
        density (str): Densidad mamaria
        laterality (str): Lateralidad (L o R)
        
    Returns:
        Path: Ruta completa del directorio creado
    """
    birads_clean = birads.replace(" ", "_").replace("-", "_")
    density_clean = density.replace(" ", "_")
    laterality_clean = laterality
    
    output_path = base_dir / birads_clean / density_clean / laterality_clean
    output_path.mkdir(exist_ok=True, parents=True)
    
    return output_path

# ============================================================================
# CARGAR Y PROCESAR DATOS
# ============================================================================
print("\nINFO: Cargando mapeo...")
df_mapeo = pd.read_csv(MAPEO_CSV)
breast_df = pd.read_csv(BASE_DATASET_DIR / 'breast-level_annotations.csv')
breast_df['birads_numeric'] = breast_df['breast_birads'].str.extract(r'(\d+)').astype(int)

# Combinar datos
df_completo = pd.merge(
    df_mapeo,
    breast_df[['study_id', 'image_id', 'breast_birads', 'breast_density', 'laterality', 'view_position']],
    left_on=['study_id', 'image_id'],
    right_on=['study_id', 'image_id'],
    how='inner'
)

# Filtrar alto riesgo (densidad C/D y BI-RADS 4/5)
df_filtrado = df_completo[
    (df_completo['breast_density'].isin(['DENSITY C', 'DENSITY D'])) & 
    (df_completo['breast_birads'].isin(['BI-RADS 4', 'BI-RADS 5']))
].copy()

print(f"INFO: Imágenes a procesar: {len(df_filtrado)}")

# ============================================================================
# CONVERSIÓN PRINCIPAL
# ============================================================================
print("\n" + "=" * 80)
print("INICIANDO CONVERSIÓN CON POLARIDAD DICOM")
print("=" * 80)

estadisticas = {
    'total': len(df_filtrado),
    'procesadas': 0,
    'exitosas': 0,
    'fallidas': [],
    'sin_archivo': []
}

# Procesar en lotes para mejor control
batch_size = 50
num_batches = (len(df_filtrado) + batch_size - 1) // batch_size

for batch_num in range(num_batches):
    start_idx = batch_num * batch_size
    end_idx = min((batch_num + 1) * batch_size, len(df_filtrado))
    batch = df_filtrado.iloc[start_idx:end_idx]
    
    print(f"\nINFO: LOTE {batch_num+1}/{num_batches} ({len(batch)} imágenes)")
    print("-" * 40)
    
    for idx, row in batch.iterrows():
        # Extraer información de la fila
        image_id = row['image_id']
        study_id = row['study_id']
        birads = row['breast_birads']
        density = row['breast_density']
        laterality = row['laterality']
        view_position = row['view_position']
        archivo_fisico = row['archivo']
        carpeta_fisica = row['carpeta']
        
        ruta_dicom = IMAGES_DIR / carpeta_fisica / archivo_fisico
        
        estadisticas['procesadas'] += 1
        item_num = estadisticas['procesadas']
        
        print(f"\n[{item_num}/{estadisticas['total']}] {study_id[:8]}..._{laterality}_{view_position}")
        
        # Verificar que el archivo existe
        if not ruta_dicom.exists():
            print(f"  ERROR: No existe: {ruta_dicom}")
            estadisticas['sin_archivo'].append(str(ruta_dicom))
            continue
        
        # Crear estructura de carpetas y convertir
        output_subdir = crear_estructura_organizada(OUTPUT_DIR, birads, density, laterality)
        png_filename = f"{study_id[:8]}_{laterality}_{view_position}_{image_id[:8]}.png"
        output_path = output_subdir / png_filename
        
        # Convertir usando función profesional
        if convert_dicom_to_png_pro(ruta_dicom, output_path):
            estadisticas['exitosas'] += 1
        else:
            estadisticas['fallidas'].append(image_id)
    
    # Guardar checkpoint después de cada lote
    checkpoint_file = OUTPUT_DIR / f"checkpoint_batch_{batch_num+1}.txt"
    with open(checkpoint_file, 'w') as f:
        f.write(f"Batch {batch_num+1}/{num_batches} completado\n")
        f.write(f"Exitosas: {estadisticas['exitosas']}/{estadisticas['procesadas']}\n")

# ============================================================================
# REPORTE Y METADATA
# ============================================================================
print("\n" + "=" * 80)
print("REPORTE FINAL")
print("=" * 80)

print(f"\nESTADÍSTICAS:")
print(f"  Total: {estadisticas['total']}")
print(f"  Procesadas: {estadisticas['procesadas']}")
print(f"  Exitosas: {estadisticas['exitosas']} ({estadisticas['exitosas']/estadisticas['total']*100:.1f}%)")
print(f"  Fallidas: {len(estadisticas['fallidas'])}")
print(f"  Archivos no encontrados: {len(estadisticas['sin_archivo'])}")

# Guardar metadata final con rutas de los PNGs
df_filtrado['ruta_png'] = ""
for idx, row in df_filtrado.iterrows():
    birads = row['breast_birads']
    density = row['breast_density']
    laterality = row['laterality']
    study_id = row['study_id']
    view_position = row['view_position']
    image_id = row['image_id']
    
    birads_clean = birads.replace(" ", "_").replace("-", "_")
    density_clean = density.replace(" ", "_")
    png_filename = f"{study_id[:8]}_{laterality}_{view_position}_{image_id[:8]}.png"
    
    df_filtrado.at[idx, 'ruta_png'] = f"{birads_clean}/{density_clean}/{laterality}/{png_filename}"

metadata_path = OUTPUT_DIR / "metadata_completo.csv"
df_filtrado.to_csv(metadata_path, index=False)
print(f"\nINFO: Metadata guardada: {metadata_path}")

# Mostrar estructura de carpetas resultante
print(f"\nESTRUCTURA FINAL:")
total_pngs = 0
for birads_dir in sorted(OUTPUT_DIR.iterdir()):
    if birads_dir.is_dir() and not birads_dir.name.startswith('checkpoint'):
        print(f"  {birads_dir.name}/")
        for density_dir in sorted(birads_dir.iterdir()):
            if density_dir.is_dir():
                print(f"    {density_dir.name}/")
                for lado_dir in sorted(density_dir.iterdir()):
                    if lado_dir.is_dir():
                        count = len(list(lado_dir.glob("*.png")))
                        total_pngs += count
                        print(f"      {lado_dir.name}/ ({count} imágenes)")

print(f"\nRESUMEN: TOTAL PNGs CREADOS: {total_pngs}")
print(f"UBICACIÓN: {OUTPUT_DIR}")
