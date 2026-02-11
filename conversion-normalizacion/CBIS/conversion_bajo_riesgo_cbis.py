"""
Autor: Jhon Jaime Vaca Hincapié
Maestría en Ingeniería
Fundación Universitaria Los Libertadores
2025

Script: conversion_bajo_riesgo_cbis.py
Descripción: Procesa el dataset CBIS-DDSM para extraer imágenes mamográficas
             correspondientes a casos de bajo riesgo (BI-RADS 1-3) con densidad
             mamaria tipo C o D. Implementa muestreo aleatorio para obtener un
             conjunto balanceado con respecto a la clase de alto riesgo y realiza
             conversión de formato DICOM a PNG con resolución estandarizada.
"""

import pandas as pd
import pydicom
import os
from PIL import Image
import numpy as np
from pathlib import Path
import warnings
import re
import random

# Suprimir advertencias no críticas para una salida más limpia
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# CONFIGURACIÓN DE PARÁMETROS Y DIRECTORIOS
# ------------------------------------------------------------
base_path = Path(".")
output_dir = base_path / "mamas_bajo_riesgo"
output_dir.mkdir(exist_ok=True)

# Número objetivo de imágenes para balancear con la clase de alto riesgo
TARGET_COUNT = 708

# ------------------------------------------------------------
# CARGA Y CONSOLIDACIÓN DE METADATOS
# ------------------------------------------------------------
csv_files = [
    "calc_case_description_test_set.csv",
    "calc_case_description_train_set.csv", 
    "mass_case_description_test_set.csv",
    "mass_case_description_train_set.csv"
]

dfs = []
for csv_file in csv_files:
    csv_path = base_path / csv_file
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Normalización de nombres de columnas para consistencia
        df.columns = df.columns.str.strip()
        dfs.append(df)
        print(f"Cargado: {csv_file} ({len(df)} registros)")

df_combined = pd.concat(dfs, ignore_index=True)
print(f"Total de registros consolidados: {len(df_combined)}")

# ------------------------------------------------------------
# FILTRADO POR CRITERIOS CLÍNICOS - BAJO RIESGO
# ------------------------------------------------------------
# Identificación automática de nombres de columnas relevantes
assessment_col = [c for c in df_combined.columns if 'assessment' in c.lower()][0]
density_col = [c for c in df_combined.columns if 'breast' in c.lower() and 'density' in c.lower()][0]

print(f"Columna utilizada para BI-RADS: {assessment_col}")
print(f"Columna utilizada para densidad mamaria: {density_col}")

# Filtrado de casos de bajo riesgo (BI-RADS 1-3) con densidad mamaria C/D
df_bajo_riesgo = df_combined[
    (df_combined[assessment_col].isin([1, 2, 3])) & 
    (df_combined[density_col].isin([3, 4]))
].copy()

print(f"Total de casos de bajo riesgo (BI-RADS 1-3, Densidad 3-4): {len(df_bajo_riesgo)}")

# Ajuste del objetivo si no hay suficientes casos disponibles
if len(df_bajo_riesgo) < TARGET_COUNT:
    print(f"Advertencia: Solo se encontraron {len(df_bajo_riesgo)} imágenes de bajo riesgo")
    print(f"Se ajustará el objetivo a: {len(df_bajo_riesgo)} imágenes")
    TARGET_COUNT = len(df_bajo_riesgo)

# ------------------------------------------------------------
# MUESTREO ALEATORIO PARA BALANCE DE CLASES
# ------------------------------------------------------------
# Mezcla aleatoria de los registros con semilla para reproducibilidad
df_bajo_riesgo = df_bajo_riesgo.sample(frac=1, random_state=42).reset_index(drop=True)
df_filtered = df_bajo_riesgo.head(TARGET_COUNT).copy()

print(f"Muestra seleccionada aleatoriamente: {len(df_filtered)} imágenes")

# Análisis de distribución en la muestra seleccionada
print("\nDistribución en la muestra seleccionada:")
print("="*40)
birads_counts = df_filtered[assessment_col].value_counts().sort_index()
for birads, count in birads_counts.items():
    print(f"  BI-RADS {birads}: {count} imágenes ({count/TARGET_COUNT*100:.1f}%)")

density_counts = df_filtered[density_col].value_counts().sort_index()
density_map = {3: 'C', 4: 'D'}
print("\nDistribución por densidad mamaria:")
for density, count in density_counts.items():
    density_name = density_map.get(density, f"Desconocida ({density})")
    print(f"  Densidad {density_name}: {count} imágenes ({count/TARGET_COUNT*100:.1f}%)")

# ------------------------------------------------------------
# BÚSQUEDA E INDEXACIÓN DE ARCHIVOS DICOM
# ------------------------------------------------------------
print("\nBuscando archivos DICOM en la estructura de directorios...")
dicom_files = list(base_path.rglob("*.dcm"))
print(f"Archivos DICOM localizados: {len(dicom_files)}")

if not dicom_files:
    print("No se encontraron archivos DICOM en el directorio especificado.")
    exit()

# Creación de diccionario de búsqueda con múltiples estrategias de indexación
dicom_search_dict = {}
for dcm_path in dicom_files:
    path_str = str(dcm_path)
    
    # Indexación por identificador de paciente
    patient_match = re.search(r'(P_\d+)', path_str, re.IGNORECASE)
    if patient_match:
        patient_id = patient_match.group(1)
        dicom_search_dict.setdefault(patient_id, []).append(dcm_path)
    
    # Indexación por número de caso
    case_match = re.search(r'(?:_)(\d{5})(?:_|\\|/)', path_str)
    if case_match:
        case_num = case_match.group(1)
        dicom_search_dict.setdefault(case_num, []).append(dcm_path)
    
    # Indexación por vista mamográfica
    view_match = re.search(r'(LEFT|RIGHT)_(CC|MLO)', path_str, re.IGNORECASE)
    if view_match:
        view_key = f"{view_match.group(1)}_{view_match.group(2)}"
        dicom_search_dict.setdefault(view_key, []).append(dcm_path)

print(f"Diccionario de búsqueda creado con {len(dicom_search_dict)} claves de indexación")

# ------------------------------------------------------------
# FUNCIÓN DE LOCALIZACIÓN DE ARCHIVOS DICOM
# ------------------------------------------------------------
def find_dicom_file(csv_path, df_row):
    """
    Localiza el archivo DICOM correspondiente a un registro del CSV mediante
    estrategias de búsqueda jerárquica basadas en metadatos.
    
    Parámetros:
    -----------
    csv_path : str
        Ruta al archivo DICOM especificada en el CSV
    df_row : pandas.Series
        Fila del DataFrame con metadatos del caso
    
    Retorna:
    --------
    Path or None
        Ruta al archivo DICOM encontrado o None si no se localiza
    """
    csv_path_str = str(csv_path).lower()
    
    # Extracción de identificador de paciente desde la ruta del CSV
    patient_match = re.search(r'p_(\d{5})', csv_path_str, re.IGNORECASE)
    if patient_match:
        patient_num = patient_match.group(1)
        # Búsqueda primaria por número de paciente
        if patient_num in dicom_search_dict:
            candidates = dicom_search_dict[patient_num]
            if len(candidates) == 1:
                return candidates[0]
    
    # Búsqueda secundaria por lateralidad y vista mamográfica
    laterality = str(df_row.get('left or right breast', '')).upper()
    view = str(df_row.get('image view', '')).upper()
    view_key = f"{laterality}_{view}"
    
    if view_key in dicom_search_dict:
        candidates = dicom_search_dict[view_key]
        # Refinamiento con identificador de paciente si está disponible
        if patient_match:
            patient_num = patient_match.group(1)
            for cand in candidates:
                if patient_num in str(cand):
                    return cand
        return candidates[0] if candidates else None
    
    # Búsqueda exhaustiva en todos los archivos DICOM como último recurso
    for dcm_path in dicom_files:
        # Filtrado por identificador de paciente
        if patient_match:
            if patient_match.group(0).lower() in str(dcm_path).lower():
                return dcm_path
        
        # Filtrado por lateralidad y vista
        if laterality and laterality in str(dcm_path).upper():
            if view and view in str(dcm_path).upper():
                return dcm_path
    
    return None

# ------------------------------------------------------------
# FUNCIÓN DE CONVERSIÓN DICOM A PNG
# ------------------------------------------------------------
def convert_dicom_to_png(dicom_path, output_path, size=(224, 224)):
    """
    Convierte un archivo DICOM a formato PNG con normalización de intensidades
    y redimensionamiento espacial.
    
    Parámetros:
    -----------
    dicom_path : Path
        Ruta al archivo DICOM de entrada
    output_path : Path
        Ruta de salida para el archivo PNG
    size : tuple, opcional
        Dimensiones de salida en píxeles (ancho, alto)
    
    Retorna:
    --------
    bool
        True si la conversión fue exitosa, False en caso contrario
    """
    try:
        # Lectura del archivo DICOM
        ds = pydicom.dcmread(dicom_path, force=True)
        img_array = ds.pixel_array
        
        # Normalización lineal a rango 0-255 (8 bits)
        img_array = ((img_array - img_array.min()) / 
                     (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)
        
        # Conversión a objeto Image de PIL y redimensionamiento
        img = Image.fromarray(img_array).convert('L')
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(output_path, 'PNG')
        return True
    except Exception as e:
        print(f"Error en la conversión de {dicom_path}: {e}")
        return False

# ------------------------------------------------------------
# PROCESAMIENTO PRINCIPAL DE IMÁGENES DE BAJO RIESGO
# ------------------------------------------------------------
processed_count = 0
error_count = 0
metadata_list = []

# Identificación de la columna que contiene rutas a archivos DICOM
csv_path_col = 'image file path'

print(f"\nIniciando procesamiento de {TARGET_COUNT} imágenes de bajo riesgo...")
for idx, row in df_filtered.iterrows():
    # Validación de existencia de ruta en el registro
    if csv_path_col not in row or pd.isna(row[csv_path_col]):
        error_count += 1
        continue
    
    csv_dicom_path = str(row[csv_path_col])
    patient_id = str(row.get('patient_id', f"case_{idx}")).replace("/", "_")
    laterality = str(row.get('left or right breast', 'unknown')).upper()
    view = str(row.get('image view', 'unknown')).upper()
    birads = row[assessment_col]
    density = row[density_col]
    
    # Localización del archivo DICOM correspondiente
    dcm_path = find_dicom_file(csv_dicom_path, row)
    
    if not dcm_path:
        print(f"No se pudo localizar archivo DICOM para: {csv_dicom_path}")
        error_count += 1
        continue
    
    # Construcción de nombre de archivo con información estructurada
    output_filename = f"B{birads}_D{density_map.get(density, density)}_{patient_id}_{laterality}_{view}_{idx}.png"
    output_full_path = output_dir / output_filename
    
    # Ejecución de conversión DICOM a PNG
    if convert_dicom_to_png(dcm_path, output_full_path):
        metadata_list.append({
            'patient_id': patient_id,
            'png_filename': output_filename,
            'birads': birads,
            'density': density,
            'density_label': density_map.get(density, 'Unknown'),
            'csv_dicom_path': csv_dicom_path,
            'real_dicom_path': str(dcm_path.relative_to(base_path)),
            'laterality': laterality,
            'view': view,
            'clase': 'bajo_riesgo'
        })
        processed_count += 1
        
        # Reporte periódico de progreso
        if processed_count % 50 == 0:
            print(f"Imágenes procesadas: {processed_count}/{TARGET_COUNT}")
    else:
        error_count += 1
    
    # Condición de terminación cuando se alcanza el objetivo
    if processed_count >= TARGET_COUNT:
        break

# ------------------------------------------------------------
# ALMACENAMIENTO DE METADATOS Y RESULTADOS
# ------------------------------------------------------------
if metadata_list:
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(output_dir / "metadata_bajo_riesgo.csv", index=False)
    print(f"\nMetadatos almacenados en: {output_dir / 'metadata_bajo_riesgo.csv'}")
    
    # Generación de archivo de resumen del procesamiento
    summary_path = output_dir / "resumen_procesamiento.txt"
    with open(summary_path, 'w') as f:
        f.write(f"IMÁGENES DE BAJO RIESGO PROCESADAS: {processed_count}\n")
        f.write(f"OBJETIVO ESTABLECIDO: {TARGET_COUNT}\n")
        f.write(f"ERRORES ENCONTRADOS: {error_count}\n")
        f.write(f"FECHA DE PROCESAMIENTO: {pd.Timestamp.now()}\n")
        f.write(f"CRITERIOS DE SELECCIÓN: BI-RADS 1-3, Densidad C/D (3-4)\n")
        f.write(f"DISTRIBUCIÓN POR CATEGORÍA BI-RADS:\n")
        for birads, count in birads_counts.items():
            f.write(f"  BI-RADS {birads}: {count}\n")
        f.write(f"DISTRIBUCIÓN POR DENSIDAD MAMARIA:\n")
        for density, count in density_counts.items():
            density_name = density_map.get(density, f"Desconocida ({density})")
            f.write(f"  {density_name}: {count}\n")

# ------------------------------------------------------------
# REPORTE FINAL DEL PROCESAMIENTO
# ------------------------------------------------------------
print("\n" + "="*60)
print("RESUMEN FINAL - PROCESAMIENTO DE BAJO RIESGO")
print("="*60)
print(f"Registros disponibles (BI-RADS 1-3, Densidad 3-4): {len(df_bajo_riesgo)}")
print(f"Objetivo de muestreo establecido: {TARGET_COUNT} imágenes")
print(f"Imágenes convertidas exitosamente: {processed_count}")
print(f"Registros con problemas de procesamiento: {error_count}")
print(f"Tasa de éxito del procesamiento: {processed_count/(processed_count+error_count)*100:.1f}%")
print(f"Directorio de salida de resultados: {output_dir}")
print("="*60)

if processed_count > 0:
    print("\nEjemplos de archivos procesados:")
    for i, meta in enumerate(metadata_list[:5]):
        print(f"  {i+1}. {meta['png_filename']}")
