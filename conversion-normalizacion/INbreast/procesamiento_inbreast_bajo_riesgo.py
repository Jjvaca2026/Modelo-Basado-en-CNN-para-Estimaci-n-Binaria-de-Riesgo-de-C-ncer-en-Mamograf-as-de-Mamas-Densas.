"""
Autor: Jhon Jaime Vaca Hincapié
Maestría en Ingeniería
Fundación Universitaria Los Libertadores
2025

Script: procesamiento_inbreast_bajo_riesgo.py
Descripción: Procesa el dataset INbreast para extraer todas las imágenes mamográficas
             correspondientes a casos de bajo riesgo (BI-RADS 1-3) con densidad
             mamaria tipo C o D. Este script procesa la totalidad de casos que
             cumplen los criterios, sin aplicar límites de cantidad, para obtener
             un conjunto completo de referencia para entrenamiento de modelos.
"""

import pandas as pd
import pydicom
import os
from PIL import Image
import numpy as np
from pathlib import Path
import warnings

# Suprimir advertencias no críticas para una salida más limpia
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# ------------------------------------------------------------
base_path = Path(r"C:\Users\Jhon\Desktop\Semestre 3\Proyecto Grado I\Tesis Maestría\inbreast-dataset\INbreast Release 1.0")
output_dir = base_path / "inbreast_bajo_riesgo_densas"
output_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. CARGA DEL ARCHIVO DE METADATOS
# ------------------------------------------------------------
csv_path = base_path / "INbreast.csv"
print("Cargando metadatos desde:", csv_path)

# Validación de existencia del archivo CSV
if not csv_path.exists():
    print("ERROR: No se encontró el archivo INbreast.csv en la ruta especificada.")
    exit()

# Carga del archivo CSV con separador punto y coma
df = pd.read_csv(csv_path, sep=';')
print("Total de registros cargados del dataset:", len(df))

# ------------------------------------------------------------
# 2. DEFINICIÓN DE COLUMNAS CLÍNICAS
# ------------------------------------------------------------
# Asignación de nombres de columnas según la estructura documentada de INbreast
birads_col = 'Bi-Rads'          # Columna que contiene la categoría BI-RADS
density_col = 'ACR'             # Columna que contiene la densidad mamaria según ACR
filename_col = 'File Name'      # Columna con identificadores de archivos DICOM
patient_col = 'Patient ID'      # Columna con identificadores únicos de pacientes
laterality_col = 'Laterality'   # Columna que indica lateralidad (Left/Right)
view_col = 'View'               # Columna que especifica la vista mamográfica

# ------------------------------------------------------------
# 3. FUNCIONES DE PROCESAMIENTO DE DATOS CLÍNICOS
# ------------------------------------------------------------
def parse_birads(value):
    """
    Convierte valores de BI-RADS a formato numérico estándar.
    
    Parámetros:
    -----------
    value : str, int, float
        Valor original de BI-RADS, puede incluir subcategorías (ej: '4a', '4b')
    
    Retorna:
    --------
    int or np.nan
        Valor numérico de BI-RADS (1-6) o NaN si no es convertible
    """
    if pd.isna(value):
        return np.nan
    value_str = str(value).strip()
    # Extracción del primer dígito numérico encontrado en la cadena
    for char in value_str:
        if char.isdigit():
            return int(char)
    return np.nan

def parse_density(value):
    """
    Convierte valores de densidad mamaria ACR a formato numérico.
    
    Parámetros:
    -----------
    value : str, int, float
        Valor original de densidad según clasificación ACR
    
    Retorna:
    --------
    int or np.nan
        Valor numérico de densidad (1=A, 2=B, 3=C, 4=D) o NaN si no es convertible
    """
    if pd.isna(value):
        return np.nan
    try:
        # Conversión directa para valores numéricos
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return np.nan

# ------------------------------------------------------------
# 4. APLICACIÓN DE CONVERSIONES Y NORMALIZACIÓN
# ------------------------------------------------------------
# Aplicar funciones de conversión a las columnas de interés clínico
df['birads_numeric'] = df[birads_col].apply(parse_birads)
df['density_numeric'] = df[density_col].apply(parse_density)

print("\nDistribución completa de categorías BI-RADS después de conversión:")
print(df['birads_numeric'].value_counts().sort_index())

print("\nDistribución completa de densidad mamaria después de conversión:")
print(df['density_numeric'].value_counts().sort_index())

# ------------------------------------------------------------
# 5. FILTRADO POR CRITERIOS DE BAJO RIESGO Y DENSIDAD C/D
# ------------------------------------------------------------
print("\n" + "="*60)
print("APLICACIÓN DE FILTROS CLÍNICOS PARA BAJO RIESGO")
print("="*60)

# Definición y aplicación de filtros combinados
df_bajo_riesgo = df[
    (df['birads_numeric'] >= 1) & (df['birads_numeric'] <= 3) &  # BI-RADS 1, 2, 3
    (df['density_numeric'] >= 3) & (df['density_numeric'] <= 4)   # Densidad C o D
].copy()

print("Total de registros que cumplen criterios (BI-RADS 1-3, Densidad C/D):", len(df_bajo_riesgo))

# Validación de que existan casos que cumplan los criterios
if df_bajo_riesgo.empty:
    print("No se encontraron imágenes que cumplan los criterios especificados de bajo riesgo.")
    exit()

# ------------------------------------------------------------
# 6. ANÁLISIS DETALLADO DE LA DISTRIBUCIÓN FILTRADA
# ------------------------------------------------------------
print("\nDistribución de categorías BI-RADS en el conjunto de bajo riesgo:")
print(df_bajo_riesgo['birads_numeric'].value_counts().sort_index())

print("\nDistribución de densidad mamaria en el conjunto de bajo riesgo:")
print(df_bajo_riesgo['density_numeric'].value_counts().sort_index())

print("\nDistribución por vista mamográfica en el conjunto de bajo riesgo:")
print(df_bajo_riesgo[view_col].value_counts())

# ------------------------------------------------------------
# 7. LOCALIZACIÓN Y PREPARACIÓN DE ARCHIVOS DICOM
# ------------------------------------------------------------
dicom_folder = base_path / "AllDICOMs"
print("\nCarpeta de archivos DICOM identificada:", dicom_folder)

# Búsqueda recursiva de todos los archivos DICOM disponibles
dicom_files = list(dicom_folder.rglob("*.dcm"))
print("Total de archivos DICOM encontrados en el sistema:", len(dicom_files))

# Validación de existencia de archivos DICOM
if not dicom_files:
    print("ERROR: No se encontraron archivos DICOM en la carpeta especificada.")
    exit()

# ------------------------------------------------------------
# 8. CONSTRUCCIÓN DE ÍNDICE DE BÚSQUEDA PARA ARCHIVOS DICOM
# ------------------------------------------------------------
print("\nConstruyendo índice de búsqueda optimizado para archivos DICOM...")
file_index = {}

for dcm_path in dicom_files:
    filename = dcm_path.stem  # Nombre del archivo sin extensión
    
    # Extracción de secuencias numéricas del nombre de archivo
    digits = ''.join(filter(str.isdigit, filename))
    
    if digits:
        # Indexación múltiple para permitir diferentes estrategias de búsqueda
        file_index[digits] = dcm_path          # Indexación por secuencia completa de dígitos
        file_index[filename] = dcm_path        # Indexación por nombre completo
        
        # Indexación adicional por últimos dígitos para coincidencias parciales
        if len(digits) > 6:
            last_6 = digits[-6:]
            file_index[last_6] = dcm_path

print("Índice de búsqueda creado con", len(file_index), "entradas de referencia")

# ------------------------------------------------------------
# 9. FUNCIÓN DE BÚSQUEDA DE ARCHIVOS DICOM
# ------------------------------------------------------------
def find_dicom_file(file_number):
    """
    Localiza un archivo DICOM en el sistema de archivos utilizando estrategias
    de búsqueda jerárquicas para maximizar la probabilidad de éxito.
    
    Parámetros:
    -----------
    file_number : int, str
        Identificador o número de archivo a localizar
    
    Retorna:
    --------
    Path or None
        Ruta al archivo DICOM encontrado o None si no se localiza
    """
    file_str = str(file_number).strip()
    
    # Estrategia 1: Búsqueda exacta en el índice construido
    if file_str in file_index:
        return file_index[file_str]
    
    # Estrategia 2: Búsqueda por coincidencia parcial en claves del índice
    for key, path in file_index.items():
        if file_str in key:
            return path
    
    # Estrategia 3: Búsqueda exhaustiva en nombres de archivo originales
    file_str_no_zeros = file_str.lstrip('0')
    for dcm_path in dicom_files:
        filename = dcm_path.stem
        if file_str in filename or file_str_no_zeros in filename:
            return dcm_path
    
    return None

# ------------------------------------------------------------
# 10. FUNCIÓN DE CONVERSIÓN DICOM A FORMATO PNG
# ------------------------------------------------------------
def convert_dicom_to_png(dicom_path, output_path, size=(224, 224)):
    """
    Convierte un archivo DICOM a formato PNG con normalización de intensidades
    y redimensionamiento espacial estándar.
    
    Parámetros:
    -----------
    dicom_path : Path
        Ruta al archivo DICOM de entrada
    output_path : Path
        Ruta donde se guardará el archivo PNG resultante
    size : tuple, opcional
        Dimensiones objetivo para redimensionamiento (ancho, alto)
    
    Retorna:
    --------
    bool
        True si la conversión se completó exitosamente, False en caso contrario
    """
    try:
        # Lectura del archivo DICOM
        ds = pydicom.dcmread(dicom_path, force=True)
        img_array = ds.pixel_array
        
        # Normalización lineal de intensidades al rango 0-255
        img_array = ((img_array - img_array.min()) / 
                     (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)
        
        # Conversión a objeto Image y redimensionamiento
        img = Image.fromarray(img_array).convert('L')  # Conversión a escala de grises
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(output_path, 'PNG')
        return True
    except Exception as e:
        print("Error durante la conversión del archivo:", e)
        return False

# ------------------------------------------------------------
# 11. PROCESAMIENTO PRINCIPAL DE TODAS LAS IMÁGENES DE BAJO RIESGO
# ------------------------------------------------------------
print("\n" + "="*60)
print("INICIANDO PROCESAMIENTO DE", len(df_bajo_riesgo), "IMÁGENES DE BAJO RIESGO")
print("="*60)

# Inicialización de contadores y estructuras para resultados
processed_count = 0
error_count = 0
metadata_list = []

# Iteración sobre todos los casos que cumplen criterios
for idx, row in df_bajo_riesgo.iterrows():
    file_number = row[filename_col]
    
    # Reporte periódico del progreso
    if idx % 20 == 0:
        print("Procesando imagen", idx + 1, "de", len(df_bajo_riesgo), "...")
    
    # Localización del archivo DICOM correspondiente
    dicom_path = find_dicom_file(file_number)
    
    if not dicom_path:
        print("Archivo no encontrado en el sistema:", file_number)
        error_count += 1
        continue
    
    # Extracción de información clínica para identificación del caso
    patient_id = str(row[patient_col]).replace(';', '_')
    laterality = str(row[laterality_col]).upper()
    view = str(row[view_col]).upper()
    birads = row['birads_numeric']
    density = row['density_numeric']
    
    # Construcción de nombre de archivo descriptivo
    density_label = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}.get(density, "D" + str(density))
    output_filename = f"INB_B{birads}_{density_label}_{patient_id}_{laterality}_{view}.png"
    output_full_path = output_dir / output_filename
    
    # Ejecución de la conversión DICOM a PNG
    if convert_dicom_to_png(dicom_path, output_full_path):
        metadata_list.append({
            'patient_id': patient_id,
            'png_filename': output_filename,
            'birads': birads,
            'birads_label': "B" + str(birads),
            'density': density,
            'density_label': density_label,
            'original_file_number': file_number,
            'dicom_filename': dicom_path.name,
            'dicom_path': str(dicom_path.relative_to(base_path)),
            'laterality': laterality,
            'view': view,
            'clase': 'bajo_riesgo',
            'dataset': 'INbreast'
        })
        processed_count += 1
    else:
        error_count += 1

# ------------------------------------------------------------
# 12. ALMACENAMIENTO DE METADATOS Y GENERACIÓN DE RESÚMENES
# ------------------------------------------------------------
print("\n" + "="*60)
print("ALMACENANDO RESULTADOS Y METADATOS DEL PROCESAMIENTO")
print("="*60)

if metadata_list:
    # Creación de DataFrame con metadatos estructurados
    metadata_df = pd.DataFrame(metadata_list)
    metadata_file = output_dir / "metadata_inbreast_bajo_riesgo.csv"
    metadata_df.to_csv(metadata_file, index=False)
    print("Metadatos almacenados en:", metadata_file)
    print("Total de registros de metadatos guardados:", len(metadata_df))
    
    # Generación de archivo de resumen detallado
    summary_path = output_dir / "resumen_procesamiento.txt"
    with open(summary_path, 'w') as f:
        f.write("PROCESAMIENTO DEL DATASET INbreast - CASOS DE BAJO RIESGO\n")
        f.write("Fecha de procesamiento: " + str(pd.Timestamp.now()) + "\n")
        f.write("="*60 + "\n")
        f.write("Total de imágenes procesadas exitosamente: " + str(processed_count) + "\n")
        f.write("Errores encontrados durante el procesamiento: " + str(error_count) + "\n")
        
        # Cálculo y registro de tasa de éxito
        if processed_count + error_count > 0:
            success_rate = processed_count / (processed_count + error_count) * 100
            f.write("Tasa de éxito del procesamiento: " + f"{success_rate:.1f}%\n")
        
        f.write("\nESTADÍSTICAS DETALLADAS DEL CONJUNTO PROCESADO:\n")
        
        # Distribución por categoría BI-RADS
        f.write("\nDistribución por categoría BI-RADS:\n")
        birads_distribution = metadata_df['birads'].value_counts().sort_index()
        for birads, count in birads_distribution.items():
            percentage = count / len(metadata_df) * 100
            f.write("  BI-RADS " + str(birads) + ": " + str(count) + " imágenes (" + f"{percentage:.1f}" + "%)\n")
        
        # Distribución por densidad mamaria
        f.write("\nDistribución por densidad mamaria:\n")
        density_distribution = metadata_df['density'].value_counts().sort_index()
        for density, count in density_distribution.items():
            label = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}.get(density, "Desconocida (" + str(density) + ")")
            percentage = count / len(metadata_df) * 100
            f.write("  Densidad " + label + ": " + str(count) + " imágenes (" + f"{percentage:.1f}" + "%)\n")
        
        # Distribución por vista mamográfica
        f.write("\nDistribución por vista mamográfica:\n")
        view_distribution = metadata_df['view'].value_counts()
        for view, count in view_distribution.items():
            percentage = count / len(metadata_df) * 100
            f.write("  " + view + ": " + str(count) + " imágenes (" + f"{percentage:.1f}" + "%)\n")
        
        # Distribución por lateralidad
        f.write("\nDistribución por lateralidad:\n")
        laterality_distribution = metadata_df['laterality'].value_counts()
        for lat, count in laterality_distribution.items():
            percentage = count / len(metadata_df) * 100
            f.write("  " + lat + ": " + str(count) + " imágenes (" + f"{percentage:.1f}" + "%)\n")
    
    print("Resumen detallado del procesamiento guardado en: resumen_procesamiento.txt")
    
    # Presentación de estadísticas clave en consola
    print("\nESTADÍSTICAS PRINCIPALES DEL CONJUNTO PROCESADO:")
    print("  Total de imágenes procesadas:", processed_count)
    print("  BI-RADS 1:", len(metadata_df[metadata_df['birads'] == 1]))
    print("  BI-RADS 2:", len(metadata_df[metadata_df['birads'] == 2]))
    print("  BI-RADS 3:", len(metadata_df[metadata_df['birads'] == 3]))
    print("  Densidad C:", len(metadata_df[metadata_df['density'] == 3]))
    print("  Densidad D:", len(metadata_df[metadata_df['density'] == 4]))
    
    print("\nEJEMPLOS DE ARCHIVOS GENERADOS:")
    sample_size = min(5, len(metadata_list))
    for i in range(sample_size):
        print(" ", i + 1, ".", metadata_list[i]['png_filename'])
else:
    print("No se procesaron imágenes durante esta ejecución.")

# ------------------------------------------------------------
# 13. REPORTE FINAL COMPREHENSIVO
# ------------------------------------------------------------
print("\n" + "="*60)
print("REPORTE FINAL COMPLETO - PROCESAMIENTO INbreast BAJO RIESGO")
print("="*60)
print("Total de registros en el dataset INbreast original:", len(df))
print("Registros que cumplen criterios (BI-RADS 1-3, Densidad C/D):", len(df_bajo_riesgo))
print("Imágenes procesadas exitosamente:", processed_count)
print("Errores durante el procesamiento:", error_count)

# Cálculo y presentación de métricas de rendimiento
if processed_count + error_count > 0:
    success_rate = processed_count / (processed_count + error_count) * 100
    print("Tasa de éxito global del procesamiento:", f"{success_rate:.1f}%")

print("Directorio de salida para resultados:", output_dir)
print("="*60)

# ------------------------------------------------------------
# 14. ANÁLISIS ADICIONAL PARA CONTEXTUALIZACIÓN DE RESULTADOS
# ------------------------------------------------------------
if processed_count > 0:
    # Cálculo de estadísticas contextuales
    total_densas = len(df[(df['density_numeric'] >= 3) & (df['density_numeric'] <= 4)])
    print("\nINFORMACIÓN CONTEXTUAL ADICIONAL:")
    print("  Total de imágenes con densidad C/D en INbreast:", total_densas)
    
    if total_densas > 0:
        processed_percentage = processed_count / total_densas * 100
        print("  Porcentaje de imágenes densas procesadas:", f"{processed_percentage:.1f}%")
    
    # Cálculo de relación entre bajo y alto riesgo (si se dispone de esa información)
    alto_riesgo_count = len(df[(df['birads_numeric'] >= 4) & (df['birads_numeric'] <= 6) & 
                               (df['density_numeric'] >= 3) & (df['density_numeric'] <= 4)])
    
    if alto_riesgo_count > 0:
        risk_ratio = processed_count / alto_riesgo_count
        print("  Relación Bajo Riesgo / Alto Riesgo en densidad C/D:", f"{risk_ratio:.2f}")
