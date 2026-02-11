"""
Autor: Jhon Jaime Vaca Hincapié
Maestría en Ingeniería
Fundación Universitaria Los Libertadores
2025

Script: procesamiento_inbreast_alto_riesgo.py
Descripción: Procesa el dataset INbreast para extraer imágenes mamográficas
             correspondientes a casos de alto riesgo (BI-RADS 4-6) con densidad
             mamaria tipo C o D. Incluye carga de metadatos, filtrado clínico,
             localización de archivos DICOM y conversión a formato PNG estandarizado.
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
output_dir = base_path / "inbreast_alto_riesgo_densas"
output_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. CARGA Y ESTRUCTURACIÓN DE METADATOS
# ------------------------------------------------------------
csv_path = base_path / "INbreast.csv"
print("Cargando metadatos desde:", csv_path)

# Verificación de existencia del archivo CSV
if not csv_path.exists():
    print("ERROR: No se encontró el archivo INbreast.csv")
    exit()

# Carga del archivo CSV con separador punto y coma
df = pd.read_csv(csv_path, sep=';')
print("Total de registros cargados:", len(df))
print("Columnas disponibles:", df.columns.tolist())

# ------------------------------------------------------------
# 2. IDENTIFICACIÓN DE COLUMNAS RELEVANTES
# ------------------------------------------------------------
# Asignación de nombres de columnas según la estructura de INbreast
birads_col = 'Bi-Rads'          # Columna que contiene la categoría BI-RADS
density_col = 'ACR'             # Columna que contiene la densidad mamaria ACR
filename_col = 'File Name'      # Columna con nombres de archivos DICOM
patient_col = 'Patient ID'      # Columna con identificadores de pacientes
laterality_col = 'Laterality'   # Columna con lateralidad (L/R)
view_col = 'View'               # Columna con vista mamográfica (CC/MLO)

print("\nColumnas clínicas identificadas:")
print("BI-RADS:", birads_col)
print("Densidad (ACR):", density_col)
print("Archivo DICOM:", filename_col)

# ------------------------------------------------------------
# 3. FUNCIONES DE PROCESAMIENTO DE DATOS CLÍNICOS
# ------------------------------------------------------------
def parse_birads(value):
    """
    Convierte valores de BI-RADS a formato numérico.
    Maneja casos especiales como '4a', '4b', '4c' que se convierten a 4.
    
    Parámetros:
    -----------
    value : str, int, float
        Valor original de BI-RADS
    
    Retorna:
    --------
    int or np.nan
        Valor numérico de BI-RADS o NaN si no es convertible
    """
    if pd.isna(value):
        return np.nan
    value_str = str(value).strip()
    # Extraer el primer dígito numérico encontrado
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
        Valor original de densidad ACR
    
    Retorna:
    --------
    int or np.nan
        Valor numérico de densidad (1-4) o NaN si no es convertible
    """
    if pd.isna(value):
        return np.nan
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return np.nan

# ------------------------------------------------------------
# 4. CONVERSIÓN Y NORMALIZACIÓN DE DATOS CLÍNICOS
# ------------------------------------------------------------
# Aplicar funciones de conversión a las columnas relevantes
df['birads_numeric'] = df[birads_col].apply(parse_birads)
df['density_numeric'] = df[density_col].apply(parse_density)

print("\nDistribución de categorías BI-RADS después de conversión:")
print(df['birads_numeric'].value_counts().sort_index())

print("\nDistribución de densidad mamaria después de conversión:")
print(df['density_numeric'].value_counts().sort_index())

# ------------------------------------------------------------
# 5. FILTRADO POR CRITERIOS CLÍNICOS ESPECÍFICOS
# ------------------------------------------------------------
# Aplicar filtros para seleccionar casos de alto riesgo con densidad C/D
df_filtered = df[
    (df['birads_numeric'] >= 4) & (df['birads_numeric'] <= 6) &  # BI-RADS 4, 5, 6
    (df['density_numeric'] >= 3) & (df['density_numeric'] <= 4)   # Densidad C, D (3, 4)
].copy()

print("\nRegistros que cumplen criterios (BI-RADS 4-6, Densidad C/D):", len(df_filtered))

if df_filtered.empty:
    print("No se encontraron imágenes que cumplan los criterios especificados.")
    exit()

# ------------------------------------------------------------
# 6. LOCALIZACIÓN DE ARCHIVOS DICOM EN EL SISTEMA DE ARCHIVOS
# ------------------------------------------------------------
dicom_folder = base_path / "AllDICOMs"
print("\nCarpeta DICOM identificada:", dicom_folder)

# Búsqueda recursiva de todos los archivos DICOM disponibles
dicom_files = list(dicom_folder.rglob("*.dcm"))
print("Total de archivos DICOM encontrados:", len(dicom_files))

if not dicom_files:
    print("ERROR: No se encontraron archivos DICOM en la carpeta especificada.")
    exit()

# ------------------------------------------------------------
# 7. CREACIÓN DE ÍNDICE DE BÚSQUEDA PARA ARCHIVOS DICOM
# ------------------------------------------------------------
print("\nConstruyendo índice de búsqueda para archivos DICOM...")
file_index = {}

for dcm_path in dicom_files:
    filename = dcm_path.stem  # Nombre sin extensión
    
    # Extraer secuencias de dígitos del nombre de archivo
    digits = ''.join(filter(str.isdigit, filename))
    
    if digits:
        # Indexación múltiple para búsquedas flexibles
        file_index[digits] = dcm_path          # Por secuencia completa de dígitos
        file_index[filename] = dcm_path        # Por nombre completo
        
        # Indexación por últimos dígitos para coincidencias parciales
        if len(digits) > 6:
            last_6 = digits[-6:]
            file_index[last_6] = dcm_path

print("Índice de búsqueda creado con", len(file_index), "entradas")

# ------------------------------------------------------------
# 8. FUNCIÓN DE BÚSQUEDA DE ARCHIVOS DICOM
# ------------------------------------------------------------
def find_dicom_file(file_number):
    """
    Localiza un archivo DICOM en el sistema de archivos utilizando múltiples
    estrategias de búsqueda.
    
    Parámetros:
    -----------
    file_number : int, str
        Número o identificador del archivo a buscar
    
    Retorna:
    --------
    Path or None
        Ruta al archivo DICOM encontrado o None si no se localiza
    """
    file_str = str(file_number).strip()
    
    # Estrategia 1: Búsqueda exacta en el índice
    if file_str in file_index:
        return file_index[file_str]
    
    # Estrategia 2: Búsqueda por coincidencia parcial en claves del índice
    for key, path in file_index.items():
        if file_str in key:
            return path
    
    # Estrategia 3: Búsqueda exhaustiva en nombres de archivo
    file_str_no_zeros = file_str.lstrip('0')
    for dcm_path in dicom_files:
        filename = dcm_path.stem
        if file_str in filename or file_str_no_zeros in filename:
            return dcm_path
    
    # Estrategia 4: Diagnóstico de archivos disponibles
    print("\nArchivos DICOM disponibles (muestra de 10):")
    for i, dcm_path in enumerate(dicom_files[:10], 1):
        print(" ", i, ".", dcm_path.name)
    
    return None

# ------------------------------------------------------------
# 9. FUNCIÓN DE CONVERSIÓN DICOM A PNG
# ------------------------------------------------------------
def convert_dicom_to_png(dicom_path, output_path, size=(224, 224)):
    """
    Convierte un archivo DICOM a formato PNG con normalización y redimensionamiento.
    
    Parámetros:
    -----------
    dicom_path : Path
        Ruta al archivo DICOM de entrada
    output_path : Path
        Ruta donde guardar el archivo PNG resultante
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
        
        # Normalización lineal de intensidades al rango 0-255
        img_array = ((img_array - img_array.min()) / 
                     (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)
        
        # Conversión a objeto Image de PIL y redimensionamiento
        img = Image.fromarray(img_array).convert('L')  # Escala de grises
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(output_path, 'PNG')
        return True
    except Exception as e:
        print("Error en conversión de archivo:", e)
        return False

# ------------------------------------------------------------
# 10. PROCESAMIENTO PRINCIPAL DE IMÁGENES
# ------------------------------------------------------------
print("\n" + "="*60)
print("INICIANDO PROCESAMIENTO DE", len(df_filtered), "IMÁGENES")
print("="*60)

processed_count = 0
error_count = 0
metadata_list = []

for idx, row in df_filtered.iterrows():
    file_number = row[filename_col]
    print("\nProceso", idx + 1, "de", len(df_filtered), "- Buscando archivo:", file_number)
    
    # Localización del archivo DICOM correspondiente
    dicom_path = find_dicom_file(file_number)
    
    if not dicom_path:
        print("  Archivo no encontrado:", file_number)
        error_count += 1
        continue
    
    print("  Archivo localizado:", dicom_path.name)
    
    # Extracción de información clínica del caso
    patient_id = str(row[patient_col]).replace(';', '_')
    laterality = str(row[laterality_col]).upper()
    view = str(row[view_col]).upper()
    birads = row['birads_numeric']
    density = row['density_numeric']
    
    # Construcción de nombre de archivo de salida
    output_filename = f"INB_{patient_id}_{laterality}_{view}_B{birads}_D{density}.png"
    output_full_path = output_dir / output_filename
    
    # Conversión DICOM a PNG
    print("  Realizando conversión a PNG...")
    if convert_dicom_to_png(dicom_path, output_full_path):
        metadata_list.append({
            'patient_id': patient_id,
            'png_filename': output_filename,
            'birads': birads,
            'density': density,
            'original_birads': row[birads_col],
            'original_density': row[density_col],
            'original_file_number': file_number,
            'dicom_filename': dicom_path.name,
            'dicom_path': str(dicom_path.relative_to(base_path)),
            'laterality': laterality,
            'view': view,
            'dataset': 'INbreast'
        })
        processed_count += 1
        print("  Archivo guardado como:", output_filename)
    else:
        error_count += 1
        print("  Error en proceso de conversión")

# ------------------------------------------------------------
# 11. ALMACENAMIENTO DE METADATOS Y RESULTADOS
# ------------------------------------------------------------
print("\n" + "="*60)
print("ALMACENANDO RESULTADOS DEL PROCESAMIENTO")
print("="*60)

if metadata_list:
    # Creación de DataFrame con metadatos
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(output_dir / "metadata_inbreast.csv", index=False)
    print("Metadatos almacenados:", len(metadata_df), "registros")
    
    # Generación de archivo de resumen
    summary_path = output_dir / "resumen_procesamiento.txt"
    with open(summary_path, 'w') as f:
        f.write("PROCESAMIENTO DEL DATASET INbreast\n")
        f.write("Fecha: " + str(pd.Timestamp.now()) + "\n")
        f.write("Imágenes procesadas exitosamente: " + str(processed_count) + "\n")
        f.write("Errores encontrados: " + str(error_count) + "\n")
        
        if processed_count + error_count > 0:
            success_rate = processed_count / (processed_count + error_count) * 100
            f.write("Tasa de éxito del procesamiento: " + f"{success_rate:.1f}%\n\n")
        
        f.write("Distribución por categoría BI-RADS:\n")
        for birads, count in metadata_df['birads'].value_counts().sort_index().items():
            f.write("  BI-RADS " + str(birads) + ": " + str(count) + "\n")
        
        f.write("\nDistribución por densidad mamaria:\n")
        for density, count in metadata_df['density'].value_counts().sort_index().items():
            label = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}.get(density, "Desconocida")
            f.write("  Densidad " + label + ": " + str(count) + "\n")
        
        f.write("\nDistribución por vista mamográfica:\n")
        for view, count in metadata_df['view'].value_counts().items():
            f.write("  " + view + ": " + str(count) + "\n")
    
    print("Resumen del procesamiento guardado en: resumen_procesamiento.txt")
    
    # Visualización de ejemplos de archivos procesados
    print("\nEjemplos de archivos generados:")
    for i, meta in enumerate(metadata_list[:3]):
        print(" ", i + 1, ".", meta['png_filename'])
else:
    print("No se procesaron imágenes durante esta ejecución.")

# ------------------------------------------------------------
# 12. REPORTE FINAL DEL PROCESAMIENTO
# ------------------------------------------------------------
print("\n" + "="*60)
print("REPORTE FINAL DEL PROCESAMIENTO")
print("="*60)
print("Total de registros en dataset INbreast:", len(df))
print("Registros que cumplen criterios (BI-RADS 4-6, Densidad C/D):", len(df_filtered))
print("Imágenes procesadas exitosamente:", processed_count)
print("Errores durante el procesamiento:", error_count)

if processed_count + error_count > 0:
    success_rate = processed_count / (processed_count + error_count) * 100
    print("Tasa de éxito del procesamiento:", f"{success_rate:.1f}%")

print("Directorio de salida para resultados:", output_dir)
print("="*60)

# ------------------------------------------------------------
# 13. DIAGNÓSTICO ADICIONAL EN CASO DE ERRORES
# ------------------------------------------------------------
if error_count > 0:
    print("\nInformación adicional para", error_count, "errores encontrados:")
    print("Posibles causas y soluciones:")
    print("1. Verificar correspondencia entre números en 'File Name' y nombres reales de archivos")
    print("2. Los archivos DICOM pueden tener variaciones en la nomenclatura")
    print("3. Ejecutar análisis de correspondencia de nombres:")
    
    print("\nAnálisis de correspondencia (primeros 5 casos con error):")
    for i, row in df_filtered.head(5).iterrows():
        file_num = row[filename_col]
        print("\n  Referencia en CSV:", file_num)
        
        # Búsqueda de coincidencias parciales para diagnóstico
        matches = []
        for dcm_path in dicom_files:
            if str(file_num) in dcm_path.stem:
                matches.append(dcm_path.name)
        
        if matches:
            print("  Coincidencias encontradas en sistema:", matches[:3])
        else:
            print("  No se encontraron coincidencias en el sistema de archivos")
