"""
Autor: Jhon Jaime Vaca Hincapié
Maestría en Ingeniería
Fundación Universitaria Los Libertadores
2025

Script: mapeo_metadatos_vindr_mammo.py
Descripción: Creación de tabla de mapeo entre metadatos clínicos y archivos DICOM físicos
             para el dataset VINDr-Mammo. Este script resuelve el problema de 
             correspondencia entre los IDs en los archivos CSV y los nombres reales
             de archivos/carpetas DICOM mediante la lectura de metadatos DICOM.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================
# Configurar rutas según la estructura del dataset
BASE_DATASET_DIR = Path(r"E:\DrMammo")
IMAGES_DIR = BASE_DATASET_DIR / "images"
BREAST_LEVEL_CSV = BASE_DATASET_DIR / 'breast-level_annotations.csv'
OUTPUT_DIR = Path(r"E:\filtered_mammograms")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def explorar_estructura_dataset():
    """
    Explora la estructura real del dataset para entender la organización
    de archivos y carpetas.
    
    Retorna:
    --------
    dict
        Diccionario con información de la estructura
    """
    print("EXPLORACIÓN DE ESTRUCTURA DEL DATASET VINDr-MAMMO")
    print("="*80)
    
    resultados = {
        'directorio_base_existe': BASE_DATASET_DIR.exists(),
        'directorio_imagenes_existe': IMAGES_DIR.exists(),
        'archivo_csv_existe': BREAST_LEVEL_CSV.exists(),
        'carpetas_estudio': 0,
        'ejemplo_carpeta': None,
        'ejemplo_archivos': []
    }
    
    # Verificar existencia de directorios y archivos
    print(f"Directorio base: {BASE_DATASET_DIR}")
    print(f"  Existe: {'SÍ' if resultados['directorio_base_existe'] else 'NO'}")
    
    print(f"\nDirectorio de imágenes: {IMAGES_DIR}")
    print(f"  Existe: {'SÍ' if resultados['directorio_imagenes_existe'] else 'NO'}")
    
    print(f"\nArchivo CSV: {BREAST_LEVEL_CSV}")
    print(f"  Existe: {'SÍ' if resultados['archivo_csv_existe'] else 'NO'}")
    
    # Explorar estructura de carpetas si existe
    if IMAGES_DIR.exists():
        # Contar carpetas de estudio
        carpetas_estudio = [d for d in IMAGES_DIR.iterdir() if d.is_dir()]
        resultados['carpetas_estudio'] = len(carpetas_estudio)
        
        print(f"\nCARPETAS DE ESTUDIO ENCONTRADAS: {len(carpetas_estudio)}")
        
        if carpetas_estudio:
            # Tomar primera carpeta como ejemplo
            carpeta_ejemplo = carpetas_estudio[0]
            resultados['ejemplo_carpeta'] = carpeta_ejemplo.name
            
            print(f"\nCARPETA DE EJEMPLO: {carpeta_ejemplo.name}")
            
            # Listar archivos en la carpeta de ejemplo
            archivos_carpeta = list(carpeta_ejemplo.iterdir())
            resultados['ejemplo_archivos'] = [f.name for f in archivos_carpeta[:5]]
            
            print(f"  Total archivos en carpeta: {len(archivos_carpeta)}")
            print(f"  Primeros 5 archivos:")
            for i, archivo in enumerate(archivos_carpeta[:5], 1):
                print(f"    {i}. {archivo.name}")
            
            # Identificar archivos DICOM
            extensiones_dicom = ['.dicom', '.dcm', '.DCM', '.DICOM']
            archivos_dicom = []
            
            for ext in extensiones_dicom:
                archivos_dicom.extend(list(carpeta_ejemplo.glob(f"*{ext}")))
            
            print(f"\n  Archivos identificados como DICOM: {len(archivos_dicom)}")
            if archivos_dicom:
                print(f"  Ejemplo archivo DICOM: {archivos_dicom[0].name}")
                
                # Leer metadatos DICOM del primer archivo
                try:
                    ds = pydicom.dcmread(archivos_dicom[0], stop_before_pixels=True)
                    
                    print(f"\n  METADATOS DICOM DEL ARCHIVO DE EJEMPLO:")
                    print(f"    SOPInstanceUID: {getattr(ds, 'SOPInstanceUID', 'No disponible')}")
                    print(f"    StudyInstanceUID: {getattr(ds, 'StudyInstanceUID', 'No disponible')}")
                    print(f"    PatientID: {getattr(ds, 'PatientID', 'No disponible')}")
                    print(f"    PhotometricInterpretation: {getattr(ds, 'PhotometricInterpretation', 'No disponible')}")
                    
                except Exception as e:
                    print(f"  ERROR al leer metadatos DICOM: {e}")
    
    return resultados

def crear_tabla_mapeo_por_metadata():
    """
    Crea tabla de mapeo leyendo SOPInstanceUID de cada archivo DICOM
    y buscando correspondencia con los image_id en el CSV.
    
    Retorna:
    --------
    pandas.DataFrame or None
        DataFrame con el mapeo completo o None si no se encontraron coincidencias
    """
    print("\nCREACIÓN DE TABLA DE MAPEO POR METADATOS DICOM")
    print("Lectura de SOPInstanceUIDs y búsqueda de correspondencia con CSV")
    print("="*80)
    
    # Estructura para almacenar el mapeo
    mapeo = {
        'study_id': [],           # ID del estudio desde CSV
        'image_id': [],           # ID de la imagen desde CSV (debe coincidir con SOPInstanceUID)
        'nombre_carpeta': [],     # Nombre de la carpeta física
        'nombre_archivo': [],     # Nombre del archivo físico
        'ruta_completa_archivo': [],  # Ruta completa al archivo DICOM
        'sop_instance_uid': [],   # SOPInstanceUID leído del archivo DICOM
        'study_instance_uid': [], # StudyInstanceUID leído del archivo DICOM
        'patient_id': []          # PatientID leído del archivo DICOM
    }
    
    # Cargar datos clínicos desde CSV
    print("Cargando datos clínicos desde CSV...")
    try:
        df_clinico = pd.read_csv(BREAST_LEVEL_CSV)
        
        # Crear columna numérica de BI-RADS para filtrado posterior
        if 'breast_birads' in df_clinico.columns:
            df_clinico['birads_numerico'] = df_clinico['breast_birads'].str.extract(r'(\d+)').astype(float)
        
        print(f"  Registros cargados: {len(df_clinico)}")
        print(f"  Columnas disponibles: {', '.join(df_clinico.columns.tolist())}")
        
    except Exception as e:
        print(f"ERROR al cargar archivo CSV: {e}")
        return None
    
    # Contadores para seguimiento del progreso
    estadisticas = {
        'archivos_procesados': 0,
        'archivos_con_sop_uid': 0,
        'coincidencias_encontradas': 0,
        'errores_lectura': 0
    }
    
    # Recorrer todas las carpetas de estudio
    print("\nProcesando archivos DICOM...")
    
    for carpeta in IMAGES_DIR.iterdir():
        if not carpeta.is_dir():
            continue
        
        # Buscar archivos DICOM en la carpeta
        patrones_dicom = ['*.dicom', '*.DICOM', '*.dcm', '*.DCM']
        archivos_dicom = []
        
        for patron in patrones_dicom:
            archivos_dicom.extend(list(carpeta.glob(patron)))
        
        # Procesar cada archivo DICOM
        for archivo_dicom in archivos_dicom:
            estadisticas['archivos_procesados'] += 1
            
            try:
                # Leer metadatos DICOM (sin cargar datos de píxeles para eficiencia)
                ds = pydicom.dcmread(archivo_dicom, stop_before_pixels=True)
                
                # Obtener SOPInstanceUID (identificador único de la imagen)
                sop_uid = getattr(ds, 'SOPInstanceUID', None)
                study_uid = getattr(ds, 'StudyInstanceUID', None)
                patient_id = getattr(ds, 'PatientID', None)
                
                if sop_uid:
                    estadisticas['archivos_con_sop_uid'] += 1
                    
                    # Buscar este SOPInstanceUID en los datos clínicos
                    registros_coincidentes = df_clinico[df_clinico['image_id'] == sop_uid]
                    
                    if not registros_coincidentes.empty:
                        estadisticas['coincidencias_encontradas'] += 1
                        
                        # Para cada registro coincidente, agregar entrada al mapeo
                        for _, registro in registros_coincidentes.iterrows():
                            mapeo['study_id'].append(registro['study_id'])
                            mapeo['image_id'].append(registro['image_id'])
                            mapeo['nombre_carpeta'].append(carpeta.name)
                            mapeo['nombre_archivo'].append(archivo_dicom.name)
                            mapeo['ruta_completa_archivo'].append(str(archivo_dicom))
                            mapeo['sop_instance_uid'].append(sop_uid)
                            mapeo['study_instance_uid'].append(study_uid)
                            mapeo['patient_id'].append(patient_id)
                        
                        # Mostrar primeras coincidencias para validación
                        if estadisticas['coincidencias_encontradas'] <= 3:
                            print(f"  Coincidencia {estadisticas['coincidencias_encontradas']}:")
                            print(f"    SOPInstanceUID: {sop_uid[:30]}...")
                            print(f"    → CSV: study_id={registro['study_id'][:12]}..., image_id={registro['image_id'][:12]}...")
                            print(f"    → Archivo: {carpeta.name}/{archivo_dicom.name}")
                
                # Mostrar progreso periódicamente
                if estadisticas['archivos_procesados'] % 100 == 0:
                    print(f"  Procesados: {estadisticas['archivos_procesados']} archivos, "
                          f"Coincidencias: {estadisticas['coincidencias_encontradas']}")
                    
            except Exception as e:
                estadisticas['errores_lectura'] += 1
                # Continuar con siguiente archivo si hay error
                continue
    
    # Crear DataFrame con el mapeo
    print("\nFINALIZANDO PROCESAMIENTO...")
    df_mapeo = pd.DataFrame(mapeo)
    
    # Reportar estadísticas finales
    print(f"\nESTADÍSTICAS DEL PROCESAMIENTO:")
    print(f"  Archivos DICOM procesados: {estadisticas['archivos_procesados']}")
    print(f"  Archivos con SOPInstanceUID: {estadisticas['archivos_con_sop_uid']}")
    print(f"  Coincidencias encontradas: {estadisticas['coincidencias_encontradas']}")
    print(f"  Errores de lectura: {estadisticas['errores_lectura']}")
    
    if len(df_mapeo) > 0:
        # Guardar tabla de mapeo
        ruta_mapeo = OUTPUT_DIR / "tabla_mapeo_completo.csv"
        df_mapeo.to_csv(ruta_mapeo, index=False)
        
        print(f"\nTABLA DE MAPEO GENERADA EXITOSAMENTE:")
        print(f"  Registros en tabla: {len(df_mapeo)}")
        print(f"  Estudios únicos mapeados: {df_mapeo['study_id'].nunique()}")
        print(f"  Imágenes únicas mapeadas: {df_mapeo['image_id'].nunique()}")
        print(f"  Carpetas físicas involucradas: {df_mapeo['nombre_carpeta'].nunique()}")
        print(f"  Archivo guardado: {ruta_mapeo}")
        
        # Mostrar ejemplo del mapeo
        print(f"\nEJEMPLO DEL MAPEO (primeras 3 filas):")
        columnas_ejemplo = ['study_id', 'image_id', 'nombre_carpeta', 'nombre_archivo']
        print(df_mapeo[columnas_ejemplo].head(3).to_string())
        
        return df_mapeo
    else:
        print("\nNO SE ENCONTRARON COINCIDENCIAS")
        print("Los SOPInstanceUIDs de los archivos DICOM no coinciden con los image_id del CSV.")
        
        # Recomendaciones para resolver el problema
        print("\nRECOMENDACIONES:")
        print("1. Verificar que los image_id en el CSV sean efectivamente SOPInstanceUIDs")
        print("2. Revisar si hay transformaciones (hashing) aplicadas a los IDs")
        print("3. Buscar archivos de mapeo adicionales en el dataset")
        print("4. Contactar a los creadores del dataset para aclarar la estructura")
        
        return None

def buscar_archivos_mapeo_adicionales():
    """
    Busca archivos de mapeo o documentación adicional en el dataset.
    
    Retorna:
    --------
    list
        Lista de archivos encontrados
    """
    print("\nBÚSQUEDA DE ARCHIVOS DE MAPEO ADICIONALES")
    print("="*80)
    
    archivos_buscados = [
        "file_mapping.csv", "mapping.csv", "image_mapping.csv",
        "dicom_mapping.csv", "file_list.csv", "metadata.csv",
        "readme.txt", "README.md", "documentation.txt"
    ]
    
    archivos_encontrados = []
    
    for nombre_archivo in archivos_buscados:
        ruta_archivo = BASE_DATASET_DIR / nombre_archivo
        if ruta_archivo.exists():
            archivos_encontrados.append(nombre_archivo)
            print(f"  Encontrado: {nombre_archivo}")
            
            # Intentar leer archivos CSV
            if nombre_archivo.endswith('.csv'):
                try:
                    df_temp = pd.read_csv(ruta_archivo)
                    print(f"    Filas: {len(df_temp)}, Columnas: {df_temp.columns.tolist()}")
                except:
                    print(f"    No se pudo leer como CSV")
    
    if not archivos_encontrados:
        print("  No se encontraron archivos de mapeo adicionales")
    
    return archivos_encontrados

def filtrar_mamas_alto_riesgo_con_mapeo(df_clinico, df_mapeo):
    """
    Filtra mamas de alto riesgo usando la tabla de mapeo.
    
    Parámetros:
    -----------
    df_clinico : pandas.DataFrame
        Datos clínicos completos
    df_mapeo : pandas.DataFrame
        Tabla de mapeo generada
    
    Retorna:
    --------
    pandas.DataFrame
        DataFrame con mamas de alto riesgo y sus rutas mapeadas
    """
    print("\nFILTRADO DE MAMAS DE ALTO RIESGO CON MAPEO")
    print("="*80)
    
    # Verificar que las columnas requeridas existan
    columnas_requeridas = ['breast_birads', 'breast_density', 'birads_numerico']
    for col in columnas_requeridas:
        if col not in df_clinico.columns:
            print(f"ERROR: Columna requerida '{col}' no encontrada en datos clínicos")
            return pd.DataFrame()
    
    # Filtrar mamas de alto riesgo: BI-RADS 4-5 y densidad C/D
    mascara_alto_riesgo = (
        (df_clinico['breast_density'].isin(['DENSITY C', 'DENSITY D'])) & 
        (df_clinico['birads_numerico'] >= 4)
    )
    
    df_alto_riesgo = df_clinico[mascara_alto_riesgo].copy()
    
    print(f"Mamas de alto riesgo identificadas: {len(df_alto_riesgo)}")
    
    if len(df_alto_riesgo) == 0:
        print("No se encontraron mamas de alto riesgo con los criterios especificados")
        return pd.DataFrame()
    
    # Cruzar con la tabla de mapeo
    df_alto_riesgo_mapeado = pd.merge(
        df_alto_riesgo,
        df_mapeo,
        left_on='image_id',
        right_on='image_id',
        how='inner'
    )
    
    print(f"Mamas de alto riesgo con mapeo encontrado: {len(df_alto_riesgo_mapeado)}")
    
    if len(df_alto_riesgo_mapeado) > 0:
        # Mostrar distribución
        print(f"\nDISTRIBUCIÓN DE MAMAS DE ALTO RIESGO MAPEADAS:")
        print(f"  BI-RADS: {df_alto_riesgo_mapeado['breast_birads'].value_counts().to_dict()}")
        print(f"  Densidad: {df_alto_riesgo_mapeado['breast_density'].value_counts().to_dict()}")
        
        # Mostrar ejemplo detallado
        print(f"\nEJEMPLO DE DATOS MAPEADOS:")
        ejemplo = df_alto_riesgo_mapeado.iloc[0]
        print(f"  Study ID: {ejemplo['study_id_x']}")
        print(f"  Image ID: {ejemplo['image_id']}")
        print(f"  BI-RADS: {ejemplo['breast_birads']}")
        print(f"  Densidad: {ejemplo['breast_density']}")
        print(f"  Lateralidad: {ejemplo.get('laterality', 'No disponible')}")
        print(f"  Archivo físico: {ejemplo['nombre_archivo']}")
        print(f"  Carpeta: {ejemplo['nombre_carpeta']}")
    
    return df_alto_riesgo_mapeado

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
def main():
    """
    Función principal que coordina el proceso de mapeo.
    """
    print("PROCESO DE MAPEO METADATOS-ARCHIVOS - VINDr-MAMMO")
    print("Creación de tabla de correspondencia entre datos clínicos y archivos DICOM")
    print("="*80)
    
    # 1. Explorar estructura del dataset
    estructura = explorar_estructura_dataset()
    
    if not all([estructura['directorio_base_existe'], 
                estructura['directorio_imagenes_existe'], 
                estructura['archivo_csv_existe']]):
        print("\nERROR: No se pudo acceder a todos los componentes del dataset")
        return
    
    # 2. Buscar archivos de mapeo adicionales
    archivos_mapeo = buscar_archivos_mapeo_adicionales()
    
    # 3. Crear tabla de mapeo principal
    df_mapeo = crear_tabla_mapeo_por_metadata()
    
    if df_mapeo is None or len(df_mapeo) == 0:
        print("\nNo se pudo crear la tabla de mapeo principal.")
        print("Se requieren estrategias alternativas para resolver la correspondencia.")
        return
    
    # 4. Cargar datos clínicos para filtrado
    print("\nCARGANDO DATOS CLÍNICOS PARA FILTRADO...")
    try:
        df_clinico = pd.read_csv(BREAST_LEVEL_CSV)
        
        # Crear columna numérica de BI-RADS si existe
        if 'breast_birads' in df_clinico.columns:
            df_clinico['birads_numerico'] = df_clinico['breast_birads'].str.extract(r'(\d+)').astype(float)
        
        print(f"  Registros clínicos cargados: {len(df_clinico)}")
        
    except Exception as e:
        print(f"ERROR al cargar datos clínicos: {e}")
        return
    
    # 5. Filtrar mamas de alto riesgo usando el mapeo
    df_alto_riesgo_mapeado = filtrar_mamas_alto_riesgo_con_mapeo(df_clinico, df_mapeo)
    
    if len(df_alto_riesgo_mapeado) > 0:
        # Guardar resultados del filtrado
        ruta_resultados = OUTPUT_DIR / "mamas_alto_riesgo_mapeadas.csv"
        df_alto_riesgo_mapeado.to_csv(ruta_resultados, index=False)
        
        print(f"\nRESULTADOS GUARDADOS:")
        print(f"  Tabla de mapeo completo: {OUTPUT_DIR / 'tabla_mapeo_completo.csv'}")
        print(f"  Mamas alto riesgo mapeadas: {ruta_resultados}")
        print(f"  Total imágenes listas para procesamiento: {len(df_alto_riesgo_mapeado)}")
        
        # Resumen ejecutivo
        print(f"\nRESUMEN EJECUTIVO:")
        print(f"  • Tabla de mapeo creada exitosamente con {len(df_mapeo)} registros")
        print(f"  • {len(df_alto_riesgo_mapeado)} imágenes de alto riesgo identificadas y mapeadas")
        print(f"  • Archivos DICOM localizados correctamente para todas las imágenes")
        print(f"  • Dataset listo para conversión a formato PNG estandarizado")
        
    else:
        print("\nNo se encontraron mamas de alto riesgo con mapeo válido.")
        print("Verificar los criterios de filtrado y la integridad del mapeo.")

# ============================================================================
# EJECUCIÓN DEL SCRIPT
# ============================================================================
if __name__ == "__main__":
    # Ejecutar proceso principal
    main()
    
    print("\n" + "="*80)
    print("PROCESO DE MAPEO COMPLETADO")
    print("="*80)
