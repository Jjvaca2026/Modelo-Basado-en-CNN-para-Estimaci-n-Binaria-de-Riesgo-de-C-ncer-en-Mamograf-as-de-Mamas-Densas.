"""
Autor: Jhon Jaime Vaca Hincapié
Maestría en Ingeniería
Fundación Universitaria Los Libertadores
2025

Script: exploracion_metadatos_vindr_mammo.py
Descripción: Análisis exploratorio de los archivos CSV del dataset VINDr-Mammo.
             Este script examina la estructura, contenido y relaciones entre los
             archivos de metadatos para entender la organización del dataset
             y planificar la estrategia de filtrado de imágenes.
"""

import pandas as pd
import numpy as np
import os
from IPython.display import display, HTML

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================
# Configurar pandas para mostrar toda la información sin truncamiento
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# ============================================================================
# FUNCIÓN PRINCIPAL DE EXPLORACIÓN
# ============================================================================
def explorar_archivo_csv(nombre_archivo, descripcion=""):
    """
    Realiza análisis exploratorio de un archivo CSV.
    
    Parámetros:
    -----------
    nombre_archivo : str
        Nombre del archivo CSV a analizar
    descripcion : str
        Descripción del archivo para el reporte
    
    Retorna:
    --------
    pandas.DataFrame
        DataFrame cargado o None si hay error
    """
    print(f"\n{'='*80}")
    print(f"ANÁLISIS DE {nombre_archivo}")
    print(f"{'='*80}")
    
    if descripcion:
        print(f"Descripción: {descripcion}")
    
    try:
        # Cargar archivo CSV
        df = pd.read_csv(nombre_archivo)
        print(f"Archivo cargado exitosamente.")
        print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
        
        # Información básica del DataFrame
        print("\nINFORMACIÓN DE COLUMNAS:")
        print(df.info())
        
        # Mostrar primeras filas
        print("\nPRIMERAS 5 FILAS DEL ARCHIVO:")
        display(df.head())
        
        # Listar todas las columnas con numeración
        print("\nCOLUMNAS DISPONIBLES:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Análisis de valores únicos por columna
        print("\nANÁLISIS DE VALORES ÚNICOS POR COLUMNA:")
        for col in df.columns:
            unique_vals = df[col].nunique()
            if unique_vals <= 20:
                print(f"\n{col} ({unique_vals} valores únicos):")
                print(f"  Valores: {df[col].unique()}")
            else:
                print(f"  {col}: {unique_vals} valores únicos")
        
        # Estadísticas descriptivas para columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\nESTADÍSTICAS DESCRIPTIVAS (COLUMNAS NUMÉRICAS):")
            display(df[numeric_cols].describe())
        
        # Estadísticas para columnas categóricas
        categorical_cols = df.select_dtypes(include=[object]).columns
        if len(categorical_cols) > 0:
            print("\nESTADÍSTICAS DESCRIPTIVAS (COLUMNAS CATEGÓRICAS):")
            display(df[categorical_cols].describe())
        
        return df
        
    except Exception as e:
        print(f"ERROR: No se pudo cargar el archivo {nombre_archivo}")
        print(f"Detalle del error: {e}")
        return None

def analizar_distribuciones(df, archivo_nombre):
    """
    Analiza distribuciones específicas relevantes para mamografías.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame a analizar
    archivo_nombre : str
        Nombre del archivo para referencia en mensajes
    """
    print(f"\nANÁLISIS DE DISTRIBUCIONES ESPECÍFICAS ({archivo_nombre}):")
    
    # Buscar columna de BI-RADS (insensible a mayúsculas/minúsculas)
    birads_cols = [col for col in df.columns if 'birads' in col.lower()]
    if birads_cols:
        print(f"\nDISTRIBUCIÓN DE CATEGORÍAS BI-RADS ({birads_cols[0]}):")
        distribucion = df[birads_cols[0]].value_counts().sort_index()
        display(distribucion)
        
        # Calcular porcentajes
        total = distribucion.sum()
        print("PORCENTAJES:")
        for categoria, count in distribucion.items():
            porcentaje = (count / total) * 100
            print(f"  {categoria}: {count} ({porcentaje:.1f}%)")
    
    # Buscar columna de densidad mamaria
    density_cols = [col for col in df.columns if 'density' in col.lower()]
    if density_cols:
        print(f"\nDISTRIBUCIÓN DE DENSIDAD MAMARIA ({density_cols[0]}):")
        distribucion = df[density_cols[0]].value_counts()
        display(distribucion)
    
    # Buscar columna de lateralidad
    laterality_cols = [col for col in df.columns 
                       if any(term in col.lower() 
                              for term in ['laterality', 'lado', 'side'])]
    if laterality_cols:
        print(f"\nDISTRIBUCIÓN DE LATERALIDAD ({laterality_cols[0]}):")
        distribucion = df[laterality_cols[0]].value_counts()
        display(distribucion)
    
    # Verificar duplicados
    duplicates = df.duplicated().sum()
    print(f"\nFILAS DUPLICADAS: {duplicates}")
    
    # Análisis de identificadores únicos
    for id_type in ['patient', 'study', 'image']:
        id_cols = [col for col in df.columns if id_type in col.lower()]
        if id_cols:
            unique_count = df[id_cols[0]].nunique()
            print(f"NÚMERO ÚNICO DE {id_type.upper()}S: {unique_count}")

def comparar_archivos(df1, df2, nombre1, nombre2):
    """
    Compara dos DataFrames para identificar relaciones y diferencias.
    
    Parámetros:
    -----------
    df1, df2 : pandas.DataFrame
        DataFrames a comparar
    nombre1, nombre2 : str
        Nombres de los archivos para referencia
    """
    print(f"\n{'='*80}")
    print(f"COMPARACIÓN ENTRE {nombre1} Y {nombre2}")
    print(f"{'='*80}")
    
    if df1 is None or df2 is None:
        print("ERROR: No se pueden comparar archivos. Uno o ambos no se cargaron correctamente.")
        return
    
    # Columnas comunes
    common_cols = set(df1.columns) & set(df2.columns)
    print(f"\nCOLUMNAS COMUNES: {len(common_cols)}")
    if common_cols:
        print("  " + ", ".join(sorted(common_cols)))
    
    # Buscar posibles claves para cruce
    posibles_claves = ['patient_id', 'study_id', 'image_id']
    for clave in posibles_claves:
        if clave in df1.columns and clave in df2.columns:
            print(f"\nANÁLISIS DE CLAVE POTENCIAL: {clave}")
            print(f"  Valores únicos en {nombre1}: {df1[clave].nunique()}")
            print(f"  Valores únicos en {nombre2}: {df2[clave].nunique()}")
            
            # Verificar solapamiento
            valores_df1 = set(df1[clave].unique())
            valores_df2 = set(df2[clave].unique())
            solapamiento = valores_df1.intersection(valores_df2)
            
            print(f"  Valores en común: {len(solapamiento)}")
            if len(solapamiento) > 0:
                print(f"  Ejemplo de valor común: {list(solapamiento)[0]}")
    
    # Comparar distribuciones de BI-RADS si existen en ambos
    birads_col1 = next((col for col in df1.columns if 'birads' in col.lower()), None)
    birads_col2 = next((col for col in df2.columns if 'birads' in col.lower()), None)
    
    if birads_col1 and birads_col2:
        print(f"\nCOMPARACIÓN DE DISTRIBUCIONES BI-RADS:")
        valores_df1 = set(df1[birads_col1].dropna().unique())
        valores_df2 = set(df2[birads_col2].dropna().unique())
        
        comunes = valores_df1.intersection(valores_df2)
        solo_en_df1 = valores_df1 - valores_df2
        solo_en_df2 = valores_df2 - valores_df1
        
        print(f"  Valores BI-RADS en {nombre1}: {sorted(valores_df1)}")
        print(f"  Valores BI-RADS en {nombre2}: {sorted(valores_df2)}")
        print(f"  Valores comunes: {sorted(comunes)}")
        
        if solo_en_df1:
            print(f"  Solo en {nombre1}: {sorted(solo_en_df1)}")
        if solo_en_df2:
            print(f"  Solo en {nombre2}: {sorted(solo_en_df2)}")

def generar_recomendaciones_filtrado(df_breast, df_finding):
    """
    Genera recomendaciones para la estrategia de filtrado basadas en el análisis.
    
    Parámetros:
    -----------
    df_breast, df_finding : pandas.DataFrame
        DataFrames analizados
    """
    print(f"\n{'='*80}")
    print("RECOMENDACIONES PARA ESTRATEGIA DE FILTRADO")
    print(f"{'='*80}")
    
    print("\nPREGUNTAS CLAVE IDENTIFICADAS:")
    print("1. ¿Cómo están organizados los datos? ¿Una fila por imagen o por hallazgo?")
    print("2. ¿Qué archivo contiene la información primaria para clasificación?")
    print("3. ¿Cómo se relacionan los hallazgos específicos con las imágenes completas?")
    print("4. ¿Existe información completa de densidad mamaria y BI-RADS por imagen?")
    print("5. ¿Cómo manejar casos con múltiples hallazgos en una mama?")
    
    print("\nOBSERVACIONES DEL ANÁLISIS:")
    
    if df_breast is not None:
        print(f"- breast-level_annotations.csv contiene {len(df_breast)} filas")
        if 'breast_birads' in df_breast.columns:
            print(f"- Incluye clasificación BI-RADS por mama/imagen")
        if 'breast_density' in df_breast.columns:
            print(f"- Incluye densidad mamaria por mama/imagen")
    
    if df_finding is not None:
        print(f"- finding_annotations.csv contiene {len(df_finding)} filas")
        if 'finding_categories' in df_finding.columns:
            print(f"- Incluye categorías específicas de hallazgos")
    
    print("\nETAPAS RECOMENDADAS PARA FILTRADO:")
    print("1. Identificar la tabla principal (probablemente breast-level_annotations)")
    print("2. Filtrar por criterios clínicos: densidad mamaria C/D y BI-RADS específico")
    print("3. Cruzar con finding_annotations para obtener detalles de hallazgos si es necesario")
    print("4. Verificar correspondencia exacta entre metadatos y archivos DICOM físicos")
    print("5. Extraer solo las imágenes correspondientes a cada mama específica")
    
    print("\nPOSIBLES DESAFÍOS IDENTIFICADOS:")
    print("- Los nombres de archivo DICOM pueden no coincidir con los IDs en los CSV")
    print("- Un estudio puede incluir imágenes de ambas mamas, pero solo una con patología")
    print("- Puede haber múltiples hallazgos anotados para una misma imagen")
    print("- La información puede estar distribuida en múltiples tablas relacionadas")

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================
def main():
    """
    Función principal que coordina el análisis exploratorio.
    """
    print("ANÁLISIS EXPLORATORIO DE METADATOS VINDr-MAMMO")
    print("Dataset: VINDr-Mammo - Vietnamese Mammography Dataset")
    print("="*80)
    
    # Listar archivos en el directorio actual
    print("\nARCHIVOS EN EL DIRECTORIO ACTUAL:")
    archivos = [f for f in os.listdir('.') if os.path.isfile(f)]
    for archivo in archivos:
        print(f"  - {archivo}")
    
    # 1. Analizar breast-level_annotations.csv
    df_breast = explorar_archivo_csv(
        'breast-level_annotations.csv',
        'Anotaciones a nivel de mama: contiene información por mama/imagen'
    )
    
    if df_breast is not None:
        analizar_distribuciones(df_breast, 'breast-level_annotations.csv')
    
    # 2. Analizar finding_annotations.csv
    df_finding = explorar_archivo_csv(
        'finding_annotations.csv',
        'Anotaciones de hallazgos: contiene información específica de lesiones'
    )
    
    if df_finding is not None:
        analizar_distribuciones(df_finding, 'finding_annotations.csv')
    
    # 3. Comparar ambos archivos
    comparar_archivos(df_breast, df_finding, 
                     'breast-level_annotations.csv', 
                     'finding_annotations.csv')
    
    # 4. Generar recomendaciones
    generar_recomendaciones_filtrado(df_breast, df_finding)
    
    print(f"\n{'='*80}")
    print("ANÁLISIS EXPLORATORIO COMPLETADO")
    print(f"{'='*80}")

# ============================================================================
# EJECUTAR ANÁLISIS
# ============================================================================
if __name__ == "__main__":
    main()
