"""
Script: Análisis Exploratorio y Unificación de Metadatos de Datasets de Mamografía
Autor: Jhon Jaime Vaca Hincapié
Institución: Fundación Universitaria Los Libertadores
Programa: Maestría en Ingeniería
Año: 2025

Descripción:
Este script unifica los metadatos clínicos provenientes de tres datasets públicos
de mamografía (CBIS-DDSM, VINDr e INbreast) y realiza un análisis exploratorio
descriptivo del conjunto combinado.

El objetivo es caracterizar la distribución de variables clínicas relevantes
como categoría BI-RADS, densidad mamaria, lateralidad y vistas, así como
identificar posibles sesgos por fuente de datos. Los resultados se utilizan
para justificar experimentalmente el uso del dataset unificado en los modelos
de clasificación desarrollados en la tesis.

Entradas:
- Archivos CSV de metadatos filtrados por dataset (CBIS, VINDr, INbreast).
  Las rutas pueden apuntar a Google Drive o a archivos locales.

Salidas:
- Archivo CSV unificado con todos los registros.
- Conjunto de gráficos estadísticos listos para inclusión en la tesis.
"""

# ============================================================================
# LIBRERÍAS REQUERIDAS
# ============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def add_value_labels(ax, spacing=5, fontsize=10):
    """
    Agrega etiquetas numéricas sobre las barras de un gráfico de barras.

    Parámetros:
    - ax: objeto Axes de matplotlib
    - spacing: desplazamiento vertical del texto
    - fontsize: tamaño de fuente del texto
    """
    for rect in ax.patches:
        height = rect.get_height()
        if height > 0:
            ax.annotate(
                f'{int(height)}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, spacing),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=fontsize
            )


# ============================================================================
# CARGA DE METADATOS DESDE DRIVE O ARCHIVOS LOCALES
# ============================================================================

archivos_drive = {
    "CBIS": "https://drive.google.com/uc?id=1SEo3k-KoH-4K3V7BWDlKoeunm1mIibz7",
    "VINDr": "https://drive.google.com/uc?id=1gLoNbqzW8nijwOJHJN-UYSLTLTdPqa-M",
    "INbreast": "https://drive.google.com/uc?id=1wKQfbU3OZZloOtjz3ecExAVc17DtsUbD"
}

datasets = {}

for nombre, url in archivos_drive.items():
    nombre_local = f"metadata_filtrado_{nombre}.csv"

    if not os.path.exists(nombre_local):
        print(f"Descargando metadatos de {nombre}...")
        df = pd.read_csv(url)
        df.to_csv(nombre_local, index=False)
        datasets[nombre] = df
    else:
        print(f"Cargando archivo local: {nombre_local}")
        datasets[nombre] = pd.read_csv(nombre_local)

    print(f"{nombre}: {len(datasets[nombre])} registros cargados")


# ============================================================================
# UNIFICACIÓN DE DATASETS
# ============================================================================

df_total = pd.concat(datasets.values(), ignore_index=True)

nombre_salida = "metadata_unificada_CBIS_VINDr_INbreast.csv"
df_total.to_csv(nombre_salida, index=False)

print("\nResumen del dataset unificado:")
print(f"Total de registros: {len(df_total)}")
print(f"Total de columnas: {len(df_total.columns)}")
print(f"Columnas: {list(df_total.columns)}")


# ============================================================================
# ANÁLISIS DESCRIPTIVO EN CONSOLA
# ============================================================================

print("\nDistribución por fuente de datos:")
print(df_total["fuente_datos"].value_counts())

print("\nDistribución de categorías BI-RADS:")
print(df_total["categoria_birads"].value_counts().sort_index())

print("\nDistribución de densidad mamaria:")
print(df_total["densidad_mamaria"].value_counts().sort_index())

if "lateralidad" in df_total.columns:
    print("\nDistribución por lateralidad:")
    print(df_total["lateralidad"].value_counts())

if "vista" in df_total.columns:
    print("\nDistribución por vista:")
    print(df_total["vista"].value_counts())


# ============================================================================
# GENERACIÓN DE GRÁFICOS
# ============================================================================

archivos_generados = [nombre_salida]

# Gráfico 1: Distribución por fuente de datos
plt.figure(figsize=(10, 6))
ax1 = df_total["fuente_datos"].value_counts().plot(kind="bar")
plt.title("Distribución por Fuente de Datos")
plt.xlabel("Fuente")
plt.ylabel("Cantidad de registros")
plt.grid(axis='y', linestyle='--', alpha=0.6)
add_value_labels(ax1)

nombre_img1 = "1_distribucion_fuente_datos.png"
plt.tight_layout()
plt.savefig(nombre_img1, dpi=300)
plt.show()
archivos_generados.append(nombre_img1)


# Gráfico 2: Distribución de densidad mamaria
plt.figure(figsize=(10, 6))
ax2 = df_total["densidad_mamaria"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribución de Densidad Mamaria")
plt.xlabel("Densidad")
plt.ylabel("Cantidad")
plt.grid(axis='y', linestyle='--', alpha=0.6)
add_value_labels(ax2)

nombre_img2 = "2_distribucion_densidad_mamaria.png"
plt.tight_layout()
plt.savefig(nombre_img2, dpi=300)
plt.show()
archivos_generados.append(nombre_img2)


# Gráfico 3: Distribución de categorías BI-RADS
plt.figure(figsize=(10, 6))
ax3 = df_total["categoria_birads"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribución de Categorías BI-RADS")
plt.xlabel("BI-RADS")
plt.ylabel("Cantidad")
plt.grid(axis='y', linestyle='--', alpha=0.6)
add_value_labels(ax3)

nombre_img3 = "3_distribucion_birads.png"
plt.tight_layout()
plt.savefig(nombre_img3, dpi=300)
plt.show()
archivos_generados.append(nombre_img3)


# Gráfico 4: Fuente vs BI-RADS (barras apiladas)
ct_fuente_birads = pd.crosstab(df_total["fuente_datos"], df_total["categoria_birads"])
ax4 = ct_fuente_birads.plot(kind="bar", stacked=True, figsize=(12, 7))
plt.title("Comparativa por Fuente de Datos y Categoría BI-RADS")
plt.xlabel("Fuente de Datos")
plt.ylabel("Cantidad")
plt.grid(axis='y', linestyle='--', alpha=0.6)

for container in ax4.containers:
    ax4.bar_label(container, label_type='center', fontsize=8)

nombre_img4 = "4_comparativa_fuente_birads.png"
plt.tight_layout()
plt.savefig(nombre_img4, dpi=300)
plt.show()
archivos_generados.append(nombre_img4)


# Gráfico 5: Mapa de calor BI-RADS vs Densidad
plt.figure(figsize=(10, 8))
ct_birads_densidad = pd.crosstab(df_total["categoria_birads"], df_total["densidad_mamaria"])
sns.heatmap(ct_birads_densidad, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Mapa de Calor: BI-RADS vs Densidad Mamaria")
plt.xlabel("Densidad Mamaria")
plt.ylabel("Categoría BI-RADS")

nombre_img5 = "5_mapa_calor_birads_densidad.png"
plt.tight_layout()
plt.savefig(nombre_img5, dpi=300)
plt.show()
archivos_generados.append(nombre_img5)


# ============================================================================
# DESCARGA DE ARCHIVOS GENERADOS
# ============================================================================

for archivo in archivos_generados:
    if os.path.exists(archivo):
        files.download(archivo)

print("\nAnálisis exploratorio completado correctamente.")
