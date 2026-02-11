# -*- coding: utf-8 -*-
"""
ANÁLISIS DE CLUSTERING MEDIANTE K-MEANS PARA SELECCIÓN DE TÉCNICAS DE PREPROCESAMIENTO

Este script realiza un análisis de clustering no supervisado utilizando el algoritmo
K-Means sobre las métricas de calidad de imagen (PSNR, SSIM y Entropía) obtenidas
durante la etapa de preprocesamiento de mamografías.

El objetivo es seleccionar de forma objetiva las técnicas de preprocesamiento más
adecuadas para el entrenamiento de modelos de aprendizaje profundo.

IMPORTANTE:
- Las rutas definidas en BASE_PATH y csv_path deben ser ajustadas por el usuario
  según la ubicación real de los archivos CSV y carpetas de resultados en su sistema
  o en Google Drive.
"""

# ============================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from google.colab import drive

# ============================================
# 2. MONTAJE DE GOOGLE DRIVE
# ============================================
drive.mount('/content/drive', force_remount=True)

# ============================================
# 3. DEFINICIÓN DE RUTAS
# ============================================
# Ruta base del proyecto (ajustar según la ubicación del usuario)
BASE_PATH = '/content/drive/My Drive/Tesis Maestría'

# Ruta del archivo CSV con las métricas finales de preprocesamiento
csv_path = os.path.join(BASE_PATH, 'metricas_preprocesamiento_FINAL.csv')

# ============================================
# 4. CARGA DE DATOS
# ============================================
print("Cargando datos para análisis de clustering...")
print("=" * 60)

df = pd.read_csv(csv_path)

print(f"Total de registros cargados: {len(df)}")
print(f"Técnicas evaluadas: {df['tecnica'].unique()}")
print(f"Clases clínicas: {df['clase'].unique()}")

# ============================================
# 5. PREPARACIÓN DE DATOS PARA CLUSTERING
# ============================================
print("\nPreparando datos para K-Means...")

# Selección de métricas
X = df[['psnr', 'ssim', 'entropia']].copy()

# Normalización de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# 6. DETERMINACIÓN DEL NÚMERO ÓPTIMO DE CLUSTERS
# ============================================
print("\nDeterminando número óptimo de clusters (método del codo)...")

inertias = []
K_range = range(1, 8)

for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Gráfica del método del codo
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para K-Means')
plt.grid(True)

plt.savefig(
    os.path.join(BASE_PATH, 'Figuras_Tesis', 'Metodo_Codo_KMeans.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# Selección de k óptimo (determinado empíricamente)
k_optimo = 3
print(f"Número óptimo de clusters seleccionado: k = {k_optimo}")

# ============================================
# 7. APLICACIÓN DE K-MEANS
# ============================================
print(f"\nAplicando K-Means con k = {k_optimo}...")

kmeans = KMeans(
    n_clusters=k_optimo,
    random_state=42,
    n_init=10,
    max_iter=300
)

df['cluster'] = kmeans.fit_predict(X_scaled)

# ============================================
# 8. ANÁLISIS DE DISTRIBUCIÓN POR CLUSTER
# ============================================
print("\nDistribución de técnicas por cluster:")

cluster_dist = (
    df.groupby(['tecnica', 'cluster'])
      .size()
      .unstack(fill_value=0)
)

print(cluster_dist)

# Gráfica de distribución
plt.figure(figsize=(12, 6))
cluster_dist.plot(kind='bar', stacked=True)
plt.title('Distribución de Técnicas por Cluster')
plt.xlabel('Técnica')
plt.ylabel('Número de imágenes')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(BASE_PATH, 'Figuras_Tesis', 'Distribucion_Clusters.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ============================================
# 9. CARACTERIZACIÓN DE CLUSTERS
# ============================================
print("\nCaracterización de cada cluster:")

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

for i, center in enumerate(cluster_centers):
    print("\n" + "=" * 40)
    print(f"CLUSTER {i}")
    print("=" * 40)
    print(f"PSNR promedio: {center[0]:.2f} dB")
    print(f"SSIM promedio: {center[1]:.4f}")
    print(f"Entropía promedio: {center[2]:.2f}")

    tecnicas_cluster = df[df['cluster'] == i]['tecnica'].value_counts()
    print("\nTécnicas presentes en el cluster:")
    for tecnica, count in tecnicas_cluster.items():
        total = len(df[df['tecnica'] == tecnica])
        porcentaje = (count / total) * 100
        print(f"  - {tecnica}: {porcentaje:.1f}%")

# ============================================
# 10. VISUALIZACIÓN MEDIANTE PCA
# ============================================
print("\nGenerando visualizaciones PCA...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(14, 10))

# PCA por técnica
plt.subplot(2, 2, 1)
for tecnica in df['tecnica'].unique():
    mask = df['tecnica'] == tecnica
    plt.scatter(df.loc[mask, 'PCA1'], df.loc[mask, 'PCA2'], label=tecnica, alpha=0.6)

plt.title('Distribución por Técnica (PCA)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.grid(True, alpha=0.3)

# PCA por cluster
plt.subplot(2, 2, 2)
scatter = plt.scatter(df['PCA1'], df['PCA2'], c=df['cluster'], cmap='viridis', alpha=0.7)
plt.title('Clusters K-Means (PCA)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# Boxplot de métricas
plt.subplot(2, 2, 3)
df_melted = df.melt(id_vars=['tecnica'], value_vars=['psnr', 'ssim', 'entropia'])
sns.boxplot(x='tecnica', y='value', hue='variable', data=df_melted)
plt.title('Distribución de Métricas por Técnica')
plt.xticks(rotation=45)

# Heatmap de correlación
plt.subplot(2, 2, 4)
corr_matrix = df[['psnr', 'ssim', 'entropia', 'cluster']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Matriz de Correlación')

plt.tight_layout()
plt.savefig(
    os.path.join(BASE_PATH, 'Figuras_Tesis', 'Analisis_Clustering_Completo.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ============================================
# 11. SELECCIÓN DE TÉCNICAS ÓPTIMAS
# ============================================
print("\nSelección de técnicas óptimas para entrenamiento:")

puntuaciones = []
for center in cluster_centers:
    score_psnr = center[0] / 50
    score_ssim = center[1]
    score_ent = 1 - abs(center[2] - 5) / 10
    puntuaciones.append(0.4 * score_psnr + 0.4 * score_ssim + 0.2 * score_ent)

cluster_optimo = int(np.argmax(puntuaciones))
print(f"Cluster óptimo identificado: {cluster_optimo}")

tecnicas_optimas = df[df['cluster'] == cluster_optimo]['tecnica'].value_counts()
top2 = tecnicas_optimas.head(2).index.tolist()

print("\nTécnicas seleccionadas para entrenamiento:")
for i, tecnica in enumerate(top2, 1):
    print(f"{i}. {tecnica}")

# ============================================
# 12. GUARDADO DE RESULTADOS
# ============================================
resultados_path = os.path.join(BASE_PATH, 'resultados_clustering_detallado.csv')
df.to_csv(resultados_path, index=False)

resumen_path = os.path.join(BASE_PATH, 'resumen_seleccion_tecnicas.txt')
with open(resumen_path, 'w') as f:
    f.write("RESULTADOS DE SELECCIÓN DE TÉCNICAS\n")
    f.write("=" * 60 + "\n")
    f.write(f"Cluster óptimo: {cluster_optimo}\n")
    f.write("Técnicas seleccionadas:\n")
    for tecnica in top2:
        f.write(f"- {tecnica}\n")

print("\nAnálisis de clustering completado correctamente.")
print(f"Resultados guardados en:\n- {resultados_path}\n- {resumen_path}")
