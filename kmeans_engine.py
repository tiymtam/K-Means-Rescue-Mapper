import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

FITUR_COLS = ['penyelamatan_hewan', 'penyelamatan_air', 'pertolongan_medis', 'keruntuhan', 'kecelakaan_transportasi']
FITUR_LABEL = ['Penyelamatan Hewan', 'Penyelamatan di Air', 'Pertolongan Pertama Medis', 'Keruntuhan', 'Kecelakaan Transportasi']


def run_clustering(data_list, k=3):
    """
    data_list: list of dicts with keys id, nama, + fitur columns
    Returns dict with full clustering results.
    """
    df = pd.DataFrame(data_list)
    X = df[FITUR_COLS].values.astype(float)

    # --- Preprocessing: StandardScaler ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Elbow Method (K=1..10) ---
    elbow_data = []
    sil_data = []
    for ki in range(1, 11):
        km_tmp = KMeans(n_clusters=ki, init='k-means++', random_state=42, n_init=10)
        km_tmp.fit(X_scaled)
        elbow_data.append({'k': ki, 'inertia': round(float(km_tmp.inertia_), 4)})
        if ki >= 2:
            sl = silhouette_score(X_scaled, km_tmp.labels_)
            sil_data.append({'k': ki, 'silhouette': round(float(sl), 4), 'selected': (ki == k)})

    # --- K-Means with chosen K ---
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42, algorithm='lloyd')
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    inertia_final = float(kmeans.inertia_)
    n_iter = int(kmeans.n_iter_)

    # --- Silhouette Score ---
    sil_score = float(silhouette_score(X_scaled, labels))

    # --- Centroid iterations (simulate up to 5 iter) ---
    max_show = min(n_iter, 5)
    centroid_iters = []
    for it in range(1, max_show + 1):
        km_it = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=1, max_iter=it)
        km_it.fit(X_scaled)
        row = {'iterasi': it, 'clusters': []}
        for ci, c in enumerate(km_it.cluster_centers_):
            row['clusters'].append({
                'cluster': ci,
                'values': [round(float(v), 4) for v in c]
            })
        centroid_iters.append(row)

    # --- Centroid in original scale ---
    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)

    # --- Determine intensity label by ranking total mean ---
    df['cluster'] = labels
    total_per_cluster = df.groupby('cluster')[FITUR_COLS].mean().sum(axis=1)
    ranked = total_per_cluster.rank(ascending=True).astype(int)
    intensity_map_raw = {}
    for c_id, rank in ranked.items():
        if rank == 1:
            intensity_map_raw[c_id] = 'Rendah'
        elif rank == 2:
            intensity_map_raw[c_id] = 'Sedang'
        else:
            intensity_map_raw[c_id] = 'Tinggi'

    df['intensitas'] = df['cluster'].map(intensity_map_raw)

    # --- PCA 2D ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_x'] = X_pca[:, 0]
    df['pca_y'] = X_pca[:, 1]
    var_ratio = pca.explained_variance_ratio_

    # --- Mean features per cluster ---
    df_mean = df.groupby('cluster')[FITUR_COLS].mean().round(2)

    # --- Build results per kecamatan ---
    results = []
    for _, row in df.iterrows():
        results.append({
            'kecamatan_id': int(row['id']),
            'nama': row['nama'],
            'cluster_label': int(row['cluster']),
            'intensitas': row['intensitas'],
            'pca_x': round(float(row['pca_x']), 6),
            'pca_y': round(float(row['pca_y']), 6),
        })

    # --- Centroid summary for output ---
    centroid_summary = []
    for ci in range(k):
        c_orig = centroids_orig[ci]
        c_scaled = kmeans.cluster_centers_[ci]
        members = df[df['cluster'] == ci]['nama'].tolist()
        mean_vals = df_mean.loc[ci].to_dict() if ci in df_mean.index else {}
        centroid_summary.append({
            'cluster': ci,
            'intensitas': intensity_map_raw[ci],
            'members': members,
            'centroid_orig': {FITUR_LABEL[j]: round(float(c_orig[j]), 4) for j in range(len(FITUR_COLS))},
            'centroid_scaled': {FITUR_LABEL[j]: round(float(c_scaled[j]), 4) for j in range(len(FITUR_COLS))},
            'mean_orig': {FITUR_LABEL[j]: round(float(mean_vals.get(FITUR_COLS[j], 0)), 2) for j in range(len(FITUR_COLS))},
        })

    return {
        'k': k,
        'n_iter': n_iter,
        'inertia': round(inertia_final, 4),
        'silhouette_score': round(sil_score, 4),
        'elbow_data': elbow_data,
        'silhouette_compare': sil_data,
        'centroid_iterations': centroid_iters,
        'centroid_summary': centroid_summary,
        'results': results,
        'pca_variance': [round(float(v) * 100, 2) for v in var_ratio],
        'fitur_labels': FITUR_LABEL,
        'fitur_cols': FITUR_COLS,
    }
