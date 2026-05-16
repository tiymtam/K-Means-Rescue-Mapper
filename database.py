from models import db, Kecamatan, ClusterResult, ClusterConfig
from kmeans_engine import run_clustering

SEED_DATA = [
    ('Mijen',             308, 1, 6,   0, 2),
    ('Gunung Pati',       208, 0, 14,  0, 1),
    ('Banyumanik',        305, 1, 9,   0, 2),
    ('Gajah Mungkur',     137, 0, 7,   0, 0),
    ('Semarang Selatan',  149, 1, 16,  0, 0),
    ('Candisari',         162, 1, 9,   0, 0),
    ('Tembalang',         283, 0, 5,   0, 1),
    ('Pedurungan',        242, 0, 16,  0, 1),
    ('Genuk',             183, 4, 4,   16, 1),
    ('Gayamsari',         124, 1, 99,  0, 0),
    ('Semarang Timur',     84, 0, 12,  0, 0),
    ('Semarang Utara',    103, 1, 13,  0, 2),
    ('Semarang Tengah',    72, 0, 6,   0, 0),
    ('Semarang Barat',    212, 0, 254, 0, 1),
    ('Tugu',              184, 0, 8,   0, 0),
    ('Ngaliyan',          264, 0, 12,  0, 1),
]


def seed_database():
    if Kecamatan.query.count() > 0:
        return  # already seeded

    # Insert kecamatan
    for row in SEED_DATA:
        k = Kecamatan(
            nama=row[0],
            penyelamatan_hewan=row[1],
            penyelamatan_air=row[2],
            pertolongan_medis=row[3],
            keruntuhan=row[4],
            kecelakaan_transportasi=row[5],
        )
        db.session.add(k)
    db.session.commit()

    # Run clustering and store results
    recalculate_clustering(k=3)


def recalculate_clustering(k=3):
    kecamatans = Kecamatan.query.all()
    data_list = [kc.to_dict() for kc in kecamatans]

    result = run_clustering(data_list, k=k)

    # Clear old results
    ClusterResult.query.delete()
    ClusterConfig.query.delete()
    db.session.commit()

    # Store config for each K evaluated
    for item in result['silhouette_compare']:
        cfg = ClusterConfig(
            k_value=item['k'],
            silhouette_score=item['silhouette'],
            inertia=next((e['inertia'] for e in result['elbow_data'] if e['k'] == item['k']), None),
            is_selected=(item['k'] == k),
        )
        db.session.add(cfg)

    # Store selected K config (K=1 has no silhouette)
    # Also add K=1 entry from elbow only
    cfg1 = ClusterConfig(
        k_value=1,
        silhouette_score=None,
        inertia=result['elbow_data'][0]['inertia'],
        is_selected=False,
    )
    db.session.add(cfg1)

    # Store cluster results
    for r in result['results']:
        cr = ClusterResult(
            kecamatan_id=r['kecamatan_id'],
            cluster_label=r['cluster_label'],
            intensitas=r['intensitas'],
            pca_x=r['pca_x'],
            pca_y=r['pca_y'],
        )
        db.session.add(cr)

    db.session.commit()
    return result
