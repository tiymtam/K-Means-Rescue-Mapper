from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from models import db, Kecamatan, ClusterResult, ClusterConfig
from database import seed_database, recalculate_clustering
from kmeans_engine import run_clustering, FITUR_LABEL, FITUR_COLS
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///kmeans_semarang.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'semarang-kmeans-2025'

db.init_app(app)

with app.app_context():
    db.create_all()
    seed_database()


# ─────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────
@app.route('/')
@app.route('/dashboard')
def dashboard():
    total_kecamatan = Kecamatan.query.count()
    # Get selected config
    cfg = ClusterConfig.query.filter_by(is_selected=True).first()
    k_value = cfg.k_value if cfg else 3
    sil_score = round(cfg.silhouette_score, 4) if cfg and cfg.silhouette_score else '-'

    # Total kejadian
    kecamatans = Kecamatan.query.all()
    total_kejadian = sum(
        k.penyelamatan_hewan + k.penyelamatan_air + k.pertolongan_medis +
        k.keruntuhan + k.kecelakaan_transportasi for k in kecamatans
    )

    # Chart: total per kecamatan
    chart_labels = [k.nama for k in kecamatans]
    chart_totals = [
        k.penyelamatan_hewan + k.penyelamatan_air + k.pertolongan_medis +
        k.keruntuhan + k.kecelakaan_transportasi for k in kecamatans
    ]

    # Cluster distribution
    results = db.session.query(ClusterResult, Kecamatan).join(
        Kecamatan, ClusterResult.kecamatan_id == Kecamatan.id).all()

    from collections import Counter
    intensitas_count = Counter(r.ClusterResult.intensitas for r in results)
    pie_labels = list(intensitas_count.keys())
    pie_values = list(intensitas_count.values())

    # Cluster summary table
    cluster_summary = {}
    for r in results:
        label = r.ClusterResult.cluster_label
        intensitas = r.ClusterResult.intensitas
        if label not in cluster_summary:
            cluster_summary[label] = {'intensitas': intensitas, 'count': 0, 'members': []}
        cluster_summary[label]['count'] += 1
        cluster_summary[label]['members'].append(r.Kecamatan.nama)

    return render_template('dashboard.html',
        total_kecamatan=total_kecamatan,
        k_value=k_value,
        sil_score=sil_score,
        total_kejadian=total_kejadian,
        chart_labels=chart_labels,
        chart_totals=chart_totals,
        pie_labels=pie_labels,
        pie_values=pie_values,
        cluster_summary=cluster_summary,
    )


# ─────────────────────────────────────────────
#  DATA KECAMATAN (CRUD)
# ─────────────────────────────────────────────
@app.route('/kecamatan')
def kecamatan_list():
    kecamatans = Kecamatan.query.order_by(Kecamatan.nama).all()
    return render_template('kecamatan.html', kecamatans=kecamatans)


@app.route('/kecamatan/tambah', methods=['POST'])
def kecamatan_tambah():
    try:
        k = Kecamatan(
            nama=request.form['nama'],
            penyelamatan_hewan=int(request.form.get('penyelamatan_hewan', 0)),
            penyelamatan_air=int(request.form.get('penyelamatan_air', 0)),
            pertolongan_medis=int(request.form.get('pertolongan_medis', 0)),
            keruntuhan=int(request.form.get('keruntuhan', 0)),
            kecelakaan_transportasi=int(request.form.get('kecelakaan_transportasi', 0)),
        )
        db.session.add(k)
        db.session.commit()
        flash('Data kecamatan berhasil ditambahkan!', 'success')
    except Exception as e:
        flash(f'Gagal menambahkan data: {str(e)}', 'danger')
    return redirect(url_for('kecamatan_list'))


@app.route('/kecamatan/edit/<int:id>', methods=['POST'])
def kecamatan_edit(id):
    k = Kecamatan.query.get_or_404(id)
    try:
        k.nama = request.form['nama']
        k.penyelamatan_hewan = int(request.form.get('penyelamatan_hewan', 0))
        k.penyelamatan_air = int(request.form.get('penyelamatan_air', 0))
        k.pertolongan_medis = int(request.form.get('pertolongan_medis', 0))
        k.keruntuhan = int(request.form.get('keruntuhan', 0))
        k.kecelakaan_transportasi = int(request.form.get('kecelakaan_transportasi', 0))
        db.session.commit()
        flash('Data berhasil diperbarui!', 'success')
    except Exception as e:
        flash(f'Gagal memperbarui data: {str(e)}', 'danger')
    return redirect(url_for('kecamatan_list'))


@app.route('/kecamatan/hapus/<int:id>', methods=['POST'])
def kecamatan_hapus(id):
    k = Kecamatan.query.get_or_404(id)
    try:
        db.session.delete(k)
        db.session.commit()
        flash('Data kecamatan berhasil dihapus!', 'success')
    except Exception as e:
        flash(f'Gagal menghapus: {str(e)}', 'danger')
    return redirect(url_for('kecamatan_list'))


# ─────────────────────────────────────────────
#  PROSES CLUSTERING
# ─────────────────────────────────────────────
@app.route('/clustering')
def clustering_page():
    elbow_data = []
    sil_data = []
    configs = ClusterConfig.query.order_by(ClusterConfig.k_value).all()
    for cfg in configs:
        if cfg.inertia is not None:
            elbow_data.append({'k': cfg.k_value, 'inertia': cfg.inertia})
        if cfg.silhouette_score is not None:
            sil_data.append({'k': cfg.k_value, 'silhouette': cfg.silhouette_score, 'selected': cfg.is_selected})

    selected_cfg = ClusterConfig.query.filter_by(is_selected=True).first()
    return render_template('clustering.html',
        elbow_data=elbow_data,
        sil_data=sil_data,
        selected_cfg=selected_cfg,
        fitur_labels=['Hewan', 'Air', 'Medis', 'Keruntuhan', 'Transport'],
    )


@app.route('/api/run-clustering', methods=['POST'])
def api_run_clustering():
    try:
        body = request.get_json()
        k = int(body.get('k', 3))
        if k < 2 or k > 10:
            return jsonify({'error': 'K harus antara 2 dan 10'}), 400

        kecamatans = Kecamatan.query.all()
        if len(kecamatans) < k:
            return jsonify({'error': 'Jumlah data kurang dari nilai K'}), 400

        result = recalculate_clustering(k=k)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
#  HASIL CLUSTERING
# ─────────────────────────────────────────────
@app.route('/hasil')
def hasil_page():
    results = db.session.query(ClusterResult, Kecamatan).join(
        Kecamatan, ClusterResult.kecamatan_id == Kecamatan.id
    ).order_by(ClusterResult.cluster_label, Kecamatan.nama).all()

    cfg = ClusterConfig.query.filter_by(is_selected=True).first()

    # Build data for scatter plot
    scatter_data = []
    for r in results:
        scatter_data.append({
            'nama': r.Kecamatan.nama,
            'cluster': r.ClusterResult.cluster_label,
            'intensitas': r.ClusterResult.intensitas,
            'pca_x': r.ClusterResult.pca_x,
            'pca_y': r.ClusterResult.pca_y,
        })

    # Centroid & mean from latest clustering
    kecamatans = Kecamatan.query.all()
    data_list = [k.to_dict() for k in kecamatans]
    k_val = cfg.k_value if cfg else 3
    engine_result = run_clustering(data_list, k=k_val)

    return render_template('hasil.html',
        results=results,
        cfg=cfg,
        scatter_data=scatter_data,
        centroid_summary=engine_result['centroid_summary'],
        pca_variance=engine_result['pca_variance'],
        fitur_labels=FITUR_LABEL,
        centroid_iters=engine_result['centroid_iterations'],
    )


# ─────────────────────────────────────────────
#  PEMETAAN
# ─────────────────────────────────────────────
@app.route('/peta')
def peta_page():
    results = db.session.query(ClusterResult, Kecamatan).join(
        Kecamatan, ClusterResult.kecamatan_id == Kecamatan.id
    ).all()

    map_data = []
    for r in results:
        kec = r.Kecamatan
        cr = r.ClusterResult
        map_data.append({
            'nama': kec.nama,
            'cluster': cr.cluster_label,
            'intensitas': cr.intensitas,
            'penyelamatan_hewan': kec.penyelamatan_hewan,
            'penyelamatan_air': kec.penyelamatan_air,
            'pertolongan_medis': kec.pertolongan_medis,
            'keruntuhan': kec.keruntuhan,
            'kecelakaan_transportasi': kec.kecelakaan_transportasi,
        })

    return render_template('peta.html', map_data=map_data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
