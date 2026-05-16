from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Kecamatan(db.Model):
    __tablename__ = 'kecamatan'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    nama = db.Column(db.String(100), nullable=False)
    penyelamatan_hewan = db.Column(db.Integer, default=0)
    penyelamatan_air = db.Column(db.Integer, default=0)
    pertolongan_medis = db.Column(db.Integer, default=0)
    keruntuhan = db.Column(db.Integer, default=0)
    kecelakaan_transportasi = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    cluster_result = db.relationship('ClusterResult', backref='kecamatan', uselist=False, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'nama': self.nama,
            'penyelamatan_hewan': self.penyelamatan_hewan,
            'penyelamatan_air': self.penyelamatan_air,
            'pertolongan_medis': self.pertolongan_medis,
            'keruntuhan': self.keruntuhan,
            'kecelakaan_transportasi': self.kecelakaan_transportasi,
            'total': self.penyelamatan_hewan + self.penyelamatan_air + self.pertolongan_medis + self.keruntuhan + self.kecelakaan_transportasi,
        }


class ClusterResult(db.Model):
    __tablename__ = 'cluster_result'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    kecamatan_id = db.Column(db.Integer, db.ForeignKey('kecamatan.id'), nullable=False)
    cluster_label = db.Column(db.Integer)
    intensitas = db.Column(db.String(20))
    pca_x = db.Column(db.Float)
    pca_y = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'kecamatan_id': self.kecamatan_id,
            'cluster_label': self.cluster_label,
            'intensitas': self.intensitas,
            'pca_x': self.pca_x,
            'pca_y': self.pca_y,
        }


class ClusterConfig(db.Model):
    __tablename__ = 'cluster_config'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    k_value = db.Column(db.Integer)
    silhouette_score = db.Column(db.Float)
    inertia = db.Column(db.Float)
    is_selected = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'k_value': self.k_value,
            'silhouette_score': round(self.silhouette_score, 4) if self.silhouette_score else None,
            'inertia': round(self.inertia, 4) if self.inertia else None,
            'is_selected': self.is_selected,
        }
