from datetime import datetime
from flask import current_app as app
from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()

class User(db.Model):
    uid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    uname = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<User {self.uid} {self.uname}>'


class File(db.Model):
    fid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    fname = db.Column(db.String(255), nullable=False)
    ftime = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    uid = db.Column(db.Integer, db.ForeignKey('user.uid'), nullable=False)
    user = db.relationship('User', backref=db.backref('files', lazy=True))

    def __repr__(self):
        return f'<File {self.fid} {self.fname} {self.ftime}>'

    def save_to_disk(self, file_content):
        file_path = os.path.join('uploads', f'{self.fid}_{self.fname}')
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            f.write(file_content)
        return file_path


class Log(db.Model):
    logid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    logtime = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    logcontent = db.Column(db.Text, nullable=False)
    logtype = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<Log {self.logid} {self.logtype} {self.logtime}>'