import os

user = 'themis'
password = '123456'
host = 'localhost'
port = 3306
database = 'themis'

class Config:
    SECRET_KEY = os.urandom(24)
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEBUG = True
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    # 其他配置项
