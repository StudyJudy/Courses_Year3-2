from flask import Flask
import os


def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    with app.app_context():
        from . import views, models
        models.db.init_app(app)
        models.db.create_all()
        # print(dir(views))
        return app
