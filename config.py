import os

class Config:
    # App config
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Email config
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'your-email@gmail.com'        # Replace with your Gmail
    MAIL_PASSWORD = 'your-app-password'           # Use Gmail App Password
    MAIL_DEFAULT_SENDER = 'your-email@gmail.com'  # Same as above (recommended)
