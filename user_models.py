from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()           #SQLAlchemy를 사용해 데이터베이스 저장

class Finderuser(db.Model):
    __tablename__ = 'finderuser'   #테이블 이름 : fcuser
    id = db.Column(db.Integer)
    password = db.Column(db.String(64))     #패스워드를 받아올 문자열길이
    userid = db.Column(db.String(32), primary_key = True)       #id를 프라이머리키로 설정
    username = db.Column(db.String(8))