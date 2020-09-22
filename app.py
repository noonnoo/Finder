# app.py
import hashlib
import json

import cv2, os
import threading
from os.path import expanduser

from flask import Flask, render_template, url_for, request, redirect, session, flash, send_file

from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import os, sys
from datetime import datetime

from user_models import db, Finderuser
from forms import SigninForm, LoginForm, UploadForm

from face_test import find_face, call_by_app

app = Flask(__name__)

runnig_threads = []
#####################################################################################

#defining function to run on shutdown
def close_running_threads():
    for thread in runnig_threads:
        thread.join()
        runnig_threads.pop()

    print("Threads complete, ready to finish")

#face찾기 스레드 pic_upload에서 콜됨
def execute(name, file_path, destination):
  if not os.path.exists(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], destination, name)):
    os.makedirs(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], destination, name))
    find_face(file_path, os.path.join(app.config['UPLOADED_PHOTOS_DEST'], "person", "")
              , os.path.join(app.config['UPLOADED_PHOTOS_DEST'], destination, name))
  else:
    return

#GET은 페이지 나오게 요청, POST는 버튼 누르고 데이터 가져오는 요청
@app.route('/signin', methods=['GET', 'POST'])     #회원가입
def signin():
  form = SigninForm()
  if form.validate_on_submit():
    fuser = Finderuser()
    fuser.userid = form.data.get('userid')
    fuser.username = form.data.get('username')
    fuser.password = form.data.get('password')

    print(fuser.userid, fuser.password)
    db.session.add(fuser)
    db.session.commit()

    return render_template('login.html', form=form)  # 가입 완료하면 로그인화면으로 redirect
  else:
    return render_template('signin.html', form=form)


@app.route('/login', methods=['GET', 'POST'])     #로그인
def login():
  form = LoginForm()  # 로그인 폼 생성
  if form.validate_on_submit():  # 유효성 검사
    userid = form.data.get('userid')
    session['userid'] = userid  # userid session에 추가
    return redirect('/index')  # 로그인에 성공하면 홈화면으로 redirect

  return render_template('login.html', form=form)


@app.route('/logout', methods=['GET'])            #로그아웃
def logout():
    session.pop('userid',None)
    return redirect('/')

@app.route('/mypic', methods=['GET', 'POST'])     #개인사진 업로드
def mypic():
  if "file_urls" not in session:
    session['file_urls'] = []
  file_urls = session['file_urls']

  if 'userid' in session:
    user_id = session['userid']
    user_dest = os.path.join("person" , str(user_id))

  # handle image upload from Dropszone
  if request.method == 'POST':
    file_obj = request.files
    for f in file_obj:
      file = request.files.get(f)
      filename = photos.save(file
                             , folder= os.path.join(app.config['UPLOADED_PHOTOS_DEST'], user_dest)
                             , name=file.filename)
      print(photos.url(filename))
      file_urls.append(photos.url(filename))

    session['file_urls'] = file_urls
    return "uploading..."

  userid = session.get('userid', None)

  if 'userid' in session:
    return render_template('mypic.html', userid=userid)
  else:
    flash("로그인 후 사용해주세요.")
    return redirect(url_for('login'))


@app.route('/picupload', methods=['GET', 'POST'])   #단체사진 업로드
def picupload():
  form = UploadForm()
  userid = session.get('userid', None)

  if 'userid' in session:
    today = datetime.today().strftime("%Y%m%d")
    user_id = session['userid']
    destination = os.path.join("pictures", str(user_id) ,str(today))

  if form.validate_on_submit():
    folder_path = destination
    for file in request.files.getlist('photo'):
      name = hashlib.md5((file.filename).encode('utf-8')).hexdigest()[:15]

      if not os.path.exists(os.getcwd() + destination + "\\" + name):
        file_path = photos.save(file
                                , folder= os.path.join(app.config['UPLOADED_PHOTOS_DEST'], destination)
                                , name=name + '.')
        print(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], destination))
        face_thread = threading.Thread(
          target=execute, name=name, args=(name, file_path, destination))
        runnig_threads.append(face_thread)
        face_thread.start()

    success = True
    return render_template('picupload_result.html', userid=userid, success = success)

  else:
    success = False

  if 'userid' in session:
    return render_template('picupload.html', userid=userid, form=form)
  else:
    flash("로그인 후 사용해주세요.")
    return redirect(url_for('login'))


@app.route('/publicupload', methods=['GET', 'POST'])   #단체사진 업로드
def publicupload():
  form = UploadForm()
  userid = session.get('userid', None)

  if 'userid' in session:
    today = datetime.today().strftime("%Y%m%d")
    user_id = session['userid']
    destination = "public"

  if form.validate_on_submit():
    folder_path = destination
    for file in request.files.getlist('photo'):
      name = hashlib.md5((file.filename).encode('utf-8')).hexdigest()[:15]

      if not os.path.exists(os.getcwd() + destination + "\\" + name):
        file_path = photos.save(file
                                , folder= os.path.join(app.config['UPLOADED_PHOTOS_DEST'], destination)
                                , name=name + '.')
        print(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], destination))
        face_thread = threading.Thread(
          target=execute, name=name, args=(name, file_path, destination))
        runnig_threads.append(face_thread)
        face_thread.start()

    success = True
    return render_template('picupload_result.html', userid=userid, success = success)

  else:
    success = False

  if 'userid' in session:
    return render_template('publicupload.html', userid=userid, form=form)
  else:
    flash("로그인 후 사용해주세요.")
    return redirect(url_for('login'))

@app.route('/picupload_result')     #단체사진 업로드 결과 화면
def picupload_result():
  userid = session.get('userid', None)

  if 'userid' in session:
    return render_template('picupload_result.html', userid=userid, success=False)
  else:
    flash("로그인 후 사용해주세요.")
    return redirect(url_for('login'))

@app.route('/results')    #개인사진 업로드 결과 화면
def results():
  if "file_urls" not in session or session['file_urls'] == []:
    flash('파일 업로드에 실패했습니다.')
    return redirect(url_for('index'))

  # set the file_urls and remove the session variable
  file_urls = session['file_urls']
  session.pop('file_urls', None)

  userid = session.get('userid', None)

  if 'userid' in session:
    return render_template('results.html', file_urls=file_urls, userid=userid)
  else:
    flash("로그인 후 사용해주세요.")
    return redirect(url_for('login'))

@app.route('/show_pictures', methods=['GET', 'POST'])     #업로드 사진 확인
def show_pictures():
  userid = session.get('userid', None)
  valid_images = [".jpg", ".jpeg", ".gif", ".png", ".tga"]

  imgs_urls = []        #이미지 주소 리스트
  faces_list_urls = []  #분류된 얼굴 이미지 주소 리스트 (이중 리스트)
  faces_name_list = []  #분류된 얼굴 이름 리스트 (이중 리스트)
  name_list = []        #분류된 얼굴 이름 중복 없는 리스트

  if 'userid' in session:
    user_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], "pictures" ,str(userid))
    select = request.form.get('name')

    for dir in os.listdir(user_path):   #이미지 폴더에서
      curr_path = os.path.join(user_path, dir)    #날짜별 디렉터리
      if (os.path.isdir(curr_path)):
        for file in os.listdir(curr_path):    #업로드한 이미지
          name = os.path.splitext(file)[0]
          ext = os.path.splitext(file)[1]
          if ext.lower() not in valid_images:
            continue
          else:
            imgs_urls.append(os.path.join("..","static", "uploads", "pictures" ,str(userid), dir, file))    #업로드에 있는 이미지 상대주소

            if os.path.exists(os.path.join(curr_path,name)):  #분류된 얼굴 이미지
              face_list = []
              face_name_list = []
              for img in os.listdir(os.path.join(curr_path,name)):
                face_list.append(os.path.join("..","static", "uploads", "pictures" ,str(userid), dir, name, img))
                face_name_list.append(str((os.path.splitext(img)[0]).split('_')[0]))
                name_list.append(str((os.path.splitext(img)[0]).split('_')[0]))
              faces_list_urls.append(face_list)
              faces_name_list.append(face_name_list)

    name_list = list(set(name_list))

    #선택된 이미지만 보여주기
    if(select is None or select == "all"):
      print(imgs_urls)
      return render_template('show_pictures.html', userid=userid
                           , img_url_info=zip(imgs_urls, faces_list_urls, faces_name_list)
                           , name_list=name_list, select=select, imgs_urls=imgs_urls)
    else:
      selected_imgs_urls = []
      selected_faces_urls = []
      selected_names_list = []

      for index, names in enumerate(faces_name_list):
        if any(str(select) in n for n in names):
          selected_imgs_urls.append(imgs_urls[index])
          selected_faces_urls.append(faces_list_urls[index])
          selected_names_list.append(faces_name_list[index])

      print(selected_imgs_urls)
      return render_template('show_pictures.html', userid=userid
                               , img_url_info=zip(selected_imgs_urls, selected_faces_urls, selected_names_list)
                               , name_list=name_list, select=select, imgs_urls=selected_imgs_urls)

  #login session 없을 때
  else:
    flash("로그인 후 사용해주세요.")
    return redirect(url_for('login'))

@app.route('/public_show', methods=['GET', 'POST'])     #업로드 사진 확인
def public_show():
  userid = session.get('userid', None)
  valid_images = [".jpg", ".jpeg", ".gif", ".png", ".tga"]

  imgs_urls = []        #이미지 주소 리스트

  if 'userid' in session:
    public_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], "public")

    for file in os.listdir(public_path):   #이미지 폴더에서
      name = os.path.splitext(file)[0]
      ext = os.path.splitext(file)[1]

      if ext.lower() not in valid_images:
        continue
      else:
        if os.path.exists(os.path.join(public_path, name)):  #분류된 얼굴 이미지 중 내얼굴이 있으면
          flag = False

          for img in os.listdir(os.path.join(public_path,name)):

            if str(userid) == str((os.path.splitext(img)[0]).split('_')[0]):
              flag = True
              break

          if flag:
            imgs_urls.append(os.path.join("..", "static", "uploads", "public", file))  # 업로드에 있는 이미지 상대주소

    print(imgs_urls)
    return render_template('public_show.html', userid=userid, imgs_urls=imgs_urls)

  #login session 없을 때
  else:
    flash("로그인 후 사용해주세요.")
    return redirect(url_for('login'))


#첫 화면
@app.route('/') # 접속하는 url
@app.route('/index')
def index():
  userid = session.get('userid', None)
  return render_template('index.html', userid=userid)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html')

if __name__=="__main__":
  basedir = os.path.abspath(os.path.dirname(__file__))
  dbfile = os.path.join(basedir, 'db.sqlite')

  app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + dbfile
  app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
  app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
  app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

  # flask-wtf csrf 공격으로부터 모든 폼을 보호
  #  csrf = CSRFProtect()
  #  csrf.init_app(app)

  #db생성
  db.init_app(app)
  db.app = app
  db.create_all()  # db 생성

  dropzone = Dropzone(app)

  # Dropzone settings
  app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
  app.config['DROPZONE_PARALLEL_UPLOADS'] = 30
  app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
  app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
  app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

  # Uploads settings
  app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(os.getcwd(), 'static', 'uploads')
  photos = UploadSet('photos', IMAGES)
  configure_uploads(app, photos)
  patch_request_class(app)  # set maximum file size, default is 16MB

#  check_thread = threading.Thread(
#    target=close_running_threads
#  )

  app.run(host="127.0.0.1", port="5000", debug=True)
