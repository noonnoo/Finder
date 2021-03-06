# finder
졸업 작품 - 등록된 얼굴이 있는 사진을 보여주는 서비스

*src만 업로드

### 기능 개요
* 본인 얼굴 등록
* 개인 서버에 사진 업로드, 사진에 등록된 얼굴들 별로 분류
* 공공 서버에 사진 업로드, 자신이 나온 사진만 받아볼 수 있는 기능

### 구현
* (얼굴 검출)face_recognition의 face_location함수를 이용해서 단체사진 속 얼굴을 크롭
* (얼굴 검출)개인 사진의 경우 facenet의 detect_face.py을 사용하여 얼굴 부분을 뽑아냄. 이 모듈은 MTCNN으로 얼굴들의 위치를 알아내기 때문에 정확도가 더 높습니다. 독사진은 얼굴인 부분만 정확하게 뽑아내야 하기 때문에 이를 활용함.
* (얼굴 인식)모델은 facenet 깃헙에서 제공하는 pretrained model 사용. 해당 모델은 330만개의 얼굴 데이터가 있는 VGGFace2 데이터셋으로 학습시킨 모델로 LFW데이터셋에서는 0.9965의 정확도를 갖고 있는 모델임.
* (웹페이지 구현) 서버는 flask로 구현하고, 이외 웹 페이지는 html, javascript, css로 구현함.
* 단체사진 업로드 이후에 쓰레드를 활용하여 얼굴 검출 및 인식을 하고 결과를 돌려받아서 결과창에 띄움


### 개발 동기
* 여행, 동아리, 동호회 활동 이후 받는 수백, 수천장의 사진을 분류하는 서비스의 필요성을 느낌

### 사용한 것들 
* 사용 모델: https://github.com/davidsandberg/facenet 의 20180402-114759
* 웹 서버: flask
* 웹 디자인 템플릿: https://html5up.net/phantom

### 첫 화면
![127 0 0 1_5000_index](https://user-images.githubusercontent.com/33820372/93913928-bbe06100-fd40-11ea-9be5-d1642cca5d0b.png)

### 졸업작품 발표 영상
링크: https://youtu.be/Y_Cp4rOoP2k
