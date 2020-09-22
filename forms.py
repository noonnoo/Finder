from flask_wtf import FlaskForm

from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, EqualTo
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from user_models import Finderuser

photos = UploadSet('photos', IMAGES)

#FlaskForm: 폼 생성
class SigninForm(FlaskForm):
    userid = StringField('userid', validators=[DataRequired()]) #데이터 필드가 비어있는지 확인
    username = StringField('username', validators=[DataRequired()])
    password = PasswordField('password', validators=[DataRequired(), EqualTo('re_password')]) #password와 같은지 확인
    re_password = PasswordField('re_password', validators=[DataRequired()])

class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file!')])
    submit = SubmitField('Upload')

class LoginForm(FlaskForm):
    class UserPassword(object):
        def __init__(self, message=None):
            self.message = message

        def __call__(self, form, field):
            userid = form['userid'].data
            password = field.data
            finderuser = Finderuser.query.filter_by(userid=userid).first()
            if(finderuser is None):
                raise ValueError('등록된 아이디가 아닙니다.')
            if finderuser.password != password:
                # raise ValidationError(message % d)
                raise ValueError('잘못된 비밀번호입니다.')
    userid = StringField('userid', validators=[DataRequired()])
    password = PasswordField('password', validators=[DataRequired(), UserPassword()])