from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import HiddenField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional  # Add Optional


class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = StringField('Phone', validators=[DataRequired(), Length(min=10, max=15)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password', message='Passwords must match')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


class SettingsForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = StringField('Phone', validators=[DataRequired(), Length(min=11, max=12)])
    password = PasswordField('New Password', validators=[
        Optional(),  # Make password optional
        Length(min=6)
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        Optional(),  # Make confirm_password optional
        EqualTo('password', message='Passwords must match')
    ])
    profile_picture = FileField('Profile Picture', validators=[
        FileAllowed(['jpg', 'png', 'jpeg'])
    ])
    submit = SubmitField('Update Profile')





class ForgotPasswordForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Send Reset Link')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('password', message='Passwords must match')])
    submit = SubmitField('Reset Password')

class AdminLoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


class DeleteForm(FlaskForm):
    # This empty form is just for CSRF protection
    class Meta:
        csrf = True

        

class MessageForm(FlaskForm):
    message = TextAreaField('Message', validators=[
        DataRequired(),
        Length(min=5, max=1000)
    ])
    submit = SubmitField('Send Message')


class CommentForm(FlaskForm):
    comment = TextAreaField('Comment', validators=[
        DataRequired(),
        Length(min=5, max=500)
    ])
    submit = SubmitField('Submit Comment')



class RiceDiagnosisForm(FlaskForm):
    image = FileField('Leaf Image', validators=[DataRequired()])
    submit = SubmitField('Analyze')



class WheatDiagnosisForm(FlaskForm):
    image = FileField('Leaf Image', validators=[FileRequired()])
    submit = SubmitField('Diagnose')

