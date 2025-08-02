from flask import Flask, render_template, redirect, url_for, flash, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_wtf import CSRFProtect
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from forms import LoginForm, RegisterForm, SettingsForm
from models import db, User
from datetime import timedelta, datetime, timezone
import os
from forms import ResetPasswordForm, ForgotPasswordForm# Import the ResetPasswordForm class
from forms import AdminLoginForm
from functools import wraps
from forms import DeleteForm
from flask_wtf.csrf import validate_csrf
from wtforms import ValidationError
from forms import MessageForm
from forms import CommentForm
from bleach import clean
from models import Comment
from werkzeug.utils import secure_filename
import cv2
import joblib
import numpy as np
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Update based on your model's needs
from tensorflow.keras.applications.inception_v3 import preprocess_input
from utils import get_treatment
from forms import RiceDiagnosisForm
from models import Activity
from forms import DeleteForm
from flask import abort  # Add this import at the top
from sqlalchemy.exc import SQLAlchemyError  # Add this import
from forms import WheatDiagnosisForm


app = Flask(__name__)



@app.template_filter('time_ago')
def time_ago(dt):
    """Convert datetime to human-readable 'time ago' format."""
    if not dt:
        return "N/A"
    
    # Get current time in UTC (aware)
    now = datetime.now(timezone.utc)

    # Normalize dt (make it UTC-aware if naive)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Calculate difference
    diff = now - dt
    seconds = diff.total_seconds()

    if seconds < 60:
        return f"{int(seconds)} sec ago"
    elif seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    elif seconds < 86400:
        return f"{int(seconds // 3600)} hrs ago"
    else:
        return f"{int(seconds // 86400)} days ago"



# ────── App Configuration ──────
app.config.from_object('config.Config')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # Add this
# app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# ────── Admin Configuration ──────
app.config['ADMIN_EMAIL'] = 'magsibalochzahidalimagsi@gmail.com'
app.config['ADMIN_PASSWORD'] = 'zahid12@'  # Store plaintext temporarily for setup


# ────── Email Configuration ──────
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'zahidalimagsimagsibaloch@gmail.com'
app.config['MAIL_PASSWORD'] = 'xeob oytu flpx guyq'
app.config['MAIL_DEFAULT_SENDER'] = 'zahidalimagsimagsibaloch@gmail.com'

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# ────── Extensions Init ──────
db.init_app(app)
migrate = Migrate(app, db)
csrf = CSRFProtect(app)

# ────── Login Manager ──────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ────── Utility Functions ──────
def generate_verification_token(email):
    return serializer.dumps(email, salt='email-confirm')

def confirm_verification_token(token, expiration=3600):
    try:
        email = serializer.loads(token, salt='email-confirm', max_age=expiration)
    except Exception:
        return False
    return email


def log_activity(user_id, activity_type, details, ip_address=None, comment_content=None, treatment_recommendations=None):
    """Log user activity to database"""
    if not ip_address:
        ip_address = request.remote_addr
    
    new_activity = Activity(
        user_id=user_id,
        activity_type=activity_type,
        details=details,
        ip_address=ip_address,
        comment_content=comment_content,
        treatment_recommendations=treatment_recommendations
    )
    db.session.add(new_activity)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error logging activity: {str(e)}")


def send_verification_email(user):
    token = generate_verification_token(user.email)
    confirm_url = url_for('verify_email', token=token, _external=True)
    html = render_template('activate.html', confirm_url=confirm_url, username=user.username)
    subject = "Please verify your email"
    msg = Message(subject, recipients=[user.email], html=html)
    mail.send(msg)




# Add near Jinja environment setup
@app.context_processor
def utility_processor():
    return dict(get_treatment=get_treatment)






# ────── Routes ──────
@app.route('/')
def index():
    return render_template('index.html')


#     return render_template('register.html', form=form)
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        try:
            existing = User.query.filter(
                (User.email == form.email.data) | 
                (User.phone == form.phone.data) |
                (User.username == form.username.data)
            ).first()
            
            if existing:
                conflict = []
                if existing.email == form.email.data:
                    conflict.append("Email")
                if existing.phone == form.phone.data:
                    conflict.append("Phone")
                if existing.username == form.username.data:
                    conflict.append("Username")
                flash(f"{', '.join(conflict)} already in use!", 'danger')
                return redirect(url_for('register'))

            hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
            new_user = User(
                username=form.username.data,
                email=form.email.data,
                phone=form.phone.data,
                password=hashed_password,
                is_verified=False
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            send_verification_email(new_user)
            flash('Registration successful! Please check your email.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'danger')
            app.logger.error(f"Registration error: {str(e)}")

    return render_template('register.html', form=form)




@app.route('/verify/<token>')
def verify_email(token):
    try:
        email = serializer.loads(token, salt='email-confirm', max_age=3600)  # 1 hour validity
    except SignatureExpired:
        flash("The verification link has expired.", "warning")
        return redirect(url_for('login'))
    except BadSignature:
        flash("Invalid verification link.", "danger")
        return redirect(url_for('login'))

    user = User.query.filter_by(email=email).first()
    if user:
        if user.is_verified:
            flash("Your account is already verified. Please login.", "info")
        else:
            user.is_verified = True
            db.session.commit()
            flash("Email verified successfully! You can now log in.", "success")
    else:
        flash("User not found.", "danger")

    return redirect(url_for('login'))




@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter(
            (User.username == form.username.data) | 
            (User.email == form.username.data)
        ).first()
        
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            
            # Log login activity
            log_activity(
                user.id,
                'login',
                f"Successful login from {request.remote_addr}",
                ip_address=request.remote_addr
            )
            
            flash(f'Welcome back {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            # ✅ Flash error message if username or password is invalid
            flash('Invalid username or password', 'danger')
            # You can log this attempt too for security
            app.logger.warning(f"Failed login attempt for username/email: {form.username.data}")

    return render_template('login.html', form=form)



@app.route('/dashboard')
@login_required
def dashboard():
    # Redirect admins to admin dashboard
    if current_user.is_admin:
        return redirect(url_for('admin_dashboard'))
    return render_template('dashboard.html', user=current_user)







@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))




# ====== Model Loading ======
def load_disease_model():
    """Load model and class names with proper error handling"""
    model_path = os.path.join(app.root_path, 'model_train/rice_disease_model.h5')
    class_path = os.path.join(app.root_path, 'model_train/class_indices.pkl')
    
    try:
        model = load_model(model_path)
        model.make_predict_function()  # For thread safety
        class_indices = joblib.load(class_path)
        class_names = {v: k.replace('_', ' ').title() for k, v in class_indices.items()}
        app.logger.info("✅ Model and classes loaded successfully")
        return model, class_names
    except Exception as e:
        app.logger.critical(f"❌ Critical model loading error: {str(e)}")
        return None, None

# Initialize model and classes
rice_model, class_names = load_disease_model()

# ====== Configuration ======
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Secure file validation"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/rice-diagnosis', methods=['GET', 'POST'])
@login_required
def rice_diagnosis():
    form = RiceDiagnosisForm()
    prediction = None
    # ... existing code ...
    if prediction:
        log_activity(
            current_user.id,
            'rice_diagnosis',
            f"Diagnosed {prediction['class']} with {prediction['confidence']}% confidence",
            request.remote_addr,
            treatment_recommendations="|".join(get_treatment(prediction['class']))
        )

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
            
        file = request.files['image']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        if not rice_model or not class_names:
            flash('Diagnosis service is unavailable', 'danger')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            try:
                # Secure file handling
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Preprocessing pipeline
                img = image.load_img(filepath, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                # Model prediction
                preds = rice_model.predict(img_array)
                pred_class = np.argmax(preds[0])
                confidence = round(100 * np.max(preds[0]), 2)
                
                # Format class name
                class_name = class_names.get(pred_class, 'Unknown')
                
                prediction = {
                    'class': class_name,
                    'confidence': confidence,
                    'filename': filename,
                    'treatment': get_treatment(class_name)
                }
                



            except Exception as e:
                app.logger.error(f"Prediction error: {str(e)}")
                flash('Error processing image', 'danger')
                # Only remove file if there was an error
                if os.path.exists(filepath):
                    os.remove(filepath)
        

    return render_template('rice.html', 
                         form=form, 
                         prediction=prediction)




# ====== Wheat Model Loading ======
def load_wheat_model():
    """Load wheat model and class names with error handling"""
    model_path = os.path.join(app.root_path, 'model_train/wheat_inceptionv3_model.h5')
    class_path = os.path.join(app.root_path, 'model_train/wheat_class_indices.pkl')

    try:
        if not os.path.exists(model_path):
            app.logger.critical(f"❌ Wheat model not found at {model_path}")
            return None, None
        if not os.path.exists(class_path):
            app.logger.critical(f"❌ Wheat class index file not found at {class_path}")
            return None, None

        model = load_model(model_path)
        model.make_predict_function()  # Thread safety

        class_indices = joblib.load(class_path)
        class_names = {v: k.replace('_', ' ').title() for k, v in class_indices.items()}

        app.logger.info("✅ Wheat model and classes loaded successfully")
        return model, class_names
    except Exception as e:
        app.logger.critical(f"❌ Error loading wheat model: {str(e)}")
        return None, None


# Initialize wheat model and classes
wheat_model, wheat_class_names = load_wheat_model()


@app.route('/wheat-diagnosis', methods=['GET', 'POST'])
@login_required
def wheat_diagnosis():
    form = WheatDiagnosisForm()
    prediction = None

    if request.method == 'POST':
        # ===== Check for image in request =====
        if 'image' not in request.files:
            app.logger.warning("DEBUG: No file uploaded in request")
            flash('No file uploaded', 'danger')
            return redirect(request.url)

        file = request.files['image']

        # ===== Validate file name =====
        if file.filename == '':
            app.logger.warning("DEBUG: File name is empty")
            flash('No selected file', 'danger')
            return redirect(request.url)

        # ===== Check if model is available =====
        if not wheat_model or not wheat_class_names:
            app.logger.error("DEBUG: Wheat model or classes not loaded")
            flash('Wheat diagnosis service is unavailable', 'danger')
            return redirect(request.url)

        # ===== Validate file type =====
        if not allowed_file(file.filename):
            app.logger.warning(f"DEBUG: Invalid file type - {file.filename}")
            flash('Invalid file type. Please upload JPG or PNG image.', 'danger')
            return redirect(request.url)

        try:
            # ===== Save file securely =====
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            app.logger.info(f"Image saved at {filepath}")

            # ===== Preprocess Image =====
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # ===== Predict using model =====
            preds = wheat_model.predict(img_array)
            pred_class = np.argmax(preds[0])
            confidence = round(100 * np.max(preds[0]), 2)

            # ===== Get readable class name =====
            class_name = wheat_class_names.get(pred_class, 'Unknown')

            # ===== Prepare prediction data =====
            prediction = {
                'class': class_name,
                'confidence': confidence,
                'filename': filename,
                'treatment': get_treatment(class_name)
            }

            app.logger.info(f"✅ Prediction: {prediction}")

            # ===== Log activity =====
            log_activity(
                current_user.id,
                'wheat_diagnosis',
                f"Diagnosed {prediction['class']} with {prediction['confidence']}% confidence",
                request.remote_addr,
                treatment_recommendations="|".join(prediction['treatment'])
            )

        except Exception as e:
            app.logger.error(f"❌ Prediction error: {str(e)}")
            flash('Error processing image', 'danger')
            if os.path.exists(filepath):
                os.remove(filepath)

    return render_template('wheat.html', form=form, prediction=prediction)





@app.route('/about')
def about():
    form = CommentForm()
    comments = Comment.query.order_by(Comment.created_at.desc()).all()
    return render_template('about.html', form=form, comments=comments)





@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    form = SettingsForm(obj=current_user)
    
    if form.validate_on_submit():
        try:
            # Update basic fields
            form.populate_obj(current_user)
            
            # Handle password update only if both fields are filled
            if form.password.data and form.confirm_password.data:
                current_user.password = generate_password_hash(form.password.data)
            
            # Handle profile picture update
            if form.profile_picture.data:
                file = form.profile_picture.data
                ext = file.filename.rsplit('.', 1)[-1].lower()
                if ext not in ['jpg', 'jpeg', 'png', 'gif']:
                    flash("Invalid file type. Please upload an image.", 'danger')
                    return redirect(url_for('settings'))
                
                filename = f"user_{current_user.id}.{ext}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Remove old image
                if current_user.profile_pic and current_user.profile_pic != 'logo.jpg':
                    old_path = os.path.join(app.config['UPLOAD_FOLDER'], current_user.profile_pic)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                
                file.save(file_path)
                current_user.profile_pic = filename
            
            # Always update the timestamp
            current_user.updated_at = datetime.utcnow()
            db.session.commit()
            flash('Profile updated successfully!', 'success')
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating profile: {str(e)}', 'danger')
        
        return redirect(url_for('settings'))
    
    form.csrf_token.data = request.form.get('csrf_token', '')
    return render_template('settings.html', form=form)
    # Add CSRF token to form
   





# ────── Password Reset Functions ──────
def verify_reset_token(token):
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=3600)  # 1 hour validity
    except SignatureExpired:
        return None  # Expired token
    except BadSignature:
        return None  # Invalid token
    return User.query.filter_by(email=email).first()

def update_password(user, password):
    user.password = generate_password_hash(password, method='pbkdf2:sha256')
    db.session.commit()

def send_email(to, subject, body):
    msg = Message(subject, recipients=[to])
    msg.body = body
    mail.send(msg)



# Forgot Password Route
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        email = form.email.data
        user = User.query.filter_by(email=email).first()
        if user:
            # Generate token
            token = s.dumps(user.email, salt='password-reset-salt')
            reset_link = url_for('reset_password', token=token, _external=True)

            # Send reset email
            send_email(user.email, "Password Reset", f"Click here to reset your password: {reset_link}")

            flash('A password reset link has been sent to your email.', 'info')
            return redirect(url_for('login'))
        else:
            flash('Email does not exist.', 'warning')

    return render_template('forgot_password.html', form=form)



# Reset Password Route
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    form = ResetPasswordForm()
    if request.method == 'POST' and form.validate_on_submit():
        password = form.password.data
        confirm_password = form.confirm_password.data

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('reset_password', token=token))

        user = verify_reset_token(token)  # Verify token
        if user:
            update_password(user, password)  # Update the password
            flash('Your password has been updated!', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid or expired token', 'danger')
            return redirect(url_for('forgot_password'))

    return render_template('reset_password.html', form=form, token=token)









def create_admin_user():
    with app.app_context():  # Ensure we're in app context
        admin_email = app.config['ADMIN_EMAIL']
        admin_user = User.query.filter_by(email=admin_email).first()
        
        if not admin_user:
            hashed_password = generate_password_hash(app.config['ADMIN_PASSWORD'])
            admin_user = User(
                username='admin',
                email=admin_email,
                password=hashed_password,
                is_verified=True,
                is_admin=True,
                phone='0000000000'  # Required field based on your model
            )
            db.session.add(admin_user)
            db.session.commit()
            print("✅ Admin user created successfully!")
        else:
            print("ℹ️ Admin user already exists")




def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not (current_user.is_authenticated and getattr(current_user, 'is_admin', False)):
            flash("Admin access required", "danger")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated and getattr(current_user, 'is_admin', False):
        return redirect(url_for('admin_dashboard'))
    
    form = AdminLoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        
        # Debug prints (remove in production)
        print(f"User found: {user}")
        if user:
            print(f"Password check: {check_password_hash(user.password, form.password.data)}")
            print(f"Is admin: {getattr(user, 'is_admin', False)}")
        
        if user and check_password_hash(user.password, form.password.data) and getattr(user, 'is_admin', False):
            login_user(user)
            flash("Admin logged in successfully.", "success")
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Invalid admin credentials.", "danger")
    return render_template('admin_login.html', form=form)






# Move this route ABOVE admin_dashboard
@app.route('/admin/user/<int:user_id>')
@login_required
@admin_required
def view_user_details(user_id):
    user = User.query.get_or_404(user_id)
    return render_template('view_user.html', user=user)





@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    try:
        users = User.query.all()
        recent_activities = Activity.query.order_by(Activity.created_at.desc()).limit(50).all()
        wheat_diagnoses = Activity.query.filter(
            Activity.activity_type == 'wheat_diagnosis'
        ).order_by(Activity.created_at.desc()).limit(5).all()
        rice_diagnoses = Activity.query.filter(
            Activity.activity_type == 'rice_diagnosis'
        ).order_by(Activity.created_at.desc()).limit(5).all()

        return render_template(
            'admin_dashboard.html',
            users=users,
            recent_activities=recent_activities,
            wheat_diagnoses=wheat_diagnoses,
            rice_diagnoses=rice_diagnoses
        )
        
    except SQLAlchemyError as e:
        app.logger.error(f"Database error in admin dashboard: {str(e)}")
        flash('Error loading dashboard data', 'danger')
        return redirect(url_for('dashboard'))
    except Exception as e:
        app.logger.critical(f"Unexpected error in admin dashboard: {str(e)}")
        abort(500)  # Single abort statement







# ────── Delete User ──────
@app.route('/admin/user/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    form = DeleteForm()
    if form.validate_on_submit():
        user = User.query.get_or_404(user_id)
        db.session.delete(user)
        db.session.commit()
        flash('User deleted successfully!', 'success')
    else:
        flash('Invalid request', 'danger')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/send_message/<int:user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def send_message(user_id):
    user = User.query.get_or_404(user_id)
    form = MessageForm()
    
    if form.validate_on_submit():
        message_content = form.message.data
        send_email(user.email, "Message from Field Shield", message_content)
        flash(f"Message sent to {user.username}.", "success")
        return redirect(url_for('admin_dashboard'))
    
    return render_template('send_message.html', user=user, form=form)




@app.route('/submit_comment', methods=['POST'])
@login_required
def submit_comment():
    form = CommentForm(request.form)
    
    if form.validate_on_submit():
        try:
            # Sanitize and save comment
            sanitized_content = clean(form.comment.data)
            new_comment = Comment(
                content=sanitized_content,
                user_id=current_user.id
            )
            db.session.add(new_comment)
            db.session.commit()

            # Log the comment activity
            log_activity(
                current_user.id,
                'comment',
                f"Posted a comment on About page",
                request.remote_addr,
                comment_content=sanitized_content
            )

            # Send email to admin
            msg = Message("New User Comment",
                        recipients=[app.config['ADMIN_EMAIL']])
            msg.html = render_template(
                'comment_alert.html',
                user=current_user,
                comment=new_comment,
                timestamp=datetime.utcnow()
            )
            mail.send(msg)

            flash('Comment posted successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error submitting comment: {str(e)}")
            flash('Error posting comment. Please try again.', 'danger')
        
        return redirect(url_for('about'))
    
    # Handle form errors
    for field, errors in form.errors.items():
        for error in errors:
            flash(f"{field}: {error}", 'danger')
    
    return redirect(url_for('about'))


@app.route('/delete_comment/<int:comment_id>', methods=['POST'])
@login_required
@admin_required
def delete_comment(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    db.session.delete(comment)
    db.session.commit()
    flash('Comment deleted successfully', 'success')
    return redirect(url_for('about'))


# ────── Run App ──────
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_admin_user()
    app.run(debug=True)