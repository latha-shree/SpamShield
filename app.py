# app.py (User-side only version)
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
import os, base64, tempfile, json
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from io import StringIO
import pytesseract
import speech_recognition as sr
from pydub import AudioSegment
import sqlite3
from PIL import Image
import pytesseract
from flask import Flask, render_template, request, redirect, session, url_for
import imaplib, email, sqlite3, joblib
from email.header import decode_header
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


app = Flask(__name__)
app.secret_key = 'secretkey'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model_data = joblib.load('model/spam_model.pkl')
model = model_data['model']
vectorizer = model_data['vectorizer']

# Database init
def init_db():
    conn = sqlite3.connect('users.db')
    cur = conn.cursor()

    # Create users table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # Create admin table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # Insert default admin only if not exists
    cur.execute("SELECT * FROM admin WHERE username = ?", ('admin',))
    if not cur.fetchone():
        cur.execute("INSERT INTO admin (username, password) VALUES (?, ?)", ('admin', 'admin123'))

    conn.commit()
    conn.close()
   
init_db()

# ─────────── Routes ────────────
@app.route('/')
def home():
    return render_template('index.html')



"""@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        conn = sqlite3.connect('users.db')
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (u, p))
            conn.commit()
            return redirect('/login')
        except:
            return "Username already exists"
    return render_template('register.html')"""

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        u = request.form['username'].strip()
        p = request.form['password'].strip()

        # === Username Validation ===
        if not u:
            return "Username cannot be left blank"
        if u[0].isdigit():
            return "Username cannot start with a number"
        if len(u) < 3:
            return "Username must be at least 3 characters"

        # === Password Validation ===
        if not p:
            return "Password cannot be blank"
        if p[0] == ' ':
            return "No leading spaces allowed in password"
        if len(p) < 6:
            return "Password must be at least 6 characters"

        # === Insert to DB ===
        conn = sqlite3.connect('users.db')
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (u, p))
            conn.commit()
            return redirect('/login')
        except:
            return "Username already exists"
    return render_template('register.html')


"""@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        conn = sqlite3.connect('users.db')
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (u,))
        row = cur.fetchone()
        if row and row[0] == p:
            session['user'] = u
            return redirect('/user_home')
        return "Invalid credentials"
    return render_template('login.html')"""

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form['username'].strip()
        p = request.form['password'].strip()

        # === Basic Field Check ===
        if not u:
            return "Username is required"
        if not p:
            return "Password is required"

        # === DB Lookup ===
        conn = sqlite3.connect('users.db')
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (u,))
        row = cur.fetchone()

        if row and row[0] == p:
            session['user'] = u
            return redirect('/user_home')
        return "Invalid credentials"
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

"""@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':
            session['admin'] = username
            return redirect('/admin')
        else:
            return render_template('admin_login.html', error="Invalid credentials.")
    return render_template('admin_login.html')"""

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':
            session['admin'] = username
            return redirect('/admin_dashboard')  # ✅ Redirect here first
        else:
            return render_template('admin_login.html', error="Invalid credentials.")
    return render_template('admin_login.html')




@app.route('/admin')
def admin():
    return render_template('admin.html')
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    file = request.files['csv_file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read file with encoding fallback
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')

    # 🔍 STEP 1: Detect label column
    possible_label_cols = ['label', 'v1', 'type', 'category', 'tag']
    label_col = None
    for col in df.columns:
        if col.strip().lower() in possible_label_cols:
            label_col = col
            break

    # 🔍 STEP 2: Detect message column
    possible_msg_cols = ['message', 'text', 'v2', 'content', 'msg']
    msg_col = None
    for col in df.columns:
        if col.strip().lower() in possible_msg_cols:
            msg_col = col
            break

    if not label_col or not msg_col:
        return "Error: Could not detect label or message columns. Expected something like ['v1', 'label'] and ['v2', 'message'].", 400

    # ✅ Rename to standard columns
    df = df.rename(columns={label_col: 'label', msg_col: 'message'})

    # STEP 3: Normalize labels
    df['label'] = df['label'].astype(str).str.lower()

    spam_labels = ['spam', 'junk', '1', 'false', 'spammsg', 'spm']
    ham_labels = ['ham', 'normal', '0', 'true', 'hammsg', 'legit']

    def clean_label(val):
        if val in spam_labels:
            return 'spam'
        elif val in ham_labels:
            return 'ham'
        else:
            return 'unknown'

    df['label'] = df['label'].apply(clean_label)

    # STEP 4: Remove unknowns
    df = df[df['label'].isin(['spam', 'ham'])]
    df = df[['label', 'message']]

    if df.empty:
        return "Error: No valid spam/ham rows found after label cleanup.", 400
    



    from sklearn.utils import resample

    # Separate majority and minority classes
    spam_df = df[df['label'] == 'spam']
    ham_df = df[df['label'] == 'ham']

    # Find the smaller class size
    min_count = min(len(spam_df), len(ham_df))

    # Downsample both classes to the same size (or upsample if you prefer)
    spam_sampled = resample(spam_df, replace=True, n_samples=min_count, random_state=42)
    ham_sampled = resample(ham_df, replace=True, n_samples=min_count, random_state=42)

    # Combine the two balanced datasets
    df = pd.concat([spam_sampled, ham_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)


    # ✅ Train model
    model = train_model(df)
    model, accuracy = train_model(df)

    joblib.dump(model, 'spam_model.pkl')

    # ✅ Charts
    ham = (df['label'] == 'ham').sum()
    spam = (df['label'] == 'spam').sum()
    generate_charts(ham, spam)


   



    return render_template('charts.html', ham=ham, spam=spam)


def train_model(df):
    X = df['message']
    y = df['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    model.fit(X_train, y_train)
    return model





def generate_charts(ham, spam):
    # Pie chart
    labels = ['Ham', 'Spam']
    sizes = [ham, spam]
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
    plt.title("Spam vs Ham")
    plt.savefig('static/pie_chart.png')
    plt.close()

    # Bar chart
    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=sizes, palette='pastel')
    plt.title("Message Distribution")
    plt.ylabel("Count")
    plt.savefig('static/bar_chart.png')
    plt.close()

@app.route('/threat_tips')
def threat_tips():
    return render_template('threat_tips.html')

@app.route('/user_home')
def user_home():
    if 'user' not in session:
        return redirect('/login')
    return render_template('user_home.html')

"""@app.route('/user', methods=['GET', 'POST'])
def user_text():
    user_input = ""
    prediction = None
    if request.method == 'POST':
        user_input = request.form.get('message')
        msg = request.form['message'].strip().lower()  # optional cleaning
        result = model.predict([msg])[0]               # ✅ correct
        prediction = "Spam" if result == 1 else "Not Spam"
    return render_template('user.html', prediction=prediction, user_input=user_input)"""

@app.route('/user', methods=['GET', 'POST'])
def user_text():
    if request.method == 'POST':
        user_input = request.form.get('message')
        msg = user_input.strip().lower()
        result = model.predict([msg])[0]
        prediction = "Spam" if result == 1 else "Not Spam"
        
        # Store in session temporarily
        session['prediction'] = prediction
        session['user_input'] = user_input

        # Redirect to GET route
        return redirect(url_for('user_text'))

    # On GET request
    prediction = session.pop('prediction', None)
    user_input = session.pop('user_input', '')

    return render_template('user.html', prediction=prediction, user_input=user_input)


@app.route('/user/image', methods=['GET', 'POST'])
def user_image():
    prediction = extracted = None
    if request.method == 'POST':
        file = request.files['image_file']
        if file:
            img = Image.open(file.stream)
            extracted = pytesseract.image_to_string(img)
            if extracted.strip():
                result = model.predict([extracted])[0]   # ✅ RAW TEXT
                prediction = "Spam" if result == 1 else "Not Spam"
            else:
                prediction = "No text found."
    return render_template('user_image.html', prediction=prediction, extracted_text=extracted)

   


@app.route('/user/audio', methods=['GET', 'POST'])
def user_audio():
    prediction = extracted = None
    if request.method == 'POST':
        audio_file = request.files['audio_file']
        if audio_file:
            # Save uploaded .wav file
            temp_path = os.path.join('uploads', 'temp_audio.wav')
            audio_file.save(temp_path)

            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_path) as source:
                audio = recognizer.record(source)
                try:
                    # Transcribe speech to text
                    extracted = recognizer.recognize_google(audio)
                    if extracted.strip():
                        # ✅ Let pipeline handle vectorizing internally
                        result = model.predict([extracted])[0]
                        prediction = "Spam" if result == 1 else "Not Spam"
                    else:
                        prediction = "No speech detected."
                except Exception as e:
                    prediction = f"Could not process audio: {str(e)}"

    return render_template('user_audio.html', prediction=prediction, extracted_text=extracted)

@app.route('/user/batch', methods=['GET', 'POST'])
def user_batch():
    if request.method == 'POST':
        file = request.files['csv_file']
        try:
            df = pd.read_csv(file)
        except:
            file.stream.seek(0)
            df = pd.read_csv(file, encoding='latin1')

        # Detect message column
        msg_col = None
        for col in df.columns:
            if col.strip().lower() in ['message', 'text', 'content']:
                msg_col = col
                break

        if not msg_col:
            return render_template('user_batch.html', error="❌ No message/text column found.")

        # Predict (send raw text directly)
        preds = model.predict(df[msg_col].astype(str))
        df['Prediction'] = ['Spam' if p == 1 else 'Not Spam' for p in preds]

        # Count data
        total = len(df)
        spam_count = df['Prediction'].value_counts().get('Spam', 0)
        ham_count = df['Prediction'].value_counts().get('Not Spam', 0)

        # Prepare chart data
        chart_data = {
            'labels': ['Spam', 'Not Spam'],
            'data': [int(spam_count), int(ham_count)]  # 👈 convert to regular int
            }


        # Downloadable CSV
        csv_io = StringIO()
        df.to_csv(csv_io, index=False)
        csv_download = base64.b64encode(csv_io.getvalue().encode()).decode()

        return render_template('user_batch.html',
                               table=df.head(10).to_html(classes="table table-bordered", index=False),
                               chart_data=json.dumps(chart_data),
                               total=total,
                               spam=spam_count,
                               ham=ham_count,
                               csv_download=csv_download)

    return render_template('user_batch.html')

@app.route("/gmail_auth", methods=["GET", "POST"])
def gmail_auth():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        session["gmail_user"] = request.form["gmail_user"].strip()
        session["gmail_pass"] = request.form["gmail_pass"]
        return redirect(url_for("inbox"))

    # simple inline form – style later if you wish
    return render_template("gmail_auth.html")



@app.route("/help")
def help():
    return render_template("help.html")

@app.route("/inbox")
def inbox():
    if "user" not in session:
        return redirect(url_for("login"))
    if "gmail_user" not in session:
        return redirect(url_for("gmail_auth"))

    emails = fetch_emails(session["gmail_user"], session["gmail_pass"])
    return render_template("inbox.html", emails=emails, username=session["user"])



def fetch_emails(gmail_user, gmail_pass, n_last=5):
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        imap.login(gmail_user, gmail_pass)

        results = []

        for folder_name, folder_label in [("INBOX", "Inbox"), ("[Gmail]/Spam", "Spam Folder")]:
            imap.select(folder_name)

            _, ids = imap.search(None, "ALL")
            id_list = ids[0].split()


            for eid in reversed(id_list):
                _, data = imap.fetch(eid, "(RFC822)")
                msg  = email.message_from_bytes(data[0][1])

                # Subject
                subj, enc = decode_header(msg["Subject"])[0]
                if isinstance(subj, bytes):
                    subj = subj.decode(enc or "utf‑8", errors="ignore")

                # From
                sender = msg.get("From")

                # Body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain" and part.get_payload(decode=True):
                            body = part.get_payload(decode=True).decode(errors="ignore")
                            break
                else:
                    if msg.get_payload(decode=True):
                        body = msg.get_payload(decode=True).decode(errors="ignore")

                # Predict
                text = subj + " " + body
                vec = vectorizer.transform([text])
                pred = model.predict([text])[0]
                label = "Spam" if pred == 1 else "Not Spam"

                # Misclassification detection
                mismatch = False
                if folder_label == "Spam Folder" and label == "Not Spam":
                    mismatch = True  # False positive
                if folder_label == "Inbox" and label == "Spam":
                    mismatch = True  # False negative

                results.append({
                    "from": sender,
                    "subject": subj,
                    "body": (body[:200] + "...") if body else "(no text)",
                    "label": label,
                    "source": folder_label,
                    "mismatch": mismatch
                })

        imap.logout()
        return results

    except Exception as e:
        print("IMAP error:", e)
        return []



@app.route('/admin_dashboard')
def admin_dashboard():
    if 'admin' not in session:
        return redirect('/admin_login')
    return render_template('admin_dashboard.html')  # ✅ Create this template



@app.route('/admin/users')
def manage_users():
    if 'admin' not in session:
        return redirect('/admin_login')

    conn = sqlite3.connect('users.db')
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users")
    users = cur.fetchall()
    conn.close()
    return render_template('manage_users.html', users=users)


@app.route('/user_management')
def user_management():
    if 'admin' not in session:
        return redirect('/admin_login')

    conn = sqlite3.connect('users.db')
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users")
    users = cur.fetchall()
    conn.close()
    return render_template('manage_users.html', users=users)


@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'admin' not in session:
        return redirect('/admin_login')

    conn = sqlite3.connect('users.db')
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return redirect('/user_management')  # 👈 Redirect to new route



"""@app.route('/admin/analytics')
def admin_analytics():
    if 'admin' not in session:
        return redirect('/admin_login')

    conn = sqlite3.connect('users.db')
    cur = conn.cursor()

    # 🧑 User stats
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    # 📄 Load model training logs if you save them
    # For now, we'll simulate model stats
    model_info = {
        'accuracy': '93.2%',   # Example value
        'trained_on': 'June 27, 2025',
        'dataset_size': 5000,
        'model_path': 'spam_model.pkl'
    }

    # 📊 Simulated detection stats
    detection_stats = {
        'total_predictions': 8421,
        'spam_detected': 3521,
        'ham_detected': 4900,
        'false_positives': 32,
        'false_negatives': 17
    }

    return render_template('analytics.html',
                           total_users=total_users,
                           model_info=model_info,
                           detection_stats=detection_stats)


"""
@app.route('/admin/analytics')
def admin_analytics():
    if 'admin' not in session:
        return redirect('/admin_login')

    conn = sqlite3.connect('users.db')
    cur = conn.cursor()

    # Total users (real value from DB)
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    # ✅ Dummy values (you can change them anytime)
    active_users_today = 2
    new_users_week = 1

    model_info = {
        'accuracy': '93.2%',
        'trained_on': 'June 27, 2025',
        'dataset_size': 5000,
        'model_path': 'spam_model.pkl'
    }

    detection_stats = {
        'total_predictions': 8421,
        'spam_detected': 3521,
        'ham_detected': 4900,
        'false_positives': 32,
        'false_negatives': 17
    }

    return render_template('analytics.html',
                           total_users=total_users,
                           active_users_today=active_users_today,
                           new_users_week=new_users_week,
                           model_info=model_info,
                           detection_stats=detection_stats)

@app.route('/download_csv', methods=['POST'])
def download_csv():
    data = base64.b64decode(request.form['csv_data'])
    return Response(data, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=predicted.csv"})

# ─────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
