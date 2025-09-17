SpamShield is an intelligent spam detection system built using Machine Learning and Deep Learning techniques. It classifies emails, text messages, and multimedia inputs into Spam or Ham (Not Spam). The system is deployed as a Flask web application with a simple and user-friendly interface for real-time detection.
🚀 Features

✅ Detects spam in text messages and emails

✅ Supports CSV upload for bulk detection

✅ Detects spam in images using OCR (Optical Character Recognition)

✅ Detects spam in audio files using speech-to-text conversion

✅ Real-time classification using ML/DL models

✅ Web-based UI built with Flask + Bootstrap

🛠️ Tech Stack

Programming Language: Python

Framework: Flask

Libraries & Tools:

TensorFlow / Keras (Deep Learning Models)

Scikit-learn (ML Algorithms)

Pandas, NumPy (Data Processing)

Matplotlib (Data Visualization – Admin Side)

OpenCV + OCR (Image Spam Detection)

SpeechRecognition (Audio Spam Detection)

📂 Project Structure
SpamShield/
│── app.py                # Flask main application
│── train_model.py        # Model training script
│── static/               # CSS, JS, Images
│── templates/            # HTML files (Bootstrap-based UI)
│── models/               # Saved ML/DL models (.h5 / .pkl)
│── uploads/              # Uploaded files (CSV, Images, Audio)
│── requirements.txt      # Required dependencies
│── README.md             # Project documentation

⚙️ Requirements

Before running the project, install the dependencies:

pip install -r requirements.txt

Main Libraries Required

Python 3.8+

Flask

TensorFlow / Keras

Scikit-learn

Pandas, NumPy

Matplotlib

OpenCV

SpeechRecognition

PyAudio (for audio handling)

▶️ How to Run

Clone this repository:

git clone https://github.com/your-username/SpamShield.git
cd SpamShield


Install dependencies:

pip install -r requirements.txt


Train the model (optional, if not using pre-trained model):

python train_model.py


Run the Flask app:

python app.py


Open in your browser:

http://127.0.0.1:5000/

📊 Admin Dashboard (Optional)

Upload CSV datasets and train new models

View spam/ham distribution with charts

Save trained models for future use

📌 Future Enhancements

Integration with Gmail/Outlook API for live spam detection in inbox

Improved deep learning models (LSTM, BERT)

Cloud deployment (AWS/Heroku)



![Homepage](https://github.com/latha-shree/SpamShield/blob/main/Homepage.png)
![register](https://github.com/latha-shree/SpamShield/blob/main/register.png)
![admin_login](https://github.com/latha-shree/SpamShield/blob/main/admin_login.png)
![model](https://github.com/latha-shree/SpamShield/blob/main/model.png)
![user_management](https://github.com/latha-shree/SpamShield/blob/main/user_management.png)
![system_analysis](https://github.com/latha-shree/SpamShield/blob/main/systen_analystcs.png)
![user_reg](https://github.com/latha-shree/SpamShield/blob/main/reg.png)
![user_login](https://github.com/latha-shree/SpamShield/blob/main/user_login.png)
![text_spam](https://github.com/latha-shree/SpamShield/blob/main/text_spam.png)
![image](https://github.com/latha-shree/SpamShield/blob/main/image_spam.png)
![img](https://github.com/latha-shree/SpamShield/blob/main/image-result.png)

