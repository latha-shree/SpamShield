import pyttsx3
import os

# Create folders for saving audio files
os.makedirs("audio_samples/spam", exist_ok=True)
os.makedirs("audio_samples/ham", exist_ok=True)

# --- Spam messages (10) ---
spam_texts = [
    "Congratulations! You won a free iPhone. Click the link now.",
    "Your account has been suspended. Verify it immediately.",
    "Claim your lottery prize before it expires.",
    "This is your final notice for car insurance renewal.",
    "Win a brand new car! Enter your details today.",
    "You've been selected for a free vacation to Dubai.",
    "Get your credit score increased instantly. Apply now.",
    "Unlock your cash prize by confirming your email.",
    "Exclusive offer for you only. Limited time deal.",
    "Your OTP is 999999. Do not share it with anyone."
]

# --- Ham (normal) messages (10) ---
ham_texts = [
    "Hey, are we still on for dinner tonight?",
    "Please send me the project files before 5 PM.",
    "I reached the office. Talk to you later.",
    "Let's catch up over coffee this weekend.",
    "Don't forget to pick up groceries on the way.",
    "Happy birthday! Hope you have a wonderful day.",
    "Call me when you're free. I have some news.",
    "Meeting is rescheduled to 3 PM tomorrow.",
    "Can you help me with my homework later?",
    "I’m booking the tickets for the movie tonight."
]

# Initialize pyttsx3 engine
engine = pyttsx3.init()

def save_audio(text, filepath):
    engine.save_to_file(text, filepath)

# Generate spam audios
for i, text in enumerate(spam_texts, 1):
    path = f"audio_samples/spam/spam{i}.wav"
    save_audio(text, path)

# Generate ham audios
for i, text in enumerate(ham_texts, 1):
    path = f"audio_samples/ham/ham{i}.wav"
    save_audio(text, path)

engine.runAndWait()
print("✅ 10 spam and 10 ham audio files generated in 'audio_samples' folder.")
