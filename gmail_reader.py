# gmail_reader.py (new helper module for Gmail spam detection)
import imaplib, email
from email.header import decode_header

# Load model and vectorizer
import joblib
model_data = joblib.load('model/spam_model.pkl')
model = model_data['model']
vectorizer = model_data['vectorizer']

def fetch_emails(gmail_user, gmail_pass):
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
                msg = email.message_from_bytes(data[0][1])

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

                # Prediction
                text = subj + " " + body
                vec = vectorizer.transform([text])
                pred = model.predict(vec)[0]
                label = "Spam" if pred == 1 else "Not Spam"

                mismatch = False
                if folder_label == "Spam Folder" and label == "Not Spam":
                    mismatch = True
                if folder_label == "Inbox" and label == "Spam":
                    mismatch = True

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
