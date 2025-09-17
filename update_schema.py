import sqlite3

# Connect to your database
conn = sqlite3.connect('users.db')
cur = conn.cursor()

# Add created_at column
try:
    cur.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
    print("✅ 'created_at' column added.")
except sqlite3.OperationalError:
    print("⚠️ 'created_at' column already exists.")

# Add last_login column
try:
    cur.execute("ALTER TABLE users ADD COLUMN last_login TEXT")
    print("✅ 'last_login' column added.")
except sqlite3.OperationalError:
    print("⚠️ 'last_login' column already exists.")

conn.commit()
conn.close()
