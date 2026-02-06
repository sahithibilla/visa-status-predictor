import sqlite3

def get_db():
    return sqlite3.connect("users.db")

def create_tables():
    db = get_db()
    cursor = db.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    """)
    
    # Predictions table with filing_date
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            education TEXT,
            job_experience INTEGER,
            job_training INTEGER,
            wage REAL,
            full_time TEXT,
            continent TEXT,
            experience REAL,
            filing_date TEXT,
            visa_status TEXT,
            processing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Migrate existing database - add filing_date column if not exists
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN filing_date TEXT")
        print("Added filing_date column to predictions table")
    except sqlite3.OperationalError:
        print("filing_date column already exists")
    
    db.commit()
    db.close()
