# Content-Based Video Retrieval  
A video segment retrieval system developed for a university course at the University of Klagenfurt.

---

## System Requirements

- **Python** 3.11.x (Mac users may need 3.10.x due to dependency issues)
- **FFmpeg** (for video processing)
- **PostgreSQL** (required as DB backend)

---

## Install FFmpeg

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

### macOS (via Homebrew)
```bash
brew install ffmpeg
```

### Windows
1. Download the "release full" ZIP from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
2. Extract to: `C:\ffmpeg\`
3. Add `C:\ffmpeg\bin` to your system `PATH`
4. Verify with:
   ```bash
   ffmpeg -version
   ffprobe -version
   ```

---

## Install PostgreSQL

### Windows
1. Download from: [https://www.postgresql.org/download/windows/](https://www.postgresql.org/download/windows/)
2. During setup:
   - Remember your **postgres password**
   - You only need to install the **server**, not pgAdmin
3. After install, launch the **SQL Shell (psql)** to create DB and user:
   ```sql
   CREATE USER video_user WITH PASSWORD 'your_secure_password';
   CREATE DATABASE videos OWNER video_user;
   ```

4. Grant permissions (optional):
   ```sql
   GRANT ALL PRIVILEGES ON DATABASE videos TO video_user;
   ```

### macOS (via Homebrew)
```bash
brew install postgresql
brew services start postgresql

# Create user and database
psql postgres
```
Then in the SQL prompt:
```sql
CREATE USER video_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE videos OWNER video_user;
GRANT ALL PRIVILEGES ON DATABASE videos TO video_user;
```

---

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib

# Switch to the postgres user
sudo -i -u postgres
psql
```
Then in the SQL prompt:
```sql
CREATE USER video_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE videos OWNER video_user;
GRANT ALL PRIVILEGES ON DATABASE videos TO video_user;
```
Type `\q` to exit `psql`, then `exit` to return to your normal user shell.

---

## Setup

### 1. Clone the repo:
```bash
git clone https://github.com/Kirschzelle/ContentBasedVideoRetrieval.git
cd ContentBasedVideoRetrieval
```

### 2. Install Python dependencies:

#### Windows:
```bash
pip install -r requirements_win.txt
```

#### macOS:
```bash
pip install -r requirements_mac.txt
```

### 3. Update `settings.py` with your PostgreSQL DB config:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'videos',
        'USER': 'video_user',
        'PASSWORD': 'your_secure_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

---

## Video Dataset

Download a subset of V3C-1 dataset from:  
https://www2.itec.aau.at/owncloud/index.php/s/4uqMvVtZEJSY7O8  
Password: `IVADL2025`

Place videos in:

```
./data/videos/
```

---

## Usage

### 1. Prepare DB
```bash
python manage.py makemigrations
python manage.py migrate
```

### 2. Import and process videos
```bash
python manage.py import_videos
python manage.py extract_clips
python manage.py extract_keyframes
```

### 3. Optional: Run the server
```bash
python manage.py runserver
```