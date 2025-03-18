# Django Face Search - README

## Project Overview
Django Face Search is a web application that allows users to upload images and search for similar faces in a dataset using facial recognition. The application leverages OpenCV, FAISS, and InsightFace for efficient face detection and matching.

## Features
- Upload images and detect faces.
- Search for similar faces using FAISS indexing.
- Store face embeddings for efficient retrieval.
- User-friendly Django interface.
- Dockerized for easy deployment.

## Tech Stack
- **Backend**: Django (Python)
- **Face Recognition**: OpenCV, FAISS, InsightFace
- **Database**:  SQLite
- **Deployment**: Docker, Gunicorn

## Installation & Setup

### 1. Clone the Repository
```bash
 git clone https://github.com/your-repo/django-face-search.git
 cd django-face-search
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Migrations
```bash
python manage.py migrate
```

### 5. Run the Development Server
```bash
python manage.py runserver
```
Access the application at: **http://127.0.0.1:8000/**

---

## Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t django-face-search .
```

### 2. Run the Docker Container
```bash
docker run -p 8000:8000 django-face-search
```

Or using Docker Compose:
```bash
docker-compose up --build
```

Access the application at: **http://localhost:8000/**

---

## API Endpoints

| Method | Endpoint           | Description                  |
|--------|-------------------|------------------------------|
| POST   | /upload/          | Upload an image for search  |
| GET    | /results/         | Retrieve matched faces      |
| GET    | /health-check/    | Check server health         |

---

## Environment Variables
Create a `.env` file in the root directory and set the following variables:
```env
DEBUG=True
DATABASE_URL=postgres://user:password@localhost:5432/dbname
```

---

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push to the branch: `git push origin feature-branch`.
5. Open a pull request.

---

## License
This project is licensed under the MIT License.

 

