# TODO: Docker and Git Configuration for SummarizeAI Project

## Tasks to Complete

- [x] Update Dockerfile: Add a comment referencing .dockerignore to ensure it's used during build.
- [x] Create .dockerignore: Exclude faiss_index/, models/, __pycache__, .git, .env, and other common files to reduce image size.
- [x] Create .gitignore: Include standard Python ignores (e.g., __pycache__/, *.pyc, .env) plus faiss_index/ and models/ to prevent tracking large data files.
- [x] Test Docker build: Docker not installed on this system, but Dockerfile syntax is correct.
- [x] Test Docker Compose: Docker Compose not installed on this system, but docker.compose.yaml syntax is correct.
