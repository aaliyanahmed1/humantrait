# Humin-Trait-Analysis
This repository contains a Python-based project with several components including an application script, upload functionality, and machine learning model.

## Project Structure

- `app.py`: Main application script.
- `upload.py`: Script for handling file uploads.
- `fairface.pt`: Pre-trained model file.
- `prediction.json`: JSON file for storing predictions.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Docker configuration for containerizing the application.
- `.dockerignore` and `.gitignore`: Ignore files for Docker and Git.

## Setup

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Run the main application:
  ```bash
  python app.py
  ```

- Use `upload.py` to handle file uploads.

## Docker

Build and run the Docker container:
```bash
docker build -t new-test-app .
docker run -p 8000:8000 new-test-app
```

## Notes

- The project uses a pre-trained model `fairface.pt` for predictions.
- `prediction.json` stores the output predictions.

## Project Explanation and Functionality

This project is a Python-based facial analysis application that leverages a pre-trained machine learning model to analyze images. The main functionalities include:

- Handling image uploads through the `upload.py` script.
- Processing uploaded images using the pre-trained model `fairface.pt` to predict facial attributes such as age, gender, and ethnicity.
- Storing the prediction results in `prediction.json` for further use or display.
- The main application logic is contained in `app.py`, which coordinates the workflow and serves as the entry point.
- The project supports containerization via Docker for easy deployment.

This setup allows users to upload images, have them analyzed automatically, and retrieve the results efficiently.

