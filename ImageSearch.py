from flask import Flask, request, render_template, redirect, url_for
import os
from google.cloud import vision
from google.auth import load_credentials_from_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def annotate(path: str, quota_project_id: str) -> vision.WebDetection:
    creds, project = load_credentials_from_file(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    creds = creds.with_quota_project(quota_project_id)

    client = vision.ImageAnnotatorClient(credentials=creds)

    if path.startswith("http") or path.startswith("gs:"):
        image = vision.Image()
        image.source.image_uri = path
    else:
        with open(path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

    web_detection = client.web_detection(image=image).web_detection
    return web_detection

def report(annotations: vision.WebDetection) -> dict:
    results = {}

    if annotations.pages_with_matching_images:
        results['pages_with_matching_images'] = [
            page.url for page in annotations.pages_with_matching_images
        ]

    if annotations.full_matching_images:
        results['full_matching_images'] = [
            image.url for image in annotations.full_matching_images
        ]

    if annotations.partial_matching_images:
        results['partial_matching_images'] = [
            image.url for image in annotations.partial_matching_images
        ]

    if annotations.web_entities:
        results['web_entities'] = [
            {
                'score': entity.score,
                'description': entity.description
            } for entity in annotations.web_entities
        ]

    return results

@app.route('/')
def index():
    return render_template('imagesearch.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        quota_project_id = request.form['quota_project_id']
        annotations = annotate(filepath, quota_project_id)
        results = report(annotations)
        return render_template('imageresults.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
