# App1.py

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from google.cloud import vision
from google.auth import load_credentials_from_file
from CaseBasedSystem import (
    load_case_database, calculate_overall_similarity,
    retrieve_similar_cases, diagnose_and_treat, predict_prognosis,
    update_case_database, save_case_database, fetch_unknown_diagnosis_cases,
    update_case
)
import RetreivalAugmentedGeneration
import ModelInference

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Swafey/Documents/work/project-2/Project/vision-application-426219-627c55c923bb.json"

QUOTA_PROJECT_ID = 'vision-application-426219'

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
            } for entity in annotations.web_entities if entity.score > 0.7
        ]

    return results

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/CBR system')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query')
    model_no = int(data.get('model_no'))

    if model_no == 1:
        answer = ModelInference.llm_chain.invoke(input=f"{user_query}")
        answer = answer['text']
    else:
        answer = RetreivalAugmentedGeneration.LLM_Run(str(user_query))

    return jsonify({'response': answer})

@app.route('/submit', methods=['POST'])
def submit():
    symptoms = request.form['symptoms'].split(',')
    animal_age = int(request.form['animal_age'])
    animal_sex = request.form['animal_sex']
    environmental_conditions = request.form['environmental_conditions']

    new_case = {
        'Symptoms': symptoms,
        'Animal Age (Months)': animal_age,
        'Animal Sex': animal_sex,
        'Environmental Conditions': environmental_conditions
    }

    file_path = 'FMD cases.csv'

    if os.path.exists(file_path):
        case_database = load_case_database(file_path)
    else:
        return render_template(
            'result.html', diagnosis="Error: Case database not found.",
            treatment=[], prognosis="N/A", similar_cases=[])

    # Debugging: Print the new case and loaded case database
    print("New Case:", new_case)
    print("Loaded Case Database:", list(case_database.items())[:5])

    weights = {
        'Symptoms': 0.6,
        'Animal Age (Months)': 0.2,
        'Environmental Conditions': 0.2
    }

    similarity_threshold = 0.5
    similar_cases = retrieve_similar_cases(
        new_case, case_database, similarity_threshold, top_n=3)

    # Debugging: Print similar cases
    print("Similar Cases:", similar_cases)

    if similar_cases:
        diagnosis, treatment = diagnose_and_treat(new_case, similar_cases)
        prognosis = predict_prognosis(new_case, similar_cases)
    else:
        overall_similarity = calculate_overall_similarity(
            new_case, case_database, weights)

        # Debugging: Print overall similarity
        print("Overall Similarity:", overall_similarity)

        if overall_similarity < similarity_threshold:
            diagnosis = "No similar cases found."
            treatment = []
            prognosis = "N/A"
            outcome = "Not determined yet"
            case_database_updated = update_case_database(
                case_database, new_case, diagnosis, treatment, outcome,
                similarity_threshold)
            save_case_database(case_database_updated, file_path)
        else:
            diagnosis = "Similar case found but below threshold."
            treatment = []
            prognosis = "N/A"

    return render_template('result.html', diagnosis=diagnosis,
                           treatment=treatment, prognosis=prognosis,
                           similar_cases=similar_cases)

@app.route('/unknown_cases')
def unknown_cases():
    file_path = 'FMD cases.csv'
    if os.path.exists(file_path):
        case_database = load_case_database(file_path)
        unknown_cases = fetch_unknown_diagnosis_cases(case_database)
        return render_template('unknown_cases.html', cases=unknown_cases)
    else:
        return render_template('unknown_cases.html', cases={})

@app.route('/edit_case/<case_id>', methods=['GET', 'POST'])
def edit_case(case_id):
    file_path = 'FMD cases.csv'
    if os.path.exists(file_path):
        case_database = load_case_database(file_path)
        case = case_database.get(case_id)

        if request.method == 'POST':
            diagnosis = request.form['diagnosis']
            treatment = request.form['treatment'].split(',')
            outcome = request.form['outcome']
            case_database_updated = update_case(
                case_database, case_id, diagnosis, treatment, outcome)
            save_case_database(case_database_updated, file_path)
            return redirect(url_for('unknown_cases'))

        return render_template('update_case.html', case=case, case_id=case_id)
    else:
        return render_template('update_case.html', case=None, case_id=case_id)


@app.route('/imagesearch')
def index_imagesearch():
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
        annotations = annotate(filepath, QUOTA_PROJECT_ID)
        results = report(annotations)
        return render_template('imageresults.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)

