from flask import Flask, render_template, request
import CaseBasedSystem as cbs

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        animal_age = int(request.form.get('animal_age', 0))
        animal_sex = request.form.get('animal_sex')
        environmental_conditions = request.form.get('environmental_conditions')

        new_case = {
            'Symptoms': symptoms,
            'Animal Age (Months)': animal_age,
            'Animal Sex': animal_sex,
            'Environmental Conditions': environmental_conditions
        }

    # Define the weights dictionary
    weights = {
        'Symptoms': 0.6,
        'Animal Age (Months)': 0.2,
        'Environmental Conditions': 0.2
    }

    similarity_threshold = 0.5
    case_database = cbs.load_case_database('FMD cases.csv')
    similar_cases = cbs.retrieve_similar_cases(
        new_case, case_database, similarity_threshold, top_n=3)

    if similar_cases:
        diagnosis, treatment = cbs.diagnose_and_treat(
            new_case, similar_cases)
        prognosis = cbs.predict_prognosis(new_case, similar_cases)
    else:
        overall_similarity = cbs.calculate_overall_similarity(
            new_case, case_database, weights)
        if overall_similarity < similarity_threshold:
            diagnosis, treatment = cbs.diagnose_and_treat(
                new_case, similar_cases)
            outcome = "Not determined yet"
            case_database_updated = cbs.update_case_database(
                case_database, new_case, diagnosis, treatment, outcome,
                similarity_threshold)

            if case_database_updated != case_database:
                print("No update to the case database.")
            else:
                cbs.save_case_database(
                    case_database_updated, 'FMD cases.csv')
                print("Case database updated with the new case.")
        else:
            print("Similarity score below threshold. New case not added.")

        diagnosis = "Unknown"
        treatment = ["Unknown"]
        prognosis = "Unable to predict prognosis."

    return render_template(
        'result.html', diagnosis=diagnosis,
        treatment=treatment, prognosis=prognosis)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
