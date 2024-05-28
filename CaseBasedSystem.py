#!/usr/bin/env python
# coding: utf-8

# # **1.DATA PRE-PROCESSING.**
#

# In[1]:


import csv
import difflib
from collections import Counter, defaultdict
import os
# import sys
# from datetime import datetime


file_path = 'FMD cases.csv'


def preprocess_dataset(file_path):
    """
    Preprocess the dataset by converting it into a dictionary format.

    Args:
        file_path (str): The path to the CSV file containing the dataset.

    Returns:
        dict: A dictionary representing the dataset, where each case is a
        dictionary containing the input parameters,
              diagnosis, treatment, and outcome.
    """
    case_database = {}

    # Open the CSV file
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)

        # Iterate over each row (case) in the CSV file
        for row in reader:
            case_id = row['Case ID']
            symptoms = row['Symptoms'].split(', ')
            # Split the symptoms into a list
            age = int(row['Animal Age (Months)'])
            sex = row['Animal Sex']
            environmental_conditions = row['Environmental Conditions']
            diagnosis = row['Diagnosis']
            treatment = row['Treatment'].split(', ')
            # Split the treatment into a list
            outcome = row['Outcome']

            # Create a dictionary for the current case
            case = {
                'Symptoms': symptoms,
                'Animal Age (Months)': age,
                'Animal Sex': sex,
                'Environmental Conditions': environmental_conditions,
                'Diagnosis': diagnosis,
                'Treatment': treatment,
                'Outcome': outcome
            }

            # Add the case to the case database
            case_database[case_id] = case

    return case_database


# Example usage
case_database = preprocess_dataset(file_path)

print("Case Database:")
print(case_database)

# Print each case individually
print("\nIndividual Cases:")
for case_id, case in case_database.items():
    print(f"\nCase ID: {case_id}")
    for key, value in case.items():
        print(f"{key}: {value}")


# # **2.CALCULATE SIMILARITY MEASURES.**

# In[2]:

def calculate_symptom_similarity(new_symptoms, existing_symptoms):
    """
    Calculate the similarity between the symptoms of a
    new case and an existing case.

    Args:
        new_symptoms (list): A list of symptoms for the new case.
        existing_symptoms (list): A list of symptoms for an existing case.

    Returns:
        float: A similarity score between 0 and 1, where 1
        indicates an exact match.
    """
    new_symptom_set = set(new_symptoms)
    existing_symptom_set = set(existing_symptoms)

    # Calculate the ratio of common symptoms
    common_symptoms = new_symptom_set.intersection(existing_symptom_set)
    symptom_similarity = len(common_symptoms) / max(
        len(new_symptom_set), len(existing_symptom_set), 1)

    return symptom_similarity


def calculate_age_similarity(new_age, existing_age):
    """
    Calculate the similarity between the
    ages of a new case and an existing case.

    Args:
        new_age (int): The age (in months) of the new case.
        existing_age (int): The age (in months) of an existing case.

    Returns:
        float: A similarity score between 0 and 1,
        where 1 indicates an exact match.
    """
    max_age = max(new_age, existing_age)
    age_difference = abs(new_age - existing_age)
    age_similarity = 1 - (age_difference / max_age) if max_age > 0 else 1

    return age_similarity


def calculate_environmental_similarity(new_conditions, existing_conditions):
    """
    Calculate the similarity between the environmental
    conditions of a new case and an existing case.

    Args:
        new_conditions (str): A string describing the environmental
        conditions for the new case.
        existing_conditions (str): A string describing the environmental
        conditions for an existing case.

    Returns:
        float: A similarity score between 0 and 1
        where 1 indicates an exact match.
    """
    sequence_matcher = difflib.SequenceMatcher(
        None, new_conditions, existing_conditions)
    environmental_similarity = sequence_matcher.ratio()

    return environmental_similarity


def calculate_overall_similarity(new_case, existing_case, weights):
    """
    Calculate the overall similarity between a new case and an existing case.

    Args:
        new_case (dict): A dictionary representing the new case.
        existing_case (dict): A dictionary representing an existing case.
        weights (dict): A dictionary containing weights for each feature
        (symptom, age, environmental conditions).

    Returns:
        float: An overall similarity score between 0 and 1,
        where 1 indicates an exact match.
    """
    symptom_similarity = calculate_symptom_similarity(
        new_case.get('Symptoms', []), existing_case.get('Symptoms', []))
    age_similarity = calculate_age_similarity(
        new_case.get('Animal Age (Months)', 0),
        existing_case.get('Animal Age (Months)', 0))
    environmental_similarity = calculate_environmental_similarity(
        new_case.get('Environmental Conditions', ''),
        existing_case.get('Environmental Conditions', ''))

    overall_similarity = (
        weights['Symptoms'] * symptom_similarity +
        weights['Animal Age (Months)'] * age_similarity +
        weights['Environmental Conditions'] * environmental_similarity)

    return overall_similarity


# # **3. IMPLEMENTING CASE RETRIEVAL**

# In[3]:


def retrieve_similar_cases(
        new_case, case_database, similarity_threshold=0.5, top_n=3):
    """
    Retrieve the most similar cases from the case database for a given new
    case.

    Args:
        new_case (dict): A dictionary representing the new case.
        case_database (dict): A dictionary containing the existing cases.
        similarity_threshold (float): The minimum similarity score required to
        consider a case as similar.
        top_n (int): The maximum number of similar cases to retrieve.

    Returns:
        list: A list of tuples, where each tuple contains the case ID, the
        corresponding case dictionary,
              and the similarity score for the top N most similar cases.
    """
    similar_cases = defaultdict(list)

    # Calculate the similarity between the new case and each existing case
    for case_id, existing_case in case_database.items():
        overall_similarity = calculate_overall_similarity(
            new_case, existing_case, weights)  # Ensure 'weights' is defined
        if overall_similarity >= similarity_threshold:
            similar_cases[overall_similarity].append(
                (case_id, existing_case, overall_similarity))

    # Sort the similar cases by similarity score in descending order
    sorted_similar_cases = sorted(similar_cases.items(), reverse=True)

    # Retrieve the top N similar cases
    top_similar_cases = []
    for similarity_score, case_list in sorted_similar_cases:
        top_similar_cases.extend(case_list[:top_n])
        top_n -= len(case_list)
        if top_n <= 0:
            break

    return top_similar_cases


# Define the weights dictionary
weights = {
    'Symptoms': 0.6,
    'Animal Age (Months)': 0.2,
    'Environmental Conditions': 0.2
}


# # **4. DETERMINING THE DIAGNOSIS AMND TREATMENT**

# In[4]:


def diagnose_and_treat(new_case, similar_cases):
    """
    Determine the diagnosis and treatment for a new case based on the most
    similar cases.

    Args:
        new_case (dict): A dictionary representing the new case.
        similar_cases (list): A list of tuples, where each tuple contains the
        case ID, the corresponding case dictionary,
                              and the similarity score for the most similar
                              cases.

    Returns:
        tuple: A tuple containing the determined diagnosis (str) and the
        recommended treatment (list).
    """
    diagnoses = []
    treatments = []

    # Collect diagnoses and treatments from the similar cases
    for case_id, case, similarity_score in similar_cases:
        diagnoses.append(case['Diagnosis'])
        treatments.append(case['Treatment'])

    # If no similar cases were found, return default values
    if not diagnoses:
        return "Unknown", ["Unknown"]

    # Determine the most common diagnosis
    diagnosis_counter = Counter(diagnoses)
    most_common_diagnosis, _ = diagnosis_counter.most_common(1)[0]

    # Determine the most common treatment
    treatment_counter = Counter(
        [item for sublist in treatments for item in sublist])
    most_common_treatment = [
        item for item, count in treatment_counter.most_common()
        if count >= len(similar_cases) // 2]

    return most_common_diagnosis, most_common_treatment


# def retrieve_similar_cases(
#         new_case, case_database, similarity_threshold=0.5, top_n=3):
#     """
#     Retrieve the most similar cases from the case database for a given new
#     case.

#     Args:
#         new_case (dict): A dictionary representing the new case.
#         case_database (dict): A dictionary containing the existing cases.
#         similarity_threshold (float): The minimum similarity score required
#         to consider a case as similar.
#         top_n (int): The maximum number of similar cases to retrieve.

#     Returns:
#         list: A list of tuples, where each tuple contains the case ID, the
# corresponding case dictionary,
#               and the similarity score for the top N most similar cases.
#     """
#     similar_cases = defaultdict(list)

#     # Calculate the similarity between the new case and each existing case
#     for case_id, existing_case in case_database.items():
#         overall_similarity = calculate_overall_similarity(new_case,
#           existing_case, weights)
#         if overall_similarity >= similarity_threshold:
#             similar_cases[overall_similarity].append((case_id, existing_case,
#               overall_similarity))

#     # Sort the similar cases by similarity score in descending order
#     sorted_similar_cases = sorted(similar_cases.items(), reverse=True)

#     # Retrieve the top N similar cases
#     top_similar_cases = []
#     for similarity_score, case_list in sorted_similar_cases:
#         top_similar_cases.extend(case_list[:top_n])
#         top_n -= len(case_list)
#         if top_n <= 0:
#             break

#     return top_similar_cases


# # **5.PREDICTING THE PROGNOSIS**

# In[5]:


def predict_prognosis(new_case, similar_cases):
    """
    Predict the prognosis for a new case based on the outcomes of the most
    similar cases.

    Args:
        new_case (dict): A dictionary representing the new case.
        similar_cases (list): A list of tuples, where each tuple contains the
        case ID, the corresponding case dictionary,
                              and the similarity score for the most similar
                              cases.

    Returns:
        str: The predicted prognosis for the new case.
    """
    outcomes = []

    # Collect outcomes from the similar cases
    for case_id, case, similarity_score in similar_cases:
        outcomes.append(case['Outcome'])

    # If no similar cases were found, return a default prognosis
    if not outcomes:
        return "Unable to predict prognosis due to lack of similar cases."

    # Determine the most common outcome
    outcome_counter = Counter(outcomes)
    most_common_outcome, _ = outcome_counter.most_common(1)[0]

    # Determine the prognosis based on the most common outcome
    if "Recovered" in most_common_outcome:
        prognosis = "Likely to recover"
    elif "Euthanized" in most_common_outcome or "Died" in most_common_outcome:
        prognosis = "High risk of complications or mortality"
    else:
        prognosis = "Possible long-term effects or complications"

    return prognosis


# # **6.UPDATING THE CASE BASE**

# In[6]:


def update_case_database(
        case_database, new_case, diagnosis, treatment, outcome,
        similarity_threshold=0.5):
    """
    Update the case database by adding a new case and its outcome if it's
    sufficiently dissimilar to existing cases.

    Args:
        case_database (dict): The existing case database.
        new_case (dict): A dictionary representing the new case.
        diagnosis (str): The diagnosed condition for the new case.
        treatment (list): A list of treatments applied for the new case.
        outcome (str): The outcome of the new case.
        similarity_threshold (float): The minimum similarity score required
        for considering a case similar. Defaults to 0.5.

    Returns:
        dict: The updated case database with the new case added if it meets
        the similarity threshold.
    """
    # Retrieve similar cases from the case database
    similar_cases = retrieve_similar_cases(
        new_case, case_database, similarity_threshold=similarity_threshold)

    # If there are no similar cases above the threshold, add the new case
    if not similar_cases:
        # Generate a unique case ID
        num_cases = len(case_database)
        case_id = f"CASE{num_cases + 1:03d}"

        # Add the new case to the database
        new_case_entry = {
            'Case ID': case_id,
            'Symptoms': new_case['Symptoms'],
            'Animal Age (Months)': new_case['Animal Age (Months)'],
            'Animal Sex': new_case['Animal Sex'],
            'Environmental Conditions': new_case['Environmental Conditions'],
            'Diagnosis': diagnosis,
            'Treatment': (treatment),
            'Outcome': outcome
        }
        case_database[case_id] = new_case_entry

    return case_database


def save_case_database(case_database, file_path):
    """
    Save the case database to a CSV file.

    Args:
        case_database (dict): The case database to be saved.
        file_path (str): The path to the CSV file where the database will be
        saved.
    """
    fieldnames = [
        'Case ID', 'Symptoms', 'Animal Age (Months)', 'Animal Sex',
        'Environmental Conditions', 'Diagnosis', 'Treatment', 'Outcome']

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header if the file is empty or doesn't exist
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writeheader()

        # Append data to the file
        for case_id, case in case_database.items():
            writer.writerow({
                'Case ID': case_id,
                'Symptoms': ', '.join(case['Symptoms']),
                'Animal Age (Months)': case['Animal Age (Months)'],
                'Animal Sex': case['Animal Sex'],
                'Environmental Conditions': case['Environmental Conditions'],
                'Diagnosis': case['Diagnosis'],
                'Treatment': ', '.join(case['Treatment']),
                'Outcome': case['Outcome']
            })


def load_case_database(file_path):
    """
    Load the case database from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing the case database.

    Returns:
        dict: A dictionary representing the case database.
    """
    case_database = {}

    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                case_id = row['Case ID']
                symptoms = row['Symptoms'].split(', ')
                age = int(row['Animal Age (Months)'])
                sex = row['Animal Sex']
                environmental_conditions = row['Environmental Conditions']
                diagnosis = row['Diagnosis']
                treatment = row['Treatment'].split(', ')
                outcome = row['Outcome']

                case = {
                    'Symptoms': symptoms,
                    'Animal Age (Months)': age,
                    'Animal Sex': sex,
                    'Environmental Conditions': environmental_conditions,
                    'Diagnosis': diagnosis,
                    'Treatment': treatment,
                    'Outcome': outcome
                }

                case_database[case_id] = case
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")

    return case_database


# # **7.UPDATING NEW CASE DATA**

# In[7]:


def fetch_unknown_diagnosis_cases(case_database):
    """
    Fetch cases with an unknown diagnosis from the case database.

    Args:
        case_database (dict): The case database.

    Returns:
        dict: A dictionary containing cases with an unknown diagnosis, where
        the keys are case IDs and values are case dictionaries.
    """
    unknown_diagnosis_cases = {}
    for case_id, case in case_database.items():
        if case['Diagnosis'] == 'Unknown':
            unknown_diagnosis_cases[case_id] = case
    return unknown_diagnosis_cases


def update_case(case_database, case_id, diagnosis, treatment, outcome):
    """
    Update the diagnosis, treatment, and outcome for a case in the case
    database.

    Args:
        case_database (dict): The case database.
        case_id (str): The ID of the case to update.
        diagnosis (str): The updated diagnosis for the case.
        treatment (list): A list of treatments for the case.
        outcome (str): The outcome for the case.

    Returns:
        dict: The updated case database.
    """
    case = case_database.get(case_id)
    if case:
        case['Diagnosis'] = diagnosis
        case['Treatment'] = treatment
        case['Outcome'] = outcome
        case_database[case_id] = case
    return case_database


# In[8]:


# # Example usage
# file_path = 'FMD cases.csv'
# case_database = load_case_database(file_path)

# # Fetch cases with an unknown diagnosis
# unknown_diagnosis_cases = fetch_unknown_diagnosis_cases(case_database)
# print(f"Fetched case : {unknown_diagnosis_cases}")


# In[9]:


# # Update the diagnosis, treatment, and outcome for a case
# case_id = 'CASE401'  # Replace with the actual case ID
# diagnosis = 'Foot and Mouth Disease'
# treatment = ['Antibiotics', 'Quarantine', 'Supportive care']
# outcome = 'Recovered'

# case_database = update_case(case_database, case_id, diagnosis, treatment,
# outcome)

# # Save the updated case database
# save_case_database(case_database, file_path)


# # **8.IMPLEMENTING THE CODE**

# In[10]:


# Define the weights dictionary
# weights = {
#     'Symptoms': 0.6,
#     'Animal Age (Months)': 0.2,
#     'Environmental Conditions': 0.2
# }

# # Example usage
# file_path = 'FMD cases.csv'
# case_database = load_case_database(file_path)

# new_case = {
#     'Symptoms': ['fever', 'mouth lesions', 'lameness'],
#     'Animal Age (Months)': 18,
#     'Animal Sex': 'Female',
#     'Environmental Conditions': 'Livestock farm, high animal density'
# }

# similarity_threshold = 0.5
# similar_cases = retrieve_similar_cases(
#     new_case, case_database, similarity_threshold, top_n=3)

# if similar_cases:
#     diagnosis, treatment = diagnose_and_treat(new_case, similar_cases)
#     prognosis = predict_prognosis(new_case, similar_cases)

#     # print(f"Similarity_score: {overall_similarity}")
#     print(f"Diagnosis: {diagnosis}")
#     print(f"Recommended Treatment: {', '.join(treatment)}")
#     print(f"Prognosis: {prognosis}")
# else:
#     # Check if overall similarity score is above the threshold
#     overall_similarity = calculate_overall_similarity(
#         new_case, case_database, weights)
#     if overall_similarity < similarity_threshold:
#         diagnosis, treatment = diagnose_and_treat(new_case, similar_cases)
#         outcome = "Not determined yet"
#         case_database_updated = update_case_database(
#             case_database, new_case, diagnosis, treatment, outcome,
#             similarity_threshold)

#         if case_database_updated != case_database:
#             print("No update to the case database.")
#         else:
#             save_case_database(case_database_updated, file_path)
#             print("Case database updated with the new case.")
#     else:
#         print("Similarity score below threshold. New case not added.")

#     print("No similar cases found. New case added to the database.")


# In[21]:


# def get_user_input():
#     """
#     Prompt the user for input and return a dictionary representing the new
#     case.
#     """
#     symptoms = input("Enter the symptoms (comma-separated): ").split(",")
#     symptoms = [symptom.strip() for symptom in symptoms]
#     animal_age = int(input("Enter the animal age (in months): "))
#     animal_sex = input("Enter the animal sex: ")
#     environmental_conditions = input("Enter the environmental conditions: ")

#     new_case = {
#         'Symptoms': symptoms,
#         'Animal Age (Months)': animal_age,
#         'Animal Sex': animal_sex,
#         'Environmental Conditions': environmental_conditions
#     }

#     return new_case


# def main():
#     # Define the weights dictionary
#     weights = {
#         'Symptoms': 0.6,
#         'Animal Age (Months)': 0.2,
#         'Environmental Conditions': 0.2
#     }

#     # Example usage
#     file_path = 'FMD cases.csv'
#     case_database = load_case_database(file_path)

#     new_case = get_user_input()

#     similarity_threshold = 0.5
#     similar_cases = retrieve_similar_cases(
#         new_case, case_database, similarity_threshold, top_n=3)

#     if similar_cases:
#         diagnosis, treatment = diagnose_and_treat(new_case, similar_cases)
#         prognosis = predict_prognosis(new_case, similar_cases)

#         print(f"Diagnosis: {diagnosis}")
#         print(f"Recommended Treatment: {', '.join(treatment)}")
#         print(f"Prognosis: {prognosis}")
#     else:
#         # Check if overall similarity score is above the threshold
#         overall_similarity = calculate_overall_similarity(
#             new_case, case_database, weights)
#         if overall_similarity < similarity_threshold:
#             diagnosis, treatment = diagnose_and_treat(
# new_case, similar_cases)
#             outcome = "Not determined yet"
#             case_database_updated = update_case_database(
#                 case_database, new_case, diagnosis, treatment, outcome,
#                 similarity_threshold)

#             if case_database_updated != case_database:
#                 print("No update to the case database.")
#             else:
#                 save_case_database(case_database_updated, file_path)
#                 print("Case database updated with the new case.")
#         else:
#             print("Similarity score below threshold. New case not added.")

#         print("No similar cases found. New case added to the database.")


# if __name__ == "__main__":
#     main()


# In[1]:


# pip install nbconvert
