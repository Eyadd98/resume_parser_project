from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from parser_utils import extract_name, extract_education, predict_department, predict_sub_department
import spacy

# Initialize spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

# Test data
test_text = """
cloud computing

Skills:
- Java
- Javascript
- Web Development
- Git

Email: eyad.theking@gmail.com
Phone: 962786197825
Education:
Bachelor's Degree Applied University Bachelor's Degree in
"""

def test_parser():
    print("\nTesting Resume Parser with Sample Data...")
    print("-" * 50)

    # Test name extraction
    name = extract_name(test_text, nlp)
    print(f"\nName Extraction:")
    print(f"Found Name: {name}")

    # Test education extraction
    education = extract_education(test_text, nlp)
    print(f"\nEducation Extraction:")
    print(f"Found Education: {education}")

    # Test skills (using basic extraction)
    skills = ['Java', 'Javascript', 'Web Development', 'Git']
    print(f"\nSkills:")
    print(f"Found Skills: {skills}")

    # Test department classification
    department = predict_department(test_text, skills)
    print(f"\nDepartment Classification:")
    print(f"Predicted Department: {department}")

    # Test sub-department classification
    sub_department = predict_sub_department(test_text, skills, department)
    print(f"\nSub-Department Classification:")
    print(f"Predicted Sub-Department: {sub_department}")

if __name__ == "__main__":
    test_parser()
