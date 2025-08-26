RESUME PARSER PROJECT
====================

Overview
--------
The Resume Parser Project is a Python-based application designed to extract and classify information from resumes (CVs). It supports uploading resumes, parsing key details (name, email, phone, skills, education, experience), and classifying candidates for specific departments or job roles using machine learning models. The project uses a Flask web interface, ML models for classification, and utility functions for parsing resumes. It also supports testing and containerization via Docker.

Directory Structure
-------------------
RESUME_PARSER_PROJECT/
│
├── __pycache__/                  # Compiled Python files (auto-generated)
├── .github/                      # GitHub workflows and CI/CD actions
├── models/                       # Folder for trained ML models
├── static/                       # Static frontend assets
│   ├── css/style.css
│   └── js/script.js
├── templates/                    # HTML templates
│   └── index.html
├── uploads/                      # Uploaded resumes storage
├── app.py                        # Main Flask application
├── classification.py             # Resume classification logic
├── Combined_Fake_CV_Dataset_5000_Entries.csv  # Training dataset
├── config.py                     # Global configuration settings
├── department_config.py          # Department-specific configuration
├── Dockerfile                    # Docker setup
├── ml_model.py                   # ML model definition and training
├── models.py                     # Database or ML model structures
├── parser_utils.py               # Resume parsing utility functions
├── requirements.txt              # Python dependencies
├── test_app.py                   # Integration tests for Flask app
├── test_directly.py              # Unit tests for functions
├── test_parser.py                # Tests for parser utilities
└── test_resume.txt               # Sample resume for testing

Installation
------------
1. Clone the repository:
   git clone <repository_url>
   cd RESUME_PARSER_PROJECT

2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate      # On Linux/Mac
   venv\Scripts\activate         # On Windows

3. Install dependencies:
   pip install -r requirements.txt

4. Run the application:
   python app.py

5. Open a browser and go to http://127.0.0.1:5000 to access the web interface.

Usage
-----
1. Upload a resume file (TXT, PDF, DOCX) via the web interface.
2. The application parses the resume using parser_utils.py.
3. Extracted details are displayed on the frontend.
4. ML classification predicts candidate suitability for departments or roles.
5. Results can optionally be stored in a database.

Testing
-------
- Run unit tests for parser utilities:
  python test_parser.py
- Run tests for application routes:
  python test_app.py
- Run direct function tests:
  python test_directly.py

Docker
------
1. Build the Docker image:
   docker build -t resume-parser .

2. Run the container:
   docker run -p 5000:5000 resume-parser

3. Access the app at http://localhost:5000.

Dependencies
------------
- Flask
- pandas
- scikit-learn
- numpy
- PyPDF2 (for PDF parsing)
- python-docx (for DOCX parsing)

All dependencies are listed in requirements.txt.

Notes
-----
- Replace Combined_Fake_CV_Dataset_5000_Entries.csv with your own dataset for production use.
- Update department_config.py to configure department-specific requirements and skills.
- Ensure uploads/ folder has proper write permissions.

Authors
-------
- Project maintained by [Your Name / Team]
- University/Organization: [Optional]

License
-------
This project is licensed under the MIT License. See LICENSE file for details.
