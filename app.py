import os
import io
import re
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
import torch
import pytesseract
from sentence_transformers import SentenceTransformer

# Import custom modules
from models import setup_database, ProcessedResume
from ml_model import TransformerClassifier
from config import Config

# Try importing optional libraries
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    from spacy.matcher import Matcher
except (ImportError, OSError):
    print("Warning: spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")
    nlp = None
    Matcher = None

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))
except ImportError:
    print("Warning: NLTK not available. Install with: pip install nltk")
    STOPWORDS = set()

try:
    import pdfplumber
    PDF_AVAILABLE = True
    PDF_BACKEND = 'plumber'
except ImportError:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        PDF_AVAILABLE = True
        PDF_BACKEND = 'miner'
    except ImportError:
        print("Warning: PDF processing not available. pip install pdfplumber or pdfminer.six")
        PDF_AVAILABLE = False
        PDF_BACKEND = None

# Flask app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize Flask app and load configuration
app = Flask(__name__)
app.config.from_object('config.Config')

# Initialize parser as part of the application context
resume_parser = None

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class ResumeParser:
    def __init__(self, db_session=None):
        # Initialize NLP components
        self.matcher = Matcher(nlp.vocab) if nlp and Matcher else None
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Initialize ML models
        self.dept_model = None
        self.sub_dept_model = None
        
        # Database session for continuous learning
        self.db_session = db_session
        
        # Initialize components
        self._setup_skill_patterns()
        self._load_or_train_models()
        
        # Cache for embeddings and skills
        self._skill_embeddings = {}
        self._known_skills = set()
        self._last_training_time = datetime.now()
        
    def _setup_skill_patterns(self):
        """Setup enhanced patterns for skill extraction with continuous learning"""
        if not self.matcher:
            return
            
        # Initialize skill categories
        self.skill_categories = {
            'programming': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 
                'php', 'swift', 'kotlin', 'rust', 'go', 'scala', 'perl'
            ],
            'web_dev': [
                'html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'django',
                'flask', 'spring', 'asp.net', 'laravel', 'graphql', 'rest'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 
                'elasticsearch', 'cassandra', 'dynamodb', 'neo4j'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform',
                'ansible', 'circleci', 'github actions', 'gitlab ci'
            ],
            'ai_ml': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
                'scikit-learn', 'pandas', 'numpy', 'opencv', 'nlp'
            ]
        }
        
        # Load known skills from database if available
        if self.db_session:
            stored_skills = self.db_session.query(SkillKeyword).all()
            for category, skills in self.skill_categories.items():
                for skill in skills:
                    self._known_skills.add(skill)
                    self._compute_skill_embedding(skill)
            
            for skill in stored_skills:
                self._known_skills.add(skill.skill_name)
                if skill.variations:
                    variations = json.loads(skill.variations)
                    for var in variations:
                        self._known_skills.add(var)
        
        # Set up spaCy patterns
        if self.matcher:
            for category, skills in self.skill_categories.items():
                for skill in skills:
                    words = skill.lower().split()
                    pattern = [{"LOWER": word} for word in words]
                    self.matcher.add("SKILL", [pattern])
                    
    def _compute_skill_embedding(self, skill):
        """Compute and cache embedding for a skill"""
        if skill not in self._skill_embeddings:
            self._skill_embeddings[skill] = self.sentence_model.encode(skill)
        return self._skill_embeddings[skill]
    
    def _find_similar_skills(self, text, threshold=0.85):
        """Find similar skills using semantic similarity"""
        text_embedding = self.sentence_model.encode(text)
        similar_skills = []
        
        for skill in self._known_skills:
            skill_embedding = self._compute_skill_embedding(skill)
            similarity = np.dot(text_embedding, skill_embedding) / \
                        (np.linalg.norm(text_embedding) * np.linalg.norm(skill_embedding))
            if similarity > threshold:
                similar_skills.append((skill, similarity))
        
        return sorted(similar_skills, key=lambda x: x[1], reverse=True)
    
    def _learn_new_skill(self, skill, frequency=1):
        """Add new skill to the database"""
        if not self.db_session:
            return
            
        skill = skill.lower().strip()
        existing = self.db_session.query(SkillKeyword).filter_by(skill_name=skill).first()
        
        if existing:
            existing.frequency += frequency
            existing.last_seen = datetime.utcnow()
        else:
            new_skill = SkillKeyword(
                skill_name=skill,
                variations=json.dumps([]),
                frequency=frequency
            )
            self.db_session.add(new_skill)
        
        self.db_session.commit()
        self._known_skills.add(skill)
        self._compute_skill_embedding(skill)
        
        for pattern in skill_patterns:
            self.matcher.add("SKILL", [pattern])

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            return "PDF processing not available. Please install pdfplumber."
            
        try:
            # Try pdfplumber first (more reliable)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    return text
            except:
                # Fallback to pdfminer
                return self._extract_with_pdfminer(pdf_path)
                
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""

    def _extract_with_pdfminer(self, pdf_path):
        """Fallback PDF extraction using pdfminer"""
        try:
            with open(pdf_path, 'rb') as file:
                resource_manager = PDFResourceManager()
                fake_file_handle = io.StringIO()
                converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
                page_interpreter = PDFPageInterpreter(resource_manager, converter)
                
                for page in PDFPage.get_pages(file, caching=True, check_extractable=True):
                    page_interpreter.process_page(page)
                
                text = fake_file_handle.getvalue()
                converter.close()
                fake_file_handle.close()
                return text
        except Exception as e:
            print(f"Error with pdfminer extraction: {str(e)}")
            return ""

    def get_email_addresses(self, text):
        """Extract email addresses from text"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)

    def get_phone_numbers(self, text):
        """Extract phone numbers from text"""
        # Convert Arabic-Indic digits to ASCII
        text = text.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
        
        # Match international-ish formats
        pattern = r'(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?){2,4}\d{2,4}'
        matches = re.findall(pattern, text)
        
        cleaned = []
        for m in matches:
            d = re.sub(r'\D', '', m)
            if len(d) >= 7:
                cleaned.append(d)
                
        # Deduplicate
        return list(dict.fromkeys(cleaned))

    def extract_name(self, text):
        """Extract name from resume text using enhanced methods"""
        # Look for common resume header patterns first
        first_lines = text.split('\n')[:5]  # Check first 5 lines
        
        if nlp:
            # Try spaCy NER first
            for line in first_lines:
                line = line.strip()
                if line and len(line.split()) <= 4:  # Most names are 1-4 words
                    doc = nlp(line)
                    for ent in doc.ents:
                        if ent.label_ == "PERSON":
                            return ent.text
            
            # Try processing first 1000 chars
            doc = nlp(text[:1000])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text
        
        # Fallback methods if spaCy fails or isn't available
        for line in first_lines:
            line = line.strip()
            if line:
                words = line.split()
                # Check if it looks like a name (2-4 capitalized words)
                if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words):
                    return line
                
        # Last resort - get first non-empty line
        for line in first_lines:
            if line.strip():
                words = line.strip().split()
                if len(words) >= 2:
                    return ' '.join(words[:2])
                
        return "Name not found"
        
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        
        # Fallback: look for proper nouns in first sentence
        sentences = list(doc.sents)
        if sentences:
            first_sentence = sentences[0]
            proper_nouns = [token.text for token in first_sentence if token.pos_ == "PROPN"]
            if len(proper_nouns) >= 2:
                return " ".join(proper_nouns[:2])
            elif proper_nouns:
                return proper_nouns[0]
        return "Name not found"

    def extract_skills(self, text):
        """Extract skills from resume text"""
        if not nlp or not self.matcher:
            # Simple regex-based fallback
            common_skills = [
                'python', 'java', 'javascript', 'sql', 'tensorflow', 'pytorch',
                'machine learning', 'data science', 'web development', 'react',
                'angular', 'node.js', 'docker', 'kubernetes', 'aws', 'azure', 'git'
            ]
            found_skills = []
            text_lower = text.lower()
            for skill in common_skills:
                if skill in text_lower:
                    found_skills.append(skill.title())
            return found_skills

        doc = nlp(text.lower())
        found_skills = set()
        
        # Use NER for technology-related entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"] and len(ent.text) > 2:
                found_skills.add(ent.text.title())
        
        # Use matcher for specific skill patterns
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            found_skills.add(span.text.title())
    
        return list(found_skills)

    def extract_education(self, text):
        """Extract education information"""
        if not nlp:
            return ["Education extraction requires NLP"]
        
        education_info = []
        doc = nlp(text)
        
        # Define comprehensive education patterns
        degree_patterns = {
            'bachelors': [
                r'(?i)b\.?\s*(?:tech|sc|a|e|ed)|bachelor\'?s?\s+(?:of|in|degree)',
                r'(?i)undergraduate\s+degree',
                r'(?i)bachelor(?:\'s)?\s+(?:of|in)\s+[a-zA-Z\s]+',
            ],
            'masters': [
                r'(?i)m\.?\s*(?:tech|sc|a|e|ed)|master\'?s?\s+(?:of|in|degree)',
                r'(?i)master(?:\'s)?\s+(?:of|in)\s+[a-zA-Z\s]+',
            ],
            'phd': [
                r'(?i)ph\.?d\.?|doctorate|doctor\s+of\s+philosophy',
            ]
        }
        
        text_blocks = text.split('\n')
        found_education = set()
        
        for block in text_blocks:
            # Check for institution names using NER
            block_doc = nlp(block)
            for ent in block_doc.ents:
                if ent.label_ == "ORG" and any(word in ent.text.lower() 
                    for word in ['university', 'college', 'institute', 'school']):
                    found_education.add(ent.text.strip())
            
            # Check for degree patterns
            for level, patterns in degree_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, block, re.IGNORECASE)
                    if matches:
                        # Try to get the full degree name with major
                        full_degree = re.search(
                            fr"{pattern}\s+(?:in|of)?\s+([A-Za-z\s]+)(?:\.|$)", 
                            block, 
                            re.IGNORECASE
                        )
                        if full_degree and full_degree.group(1):
                            found_education.add(f"{level.title()} in {full_degree.group(1).strip()}")
                        else:
                            found_education.add(f"{level.title()} Degree")
        
        education_list = list(found_education)
        # Sort by importance (PhD > Masters > Bachelors)
        education_list.sort(key=lambda x: (
            'phd' in x.lower(),
            'master' in x.lower(),
            'bachelor' in x.lower()
        ), reverse=True)
        
        return education_list

    def create_sample_data(self):
        """Create sample training data if CSV is not available"""
        sample_data = {
            'Skills': [
                'Python, Machine Learning, TensorFlow, Data Analysis',
                'Java, Spring Boot, Microservices, REST APIs',
                'JavaScript, React, Node.js, MongoDB',
                'SQL, Database Design, Data Warehousing, ETL',
                'AWS, Docker, Kubernetes, DevOps',
                'C++, Algorithms, Data Structures, System Design',
                'HTML, CSS, JavaScript, Frontend Development',
                'Python, Django, PostgreSQL, Backend Development'
            ],
            'Department': [
                'Data Science', 'Software Development', 'Web Development', 
                'Data Engineering', 'DevOps', 'Software Development',
                'Web Development', 'Software Development'
            ],
            'Sub-Department': [
                'Machine Learning', 'Backend Development', 'Frontend Development',
                'Data Analytics', 'Cloud Engineering', 'Systems Programming',
                'UI/UX Development', 'Full Stack Development'
            ]
        }
        return pd.DataFrame(sample_data)

    def preprocess_skills(self, skills_text):
        """Preprocess skills text for better prediction"""
        # Convert to string if not already
        if not isinstance(skills_text, str):
            skills_text = str(skills_text)
        
        # Normalize text
        skills_text = skills_text.lower()
        
        # Replace variations
        replacements = {
            'javascript': ['js', 'ecmascript'],
            'python': ['py', 'python3'],
            'java': ['java8', 'java11', 'jdk'],
            'machine learning': ['ml', 'machine-learning'],
            'artificial intelligence': ['ai'],
            'typescript': ['ts'],
            'react': ['reactjs', 'react.js'],
            'node': ['nodejs', 'node.js'],
        }
        
        for standard, variants in replacements.items():
            for variant in variants:
                skills_text = skills_text.replace(variant, standard)
        
        return skills_text

    def _load_or_train_models(self):
        """Initialize or load pre-trained models with enhanced error handling"""
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        try:
            # Try to load existing models
            dept_model_path = os.path.join(models_dir, 'dept_model.pkl')
            sub_dept_model_path = os.path.join(models_dir, 'sub_dept_model.pkl')
            
            if os.path.exists(dept_model_path) and os.path.exists(sub_dept_model_path):
                print("Loading pre-trained models...")
                with open(dept_model_path, 'rb') as f:
                    self.dept_model = pickle.load(f)
                with open(sub_dept_model_path, 'rb') as f:
                    self.sub_dept_model = pickle.load(f)
                print("Loaded pre-trained models successfully")
                return
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Will train new models...")
        
        self.train_classification_model()

    def train_classification_model(self):
        """Train classification models with continuous learning support"""
        try:
            # Combine historical and new data
            data = self._get_training_data()
            
            if data.empty:
                print("No training data available")
                return None, None
            
            # Prepare features with improved processing
            X = data['Skills'].fillna('').apply(self._prepare_skill_features)
            y_dept = data['Department'].fillna('Unknown')
            y_sub_dept = data.get('Sub-Department', data['Department']).fillna('Unknown')
            
            # Initialize transformers-based classifiers
            n_dept_classes = len(set(y_dept))
            n_sub_dept_classes = len(set(y_sub_dept))
            
            self.dept_classifier = TransformerClassifier(n_classes=n_dept_classes)
            self.sub_dept_classifier = TransformerClassifier(n_classes=n_sub_dept_classes)

            # Split data
            X_train, X_test, y_dept_train, y_dept_test = train_test_split(
                X, y_dept, test_size=0.2, random_state=42
            )
            X_train_sub, X_test_sub, y_sub_dept_train, y_sub_dept_test = train_test_split(
                X, y_sub_dept, test_size=0.2, random_state=42
            )

            # Create improved pipeline with better feature engineering
            self.dept_model = make_pipeline(
                TfidfVectorizer(
                    max_features=2000,
                    ngram_range=(1, 2),  # Include bigrams
                    stop_words='english',
                    min_df=2,  # Ignore terms that appear in less than 2 documents
                    max_df=0.95  # Ignore terms that appear in more than 95% of documents
                ),
                MultinomialNB(alpha=0.1)  # Reduced smoothing for better precision
            )
            
            self.sub_dept_model = make_pipeline(
                TfidfVectorizer(
                    max_features=2000,
                    ngram_range=(1, 2),
                    stop_words='english',
                    min_df=2,
                    max_df=0.95
                ),
                MultinomialNB(alpha=0.1)
            )

            # Train models
            self.dept_model.fit(X_train, y_dept_train)
            self.sub_dept_model.fit(X_train_sub, y_sub_dept_train)

            # Evaluate models with probability scores
            dept_pred = self.dept_model.predict(X_test)
            dept_proba = self.dept_model.predict_proba(X_test)
            sub_dept_pred = self.sub_dept_model.predict(X_test_sub)
            sub_dept_proba = self.sub_dept_model.predict_proba(X_test_sub)

            print("Department Classification Report:")
            print(classification_report(y_dept_test, dept_pred))
            print("\nSub-Department Classification Report:")
            print(classification_report(y_sub_dept_test, sub_dept_pred))

            return self.dept_model, self.sub_dept_model

        except Exception as e:
            print(f"Error training models: {str(e)}")
            return None, None

    def predict_department_and_sub_dept(self, skills_text):
        """Predict department and sub-department from skills with enhanced accuracy"""
        if not self.dept_model or not self.sub_dept_model:
            self._load_or_train_models()  # Try loading/training models
            if not self.dept_model or not self.sub_dept_model:
                return "Unknown", "Unknown", 0.0, 0.0
        
        try:
            # Preprocess the input skills
            processed_skills = self.preprocess_skills(skills_text)
            if not processed_skills.strip():
                return "Unknown", "Unknown", 0.0, 0.0
            
            # Get department predictions and probabilities
            dept_prediction = self.dept_model.predict([processed_skills])[0]
            dept_probas = self.dept_model.predict_proba([processed_skills])[0]
            dept_proba = max(dept_probas)
            dept_idx = np.argmax(dept_probas)
            
            # Get sub-department predictions and probabilities
            sub_dept_prediction = self.sub_dept_model.predict([processed_skills])[0]
            sub_dept_probas = self.sub_dept_model.predict_proba([processed_skills])[0]
            sub_dept_proba = max(sub_dept_probas)
            sub_dept_idx = np.argmax(sub_dept_probas)
            
            # Define confidence thresholds
            LOW_CONFIDENCE = 0.3
            MEDIUM_CONFIDENCE = 0.5
            HIGH_CONFIDENCE = 0.7
            # Apply confidence-based logic for department
            if dept_proba < LOW_CONFIDENCE:
                dept_prediction = "Unknown"
            elif dept_proba < MEDIUM_CONFIDENCE:
                dept_prediction = f"Possible {dept_prediction}"
            elif dept_proba < HIGH_CONFIDENCE:
                # Get second best prediction if close
                sorted_indices = np.argsort(dept_probas)[::-1]
                if len(sorted_indices) > 1:
                    second_best_prob = dept_probas[sorted_indices[1]]
                    if dept_proba - second_best_prob < 0.2:  # Close predictions
                        dept_prediction = f"{dept_prediction} / {self.dept_model.classes_[sorted_indices[1]]}"
            
            # Apply confidence-based logic for sub-department
            if sub_dept_proba < LOW_CONFIDENCE:
                sub_dept_prediction = "General"
            elif sub_dept_proba < MEDIUM_CONFIDENCE:
                sub_dept_prediction = f"Possible {sub_dept_prediction}"
            elif sub_dept_proba < HIGH_CONFIDENCE:
                # Get second best prediction if close
                sorted_indices = np.argsort(sub_dept_probas)[::-1]
                if len(sorted_indices) > 1:
                    second_best_prob = sub_dept_probas[sorted_indices[1]]
                    if sub_dept_proba - second_best_prob < 0.2:  # Close predictions
                        sub_dept_prediction = f"{sub_dept_prediction} / {self.sub_dept_model.classes_[sorted_indices[1]]}"
            
            return dept_prediction, sub_dept_prediction, float(dept_proba), float(sub_dept_proba)
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return "Unknown", "Unknown", 0.0, 0.0

# Initialize parser and train models on startup
parser = ResumeParser()
print("Training classification models...")
parser.train_classification_model()
print("Models trained successfully!")

# Initialize parser at module level
def init_parser():
    """Initialize the resume parser and train models"""
    global resume_parser
    if resume_parser is None:
        resume_parser = ResumeParser()
        print("Training classification models...")
        resume_parser.train_classification_model()
        print("Models trained successfully!")

# Initialize parser when the module loads
init_parser()

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'cvFile' not in request.files:
            return jsonify({'error': 'No CV file uploaded'}), 400
        
        file = request.files['cvFile']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Please upload a PDF file only'}), 400

        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        try:
            # Extract text from PDF
            text = resume_parser.extract_text_from_pdf(filepath)
            if not text or not text.strip():
                return jsonify({'error': 'Could not extract text from PDF. The file might be scanned/image-based.'}), 400
            
            # Import improved parsing utilities
            from parser_utils import extract_name, extract_education, predict_department, predict_sub_department
            
            # Extract information using improved utilities
            skills = resume_parser.extract_skills(text)
            name = extract_name(text, nlp)
            emails = resume_parser.get_email_addresses(text)
            phones = resume_parser.get_phone_numbers(text)
            education = extract_education(text, nlp)
            
            # Predict department and sub-department using improved classification
            department = predict_department(text, skills)
            sub_department = predict_sub_department(text, skills, department)
            
            # Keep file temporarily for debugging
            # os.remove(filepath)  # Uncomment in production
            
            return jsonify({
                'name': name,
                'skills': skills,
                'emails': emails,
                'phones': phones,
                'education': education,
                'predicted_department': department,
                'predicted_sub_department': sub_department,
                'department_confidence': 100,  # Using rule-based system now
                'sub_department_confidence': 100,
                'extracted_text_length': len(text)
            })
            
        finally:
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        parser = ResumeParser()
        text = parser.extract_text_from_pdf(filepath)
        
        if not text:
            return jsonify({'error': 'Could not extract text from PDF. The file might be scanned/image-based.'}), 400
            
        result = {
            'name': parser.extract_name(text),
            'emails': parser.get_email_addresses(text),
            'phones': parser.get_phone_numbers(text),
            'skills': parser.extract_skills(text),
            'education': parser.extract_education(text),
            'extracted_text_length': len(text)
        }
        
        # Keep file temporarily for debugging
        # os.remove(filepath)  # Uncomment in production
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize parser before starting the app
    init_parser()
    
    # In production, set debug=False and use environment variables for host/port
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')  # More secure default
    port = int(os.environ.get('FLASK_PORT', 5000))
    app.run(debug=debug, host=host, port=port)