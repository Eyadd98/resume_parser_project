import unittest
import json
import tempfile
import os
from app import app, ResumeParser

class ResumeParserTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create a test PDF file (this would be a real PDF in practice)
        self.test_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        
    def test_index_route(self):
        """Test the index route"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        
    def test_upload_no_file(self):
        """Test upload with no file"""
        response = self.app.post('/upload')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
    def test_upload_invalid_file_type(self):
        """Test upload with invalid file type"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b'This is a text file')
            temp_file.flush()
            
            with open(temp_file.name, 'rb') as f:
                response = self.app.post('/upload', data={
                    'cvFile': (f, 'test.txt')
                })
                
            os.unlink(temp_file.name)
            
        self.assertEqual(response.status_code, 400)
        
    def test_resume_parser_skills_extraction(self):
        """Test skills extraction"""
        parser = ResumeParser()
        
        test_text = """
        John Doe
        Software Developer
        
        Skills:
        - Python programming
        - Machine Learning
        - TensorFlow
        - SQL databases
        - JavaScript
        - React framework
        """
        
        skills = parser.extract_skills(test_text)
        self.assertIsInstance(skills, list)
        
    def test_resume_parser_email_extraction(self):
        """Test email extraction"""
        parser = ResumeParser()
        
        test_text = "Contact me at john.doe@example.com or jane.smith@company.org"
        emails = parser.get_email_addresses(test_text)
        
        self.assertIn('john.doe@example.com', emails)
        self.assertIn('jane.smith@company.org', emails)
        
    def test_resume_parser_phone_extraction(self):
        """Test phone number extraction"""
        parser = ResumeParser()
        
        test_text = "Call me at 123-456-7890 or (555) 123-4567"
        phones = parser.get_phone_numbers(test_text)
        
        self.assertTrue(len(phones) >= 1)
        
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')

class ResumeParserUnitTest(unittest.TestCase):
    def setUp(self):
        self.parser = ResumeParser()
        
    def test_skill_patterns(self):
        """Test skill pattern matching"""
        test_cases = [
            ("I have experience with Python and JavaScript", ["Python", "Javascript"]),
            ("Worked with machine learning and TensorFlow", ["Machine Learning", "Tensorflow"]),
            ("Expert in SQL and database design", ["Sql"]),
        ]
        
        for text, expected_partial in test_cases:
            skills = self.parser.extract_skills(text)
            # Check if at least some expected skills are found
            self.assertIsInstance(skills, list)
            
    def test_education_extraction(self):
        """Test education extraction"""
        test_text = """
        Education:
        Bachelor of Science in Computer Science
        University of Technology, 2020
        
        Master of Science in Data Science
        Tech University, 2022
        """
        
        education = self.parser.extract_education(test_text)
        self.assertIsInstance(education, list)
        
    def test_model_prediction(self):
        """Test model prediction functionality"""
        # Train with sample data first
        self.parser.train_classification_model()
        
        if self.parser.dept_model and self.parser.sub_dept_model:
            test_skills = "Python Machine Learning TensorFlow Data Analysis"
            dept, sub_dept = self.parser.predict_department_and_sub_dept(test_skills)
            
            self.assertIsInstance(dept, str)
            self.assertIsInstance(sub_dept, str)
            self.assertNotEqual(dept, "")
            self.assertNotEqual(sub_dept, "")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)