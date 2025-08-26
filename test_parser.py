import requests
import json

# Test resume content
resume_text = """
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

# Save this as a text file first
with open('test_resume.txt', 'w', encoding='utf-8') as f:
    f.write(resume_text)

# Convert text file to PDF using fpdf
from fpdf import FPDF

def text_to_pdf(txt_path, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    for line in text.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    
    pdf.output(pdf_path)

# Create PDF
text_to_pdf('test_resume.txt', 'test_resume.pdf')

# Test the parser
def test_resume_parser():
    url = 'http://localhost:5000/upload'
    files = {'cvFile': ('test_resume.pdf', open('test_resume.pdf', 'rb'), 'application/pdf')}
    
    response = requests.post(url, files=files)
    print("\nParser Response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_resume_parser()
