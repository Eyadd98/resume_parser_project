"""Resume classification functions"""
from department_config import calculate_department_score, calculate_sub_department_score

def predict_department(text, skills):
    """Predict the most likely department based on resume text and skills"""
    # Calculate scores for each department
    scores = calculate_department_score(text, skills)
    
    if not scores:
        return "Unknown"
    
    # Get the department with the highest score
    max_score = max(scores.values())
    if max_score == 0:
        return "Unknown"
    
    return max(scores.items(), key=lambda x: x[1])[0]

def predict_sub_department(text, skills, department):
    """Predict the most likely sub-department based on resume text, skills, and department"""
    if department == "Unknown":
        return "Unknown"
        
    # Calculate scores for sub-departments
    scores = calculate_sub_department_score(text, skills, department)
    
    if not scores:
        return "General"
    
    # Get the sub-department with the highest score
    max_score = max(scores.values())
    if max_score == 0:
        return "General"
    
    return max(scores.items(), key=lambda x: x[1])[0]

def extract_name(text, nlp=None):
    """Extract name from resume text using improved pattern matching"""
    import re
    
    # Common name prefixes and titles to help identify names
    titles = r'(?:mr\.?|mrs\.?|ms\.?|dr\.?|prof\.?|eng\.?)'
    
    # Try to find name with common resume patterns
    patterns = [
        # Look for name after explicit headers
        r'(?i)name\s*[:]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
        # Look for name at the start of the resume
        r'^[\s\n]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})[\s\n]*$',
        # Look for name with titles
        f'(?i){titles}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{1,2}})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            name = match.group(1).strip()
            # Validate that it's not a technical term
            if not any(tech_term in name.lower() for tech_term in [
                'resume', 'cv', 'curriculum', 'vitae', 'cloud', 'computing',
                'software', 'developer', 'engineer'
            ]):
                return name
    
    # If we have spaCy available, try NER
    if nlp:
        # Only process the first 1000 characters for efficiency
        doc = nlp(text[:1000])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                # Validate that it's not a technical term
                if not any(tech_term in name.lower() for tech_term in [
                    'resume', 'cv', 'curriculum', 'vitae', 'cloud', 'computing',
                    'software', 'developer', 'engineer'
                ]):
                    return name
    
    return "Name not found"

def extract_education(text, nlp=None):
    """Extract education information using improved pattern matching"""
    import re
    
    education_info = []
    
    # Define comprehensive education patterns
    patterns = {
        'degree_level': [
            r"(?i)(?:bachelor'?s?|b\.?(?:tech|sc|a|e)|undergraduate)\s+(?:degree\s+)?(?:of|in)?\s+([^,\n]+)",
            r"(?i)(?:master'?s?|m\.?(?:tech|sc|a|e)|postgraduate)\s+(?:degree\s+)?(?:of|in)?\s+([^,\n]+)",
            r"(?i)(?:ph\.?d\.?|doctorate|doctor\s+of)\s+(?:in\s+)?([^,\n]+)",
        ],
        'university': [
            r"(?i)(?:from|at|graduated\s+from)\s+([^,\n]+(?:university|institute|college)[^,\n]+)",
            r"(?i)([^,\n]+(?:university|institute|college)[^,\n]+)",
        ]
    }
    
    # Process text by lines to maintain context
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for degrees
        for pattern in patterns['degree_level']:
            matches = re.finditer(pattern, line)
            for match in matches:
                degree = match.group(1).strip()
                if len(degree) > 3:  # Avoid very short matches
                    education_info.append(degree)
        
        # Check for universities
        for pattern in patterns['university']:
            matches = re.finditer(pattern, line)
            for match in matches:
                university = match.group(1).strip()
                if len(university) > 5:  # Avoid very short matches
                    education_info.append(university)
    
    # Use spaCy if available
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG" and any(edu_term in ent.text.lower() 
                for edu_term in ['university', 'college', 'institute', 'school']):
                education_info.append(ent.text.strip())
    
    # Remove duplicates while maintaining order
    seen = set()
    return [x for x in education_info if not (x.lower() in seen or seen.add(x.lower()))]
