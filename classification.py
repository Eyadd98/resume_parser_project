def predict_department(self, text, skills):
    """Predict department based on resume text and extracted skills"""
    from department_config import DEPARTMENT_MAPPING
    
    # Convert everything to lowercase for matching
    text = text.lower()
    skills = [s.lower() for s in skills]
    
    # Calculate scores for each department
    department_scores = {}
    
    for dept, config in DEPARTMENT_MAPPING.items():
        score = 0
        # Check keywords in text
        for keyword in config['keywords']:
            if keyword in text:
                score += 2  # Higher weight for department keywords
        
        # Check skills
        for skill in skills:
            # Check if skill is related to department keywords
            if any(keyword in skill or skill in keyword for keyword in config['keywords']):
                score += 1
            
            # Check sub-departments
            for sub_dept, sub_keywords in config['sub_departments'].items():
                if any(keyword in skill or skill in keyword for keyword in sub_keywords):
                    score += 0.5
        
        department_scores[dept] = score
    
    # Get department with highest score
    if not department_scores:
        return "Unknown"
    
    max_score = max(department_scores.values())
    if max_score == 0:
        return "Unknown"
    
    return max(department_scores.items(), key=lambda x: x[1])[0]

def predict_sub_department(self, text, skills, department):
    """Predict sub-department based on resume text, skills, and predicted department"""
    from department_config import DEPARTMENT_MAPPING
    
    if department == "Unknown" or department not in DEPARTMENT_MAPPING:
        return "Unknown"
    
    # Convert everything to lowercase for matching
    text = text.lower()
    skills = [s.lower() for s in skills]
    
    # Get sub-departments for the predicted department
    sub_departments = DEPARTMENT_MAPPING[department]['sub_departments']
    
    # Calculate scores for each sub-department
    sub_dept_scores = {}
    
    for sub_dept, keywords in sub_departments.items():
        score = 0
        # Check keywords in text
        for keyword in keywords:
            if keyword in text:
                score += 2
        
        # Check skills
        for skill in skills:
            if any(keyword in skill or skill in keyword for keyword in keywords):
                score += 1
        
        sub_dept_scores[sub_dept] = score
    
    # Get sub-department with highest score
    if not sub_dept_scores:
        return "General"
    
    max_score = max(sub_dept_scores.values())
    if max_score == 0:
        return "General"
    
    return max(sub_dept_scores.items(), key=lambda x: x[1])[0]
