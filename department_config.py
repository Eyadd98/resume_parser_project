"""Department and skill classification configuration"""

DEPARTMENT_MAPPING = {
    'Software Development': {
        'keywords': ['software development', 'programming', 'coding', 'software engineering'],
        'sub_departments': {
            'Backend Development': ['backend', 'server-side', 'api development', 'microservices'],
            'Frontend Development': ['frontend', 'ui development', 'web development', 'react', 'angular'],
            'Full Stack': ['full stack', 'fullstack', 'end-to-end', 'full-stack'],
            'Mobile Development': ['android', 'ios', 'mobile app', 'react native', 'flutter'],
        }
    },
    'DevOps': {
        'keywords': ['devops', 'ci/cd', 'infrastructure', 'deployment'],
        'sub_departments': {
            'Cloud Infrastructure': ['aws', 'azure', 'gcp', 'cloud computing'],
            'Platform Engineering': ['kubernetes', 'docker', 'containerization'],
            'Site Reliability': ['sre', 'monitoring', 'observability'],
        }
    },
    'Data Science': {
        'keywords': ['data science', 'machine learning', 'analytics', 'statistics'],
        'sub_departments': {
            'Machine Learning': ['ml', 'deep learning', 'neural networks'],
            'Data Analytics': ['data analysis', 'business intelligence', 'analytics'],
            'Research': ['research', 'nlp', 'computer vision'],
        }
    },
    'Security': {
        'keywords': ['security', 'cybersecurity', 'infosec', 'information security'],
        'sub_departments': {
            'Application Security': ['appsec', 'secure coding', 'security testing'],
            'Network Security': ['network security', 'firewall', 'penetration testing'],
            'Security Operations': ['secops', 'incident response', 'threat analysis'],
        }
    },
    'Quality Assurance': {
        'keywords': ['qa', 'quality assurance', 'testing', 'quality control'],
        'sub_departments': {
            'Test Automation': ['automated testing', 'selenium', 'test automation'],
            'Manual Testing': ['manual testing', 'user testing', 'test cases'],
            'Performance Testing': ['performance testing', 'load testing', 'stress testing'],
        }
    }
}

# Additional patterns for skill matching
SKILL_PATTERNS = {
    'programming_languages': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust',
        'php', 'scala', 'kotlin', 'swift', 'r', 'matlab'
    ],
    'web_technologies': [
        'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'express', 'django',
        'flask', 'spring', 'asp.net', 'laravel', 'jquery', 'bootstrap'
    ],
    'databases': [
        'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'elasticsearch',
        'cassandra', 'dynamodb', 'neo4j', 'sqlite'
    ],
    'cloud_platforms': [
        'aws', 'azure', 'gcp', 'heroku', 'digitalocean', 'openstack', 'alibaba cloud',
        'ibm cloud', 'oracle cloud'
    ],
    'tools': [
        'git', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'prometheus',
        'grafana', 'jira', 'confluence', 'bitbucket', 'gitlab'
    ]
}
