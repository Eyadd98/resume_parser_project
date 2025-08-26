import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
import numpy as np

class ResumeClassifier(nn.Module):
    def __init__(self, n_classes, bert_model='distilbert-base-uncased'):
        super(ResumeClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # Take CLS token output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes, model_name='distilbert-base-uncased', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_classes = n_classes
        self.model_name = model_name
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = ResumeClassifier(n_classes, model_name).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit the model to the training data"""
        self.classes_ = np.unique(y)
        
    def _get_training_data(self):
        """Get combined training data from CSV and database"""
        data_sources = []
        
        # 1. Load CSV data
        csv_path = 'Combined_Fake_CV_Dataset_5000_Entries.csv'
        if os.path.exists(csv_path):
            csv_data = pd.read_csv(csv_path)
            # Clean and preprocess the data
            csv_data = csv_data.dropna(subset=['Skills', 'Department', 'Sub-Department'])
            csv_data['Skills'] = csv_data['Skills'].str.lower()
            csv_data['Department'] = csv_data['Department'].fillna('Unknown')
            csv_data['Sub-Department'] = csv_data['Sub-Department'].fillna('General')
            data_sources.append(csv_data)
        
        # 2. Load data from database
        if self.db_session:
            stored_resumes = self.db_session.query(ProcessedResume).all()
            if stored_resumes:
                db_data = pd.DataFrame([{
                    'Skills': resume.skills,
                    'Department': resume.predicted_department,
                    'Sub-Department': resume.predicted_sub_department,
                    'Confidence': resume.department_confidence,
                    'Feedback': resume.feedback_correct
                } for resume in stored_resumes])
                
                # Filter high-confidence predictions with positive feedback
                db_data = db_data[
                    (db_data['Confidence'] > 0.8) & 
                    (db_data['Feedback'] >= 0)
                ]
                data_sources.append(db_data)
        
        # Combine and deduplicate data
        if data_sources:
            combined_data = pd.concat(data_sources, ignore_index=True)
            return combined_data.drop_duplicates(subset=['Skills'])
        
        return pd.DataFrame()

    def _prepare_skill_features(self, skills_text):
        """Prepare skill features with embedding support"""
        if isinstance(skills_text, str):
            # Get embeddings for skills
            skills = skills_text.split(',')
            skill_embeddings = []
            
            for skill in skills:
                skill = skill.strip().lower()
                if skill in self._skill_embeddings:
                    skill_embeddings.append(self._skill_embeddings[skill])
                else:
                    embedding = self.sentence_model.encode(skill)
                    skill_embeddings.append(embedding)
            
            if skill_embeddings:
                return np.mean(skill_embeddings, axis=0)
                
        return None
        # Convert labels to numeric
        unique_labels = np.unique(y)
        self.classes_ = unique_labels
        label_map = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_map[label] for label in y])
        
        # Prepare data
        encodings = self.tokenizer(X.tolist(), truncation=True, padding=True, return_tensors='pt')
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'].to(self.device),
            encodings['attention_mask'].to(self.device),
            torch.tensor(numeric_labels).to(self.device)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Training loop
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(3):  # 3 epochs by default
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
        return self
    
    def predict(self, X):
        self.model.eval()
        encodings = self.tokenizer(X.tolist(), truncation=True, padding=True, return_tensors='pt')
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return np.array([self.classes_[pred] for pred in predictions])
    
    def predict_proba(self, X):
        self.model.eval()
        encodings = self.tokenizer(X.tolist(), truncation=True, padding=True, return_tensors='pt')
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probas = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probas
