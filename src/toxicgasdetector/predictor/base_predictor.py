from .helpers import metric
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
import pickle


class Pipeline:

    def __init__(self, model, data_filepath='', output_dir='',):
        self.model = model
        self.data_filepath = Path(data_filepath)
        self.output_dir = Path(output_dir)
    
    def load_data(self):
        X = pd.read_csv(self.data_filepath / 'X_train.csv', index_col='ID').values
        y = pd.read_csv(self.data_filepath / 'y_train.csv', index_col='ID').values
        return X, y
    
    def setup_save_file(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir(exist_ok=True)
    
    def save_job_infos(self):
        model_class_name = self.model.__class__.__name__
        model_bases =  [str(base.__module__) for base in self.model.__class__.__bases__]
        job_info = {
            'Model Class Name': model_class_name,
            'Model Bases': model_bases,
            'Data Filepath': str(self.data_filepath),
            'Output Directory': str(self.output_dir),}
        if pd.Series(model_bases).str.contains('sklearn').any():
            #TODO : for sklearn, save parameters and pickle the model.
            job_info['Python ML Module'] = 'sklearn'
            with open(self.output_dir / 'model_trained.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
        
        elif pd.Series(model_bases).str.contains('torch').any():
            #TODO : for torch models, save only the model with the torch method.
            torch.save(self.model, self.output_dir / "model_trained.pt")
            job_info['Python ML Module'] = 'torch'

        with open(self.output_dir / 'job_info.json', 'w') as f:
            json.dump(job_info, f)
        
    
    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return metric(y, self.predict(X))
