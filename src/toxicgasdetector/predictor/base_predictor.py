from .helpers import metric


class Pipeline:

    def __init__(self, model):
        self.model = model
    
    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return metric(y, self.predict(X))
