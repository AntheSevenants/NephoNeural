from .model import Model

class ModelCollection:
    def __init__(self):
        self.models = {}
        
        # Distance matrices (over models)
        self.mds_distance_matrix = None
        self.tsne_distance_matrix = None
        
    def register_model(self, model):
        if not type(model) == Model:
            raise Exception("Model input should be of type \"Model\"")
        
        self.models[model.name] = model
    
    def get_models(self):
        return self.models
    
    def get_model_names(self):
        return list(self.models.keys())
