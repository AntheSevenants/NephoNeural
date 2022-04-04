class DimensionReductionTechnique:
    def __init__(self, name, settings=None):
        self.name = name
        self.settings = settings
        
    def reduce(self, data):
        raise Exception("Reduction not implemented. Please override this method.")

    def reduce_model(self, data):
        return self.reduce(data)