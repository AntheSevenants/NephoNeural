from sklearn.manifold import MDS

class DimMds(DimensionReductionTechnique):
    def reduce(self, data):
        self.mds = MDS(random_state=0, **self.settings) # TODO: Euclidean or manhattan distances?
        
        return self.mds.fit_transform(data)