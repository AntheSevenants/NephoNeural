from sklearn.manifold import MDS
from nepho_nn.dimension_reduction_technique import DimensionReductionTechnique

class DimMds(DimensionReductionTechnique):
    def reduce(self, data):
        self.mds = MDS(random_state=0, **self.settings) # TODO: Euclidean or manhattan distances?
        
        return self.mds.fit_transform(data)