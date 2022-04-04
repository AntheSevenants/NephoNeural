from sklearn.manifold import MDS
from nepho_nn.dimension_reduction_technique import DimensionReductionTechnique

class DimMds(DimensionReductionTechnique):
    def reduce(self, data):
        self.mds = MDS(random_state=0, **self.settings) # TODO: Euclidean or manhattan distances?
        
        return self.mds.fit_transform(data)

    def reduce_model(self, data):
        self.mds_model = MDS(random_state=0,
                             dissimilarity='precomputed',
                             **self.settings) # TODO: Euclidean or manhattan distances?
        
        return self.mds_model.fit_transform(data)