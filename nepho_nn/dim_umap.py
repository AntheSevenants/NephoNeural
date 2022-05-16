import umap
from nepho_nn.dimension_reduction_technique import DimensionReductionTechnique

class DimUmap(DimensionReductionTechnique):
    def reduce(self, data):
        self.umap = umap.UMAP()
        
        return self.umap.fit_transform(data)