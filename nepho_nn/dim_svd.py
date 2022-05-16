import numpy as np
from nepho_nn.dimension_reduction_technique import DimensionReductionTechnique

class DimSvd(DimensionReductionTechnique):
    def reduce(self, data):
        return np.linalg.svd(data)[1]