from sklearn.manifold import TSNE

class DimTsne(DimensionReductionTechnique):
    def reduce(self, data):
        self.tsne = TSNE(random_state=0, **self.settings) # TODO: what perplexity?
        
        return self.tsne.fit_transform(data)