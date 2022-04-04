import numpy as np

from anthevec.anthevec.embedding_retriever import EmbeddingRetriever

class Type:
    def __init__(self, lemma, sentences, layer_indices, dimension_reduction_techniques):
        print("Processing \"{}\"".format(lemma))
        self.lemma = lemma
        self.pos = "miep"
        self.source = "hallo"
        self.sentences = sentences
        self.layer_indices = layer_indices
        self.dimension_reduction_techniques = dimension_reduction_techniques
        
        self.model_collection = ModelCollection()
        
        self.get_token_vectors()
        
        # Register model names
        self.model_names = self.model_collection.get_model_names()
        
        self.do_level_3_dimension_reduction()
        self.create_similarity_matrices()
        self.create_distance_matrix()
        self.do_level_1_dimension_reduction()
        
    def get_token_vectors(self):
        print("Retrieving hidden states for all tokens...")
        
        token_vector_list = []
        
        # Create an empty list for all tokens
        self.token_list = []
        self.token_ids = []
        
        # Create an empty list for each layer we are interested in
        layer_list = { layer_index: [] for layer_index in self.layer_indices }

        # Go over each corpus sentence for this type
        i = 0
        for sentence in tqdm(self.sentences):
            # Create hidden representations for the entire sentence. This creates representations for all
            # twelve layers in the network (plus the embedding layer).
            embedding_retriever = EmbeddingRetriever(bert_model, tokenizer, nlp, [ sentence ])
            
            # The token corresponding to our type might be inflected, or in some other form.
            # We use the spaCy tokenised forms to find the corresponding lemmas
            lemmas = list(map(lambda token: token.lemma_, embedding_retriever.tokens[0]))
            # We also do the same to find the actual tokens
            tokens = list(map(lambda token: token.text, embedding_retriever.tokens[0]))
    
            # The index of the token is the index of the type we are interested in
            # e.g. "I am going to the supermarket"
            #      "I be go to the supermarket"
            # lemma = go, index = 2 -> we find "going" in the token list
            token_index = lemmas.index(self.lemma)
    
            # Add this type instantiation / token to the list of tokens
            self.token_list.append(tokens[token_index])
            
            # Add the id for this token to the list of token ids
            self.token_ids.append("{}/{}/{}/{}".format(self.lemma, self.pos, self.source, i))
    
            # Go over each layer we want to know about, and save the hidden state from that layer
            # We only save the hidden state for the specific token we are interested in
            for layer_index in self.layer_indices:
                layer_list[layer_index].append(embedding_retriever.get_hidden_state(0, token_index, [ layer_index ]))
                
            i += 1
                
        # Create models based on layers
        for layer_index in layer_list:  
            # Create a model for each layer
            model = Model("layer_{}".format(layer_index), { "architecture": "BERT", "layer": layer_index })
            model.hidden_states = layer_list[layer_index]
            
            # Register the model
            self.model_collection.register_model(model)
                    
    def do_level_3_dimension_reduction(self):
        print("Applying dimension reduction (level 3)...")
        
        # We go over each model
        for model_name in tqdm(self.model_names):
            # We create a numpy array
            # rows = tokens, columns = dimensions of the hidden state of that layer
            layer_matrix = np.array(self.model_collection.models[model_name].hidden_states)
            
            for dimension_reduction_technique in self.dimension_reduction_techniques:
                self.model_collection.models[model_name].solutions[dimension_reduction_technique.name] = \
                    dimension_reduction_technique.reduce(layer_matrix)
            
    def create_similarity_matrices(self):
        print("Calculating similarity matrices...")
                
        # We go over each model
        for model_name in self.model_names:
            # Compute cosine similarity among tokens
            # https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
            dist_out = 1 - pairwise_distances(self.model_collection.models[model_name].hidden_states, metric="cosine")
            # Save the similarity matrix for this model
            self.model_collection.models[model_name].token_similarity_matrix = dist_out
            
    def create_distance_matrix(self):
        print("Calculating distances between models...")
                
        # We go over each model
        # Currently, the only available models are the different layers
        # This might need to be changed in the future if there are other parameters
        for model_name_i in self.model_names:
            # Create a dict for each model name (both rows and columns in the similarity matrix are the same)
            self.model_collection.models[model_name_i].model_similarity_vector = { model_name: None for model_name in self.model_names }
            for model_name_j in self.model_names:
                self.model_collection.models[model_name_i].model_similarity_vector[model_name_j] = \
                    self.get_models_euclidean_distance(self.model_collection.models[model_name_i].token_similarity_matrix,
                                                       self.model_collection.models[model_name_j].token_similarity_matrix)
                
    def get_models_euclidean_distance(self, model_a, model_b):
        return np.mean(np.linalg.norm(model_a - model_b, axis=1))
    
    def do_level_1_dimension_reduction(self):
        print("Applying dimension reduction (level 1)...")
        
        # model matrix
        # rows = models, columns = models
        model_matrix = []
        
        # We go over each model
        for model_name_i in self.model_names:
            # We retrieve the distance values for all models compared to this model
            row = [self.model_collection.models[model_name_i].model_similarity_vector[model_name_j] for model_name_j \
                   in self.model_collection.models[model_name_i].model_similarity_vector]
            model_matrix.append(row)
        
        model_matrix = np.array(model_matrix)
            
        # --- MDS ---
        # Apply multi-dimensional scaling
        mds = MDS(random_state=0, dissimilarity='precomputed') # TODO: Euclidean or manhattan distances?
        self.model_collection.mds_distance_matrix = mds.fit_transform(model_matrix)
            
        # Stress?
        stress = mds.stress_
        
        # --- TSNE ---
        # Apply TSNE
        tsne = TSNE(random_state=0) # TODO: what perplexity?
        self.model_collection.tsne_distance_matrix = tsne.fit_transform(model_matrix)