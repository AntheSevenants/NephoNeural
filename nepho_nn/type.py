import numpy as np
import itertools

from anthevec.anthevec.embedding_retriever import EmbeddingRetriever
from .model_collection import ModelCollection
from .model import Model
from tqdm.auto import tqdm

from sklearn.metrics import pairwise_distances

class Type:
    def __init__(self,
                 lemma,
                 sentences,
                 bert_model,
                 tokenizer,
                 nlp,
                 dimension_reduction_techniques,
                 layer_indices,
                 attention_head_indices=[ None ]):
        print("Processing \"{}\"".format(lemma))

        # Type-related arguments
        self.lemma = lemma
        self.pos = "miep"
        self.source = "hallo"
        self.sentences = sentences

        if len(self.sentences) < 4:
            print("Warning: level 3 dimension reduction may fail because fewer than 4 sentences are given.")

        # NLP technology arguments
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.layer_indices = layer_indices

        # Dimension reduction 
        self.dimension_reduction_techniques = dimension_reduction_techniques

        # We want to get a model for each combination of arguments
        # e.g. if we are interested in layers 0, 1 and heads 1, 2,
        # we want to have the following models:
        # LAYER HEAD
        #     0    1
        #     0    2
        #     1    1
        #     1    2
        # To this end, we use itertools.product
        parameters = { "layer_index": layer_indices,
                        "attention_head_index": attention_head_indices }
        self.parameter_combinations = list(dict(zip(parameters, x)) for x in itertools.product(*parameters.values()))

        if len(self.parameter_combinations) < 4:
            print("Warning: level 1 dimension reduction may fail because fewer than 4 models will be generated.")
        
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
        embedding_retrievers = []
        self.token_list = []
        self.token_ids = []
        self.token_indices = []

        # Go over each corpus sentence for this type
        for i, sentence in tqdm(enumerate(self.sentences), total=len(self.sentences)):
            # Create hidden representations for the entire sentence. This creates representations for all
            # twelve layers in the network (plus the embedding layer).
            embedding_retriever = EmbeddingRetriever(self.bert_model,
                                                     self.tokenizer,
                                                     self.nlp,
                                                     [ sentence["sentence"] ])

            # Add this embedding retriever object
            embedding_retrievers.append(embedding_retriever)
    
            # The index of the token is pre-supplied, so we can just take it from the sentence object
            # Add the token index to the list of token indices
            token_index = sentence["token_index"]
            self.token_indices.append(token_index)
    
            # Add this type instantiation / token to the list of tokens
            self.token_list.append(embedding_retriever.tokens[0][token_index].text)
            
            # Add the id for this token to the list of token ids
            self.token_ids.append(sentence["token_id"])

            word_piece_index = embedding_retriever.correspondence[0][token_index]
                
        print("Going over parameter combinations...")
        # Go over each parameter combination that was precomputed and create a model for this combination
        for parameter_combination in tqdm(self.parameter_combinations):
            attention_heads = [ parameter_combination["attention_head_index"] ] if \
                                parameter_combination["attention_head_index"] is not None \
                                else None

            # Put together the model name
            layer_index_text = str(parameter_combination["layer_index"])
            attention_head_index_text = str(parameter_combination["attention_head_index"]) if \
                                            parameter_combination["attention_head_index"] is not None \
                                            else "no"

            model_name = f"{self.lemma}.layer{layer_index_text}.head{attention_head_index_text}"

            # Create a model based on the parameters
            model = Model(model_name,
                          { "architecture": "BERT",
                            "layer": f"layer{layer_index_text}",
                            "head": f"head{attention_head_index_text}"
                          })

            # Get the hidden states for all sentences
            hidden_states = []
            for i, sentence in enumerate(self.sentences):
                hidden_state = embedding_retrievers[i].get_hidden_state(0,
                                                                        self.token_indices[i],
                                                                        [ parameter_combination["layer_index"] ],
                                                                        attention_heads)

                hidden_states.append(hidden_state) 

            model.hidden_states = hidden_states

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

        # Will hold all reduced distance matrices
        self.solutions = {}

        # We do a dimension reduction on the distance matrix for each registered technique
        for dimension_reduction_technique in self.dimension_reduction_techniques:
            self.solutions[dimension_reduction_technique.name] = \
                    dimension_reduction_technique.reduce_model(model_matrix)