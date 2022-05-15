import numpy as np
import itertools

from anthevec.anthevec.embedding_retriever import EmbeddingRetriever
from .model_collection import ModelCollection
from .model import Model
from .context_words import ContextWords
from .context_words_lemma import ContextWordsLemma
from tqdm.auto import tqdm

from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids

class Type:
    def __init__(self,
                 lemma,
                 sentences,
                 bert_model,
                 tokenizer,
                 nlp,
                 dimension_reduction_techniques,
                 medoid_clusters,
                 layer_indices,
                 attention_head_indices=[ None ],
                 mask_special_tokens=False,
                 collect_context_words=True):
        print("Processing \"{}\"".format(lemma))

        # Type-related arguments
        self.lemma = lemma
        self.pos = "miep"
        self.source = "hallo"
        self.sentences = sentences
        self.collect_context_words = collect_context_words
        self.mask_special_tokens = mask_special_tokens

        if 0 in layer_indices:
            raise ValueError("Embedding layer is not supported.")

        if len(self.sentences) < 4:
            print("Warning: level 3 dimension reduction may fail because fewer than 4 sentences are given.")

        # NLP technology arguments
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.layer_indices = layer_indices
        self.attention_threshold = 0.09

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

        if self.collect_context_words:
            self.do_level_3_dimension_reduction_context()
            
        self.create_similarity_matrices()
        self.create_distance_matrix()
        self.do_level_1_dimension_reduction()

        self.medoids = None
        if medoid_clusters:
            self.do_medoid_clustering(medoid_clusters)

    # Put together the model name
    def get_model_name(self, parameter_combination):
        layer_index_text = str(parameter_combination["layer_index"])
        attention_head_index_text = str(parameter_combination["attention_head_index"]) if \
                                        parameter_combination["attention_head_index"] is not None \
                                        else "no"

        model_name = f"{self.lemma}.layer{layer_index_text}.head{attention_head_index_text}"

        return model_name

        
    def get_token_vectors(self):
        print("Retrieving hidden states for all tokens...")
        print("Retrieving token embeddings for all context words...")
        
        token_vector_list = []
        
        # Create an empty list for all tokens
        self.token_list = []
        self.token_ids = []
        self.token_indices = []
        self.input_ids = []

        # This is how it's gonna work. We will keep all vector data in a large dict.
        # key = model name
        # Then, we go over each sentence, and piece together a vector for each parameter combination.
        # We could also just save the embedding retriever for each sentence and piece together the vectors later,
        # but then you'll quickly run out of RAM.
        #
        # Trust me. I've tried.
        models = {}
        models_meta = {}
        context_words = {}
        context_words_lemma = {}
        for parameter_combination in self.parameter_combinations:
            model_name = self.get_model_name(parameter_combination)

            # Create a list for this model
            models[model_name] = []

            models_meta[model_name] = { "architecture": "BERT",
                                        "layer": f"layer{parameter_combination['layer_index']}",
                                        "head": f"head{parameter_combination['attention_head_index']}"
                                      }

            context_words[model_name] = ContextWords()
            context_words_lemma[model_name] = ContextWordsLemma()


        # Go over each corpus sentence for this type
        for i, sentence in tqdm(enumerate(self.sentences), total=len(self.sentences)):
            # Create hidden representations for the entire sentence. This creates representations for all
            # twelve layers in the network (plus the embedding layer).
            embedding_retriever = EmbeddingRetriever(self.bert_model,
                                                     self.tokenizer,
                                                     self.nlp,
                                                     [ sentence["sentence"] ],
                                                     mask_special_tokens=self.mask_special_tokens)

            # The index of the token is pre-supplied, so we can just take it from the sentence object
            token_index = sentence["token_index"]
    
            # Add this type instantiation / token to the list of tokens
            self.token_list.append(embedding_retriever.tokens[0][token_index].text)
            
            # Add the id for this token to the list of token ids
            self.token_ids.append(sentence["token_id"])

            input_ids = embedding_retriever.input_ids.detach().numpy()[0]
            self.input_ids.append(input_ids)

            # Go over each parameter combination that was precomputed and get the hidden state
            for parameter_combination in self.parameter_combinations:
                attention_heads = [ parameter_combination["attention_head_index"] ] if \
                                    parameter_combination["attention_head_index"] is not None \
                                    else None

                hidden_state = embedding_retriever.get_hidden_state(0,
                                                                    token_index,
                                                                    [ parameter_combination["layer_index"] ],
                                                                    attention_heads)

                model_name = self.get_model_name(parameter_combination)
                models[model_name].append(hidden_state)

                if not self.collect_context_words:
                    continue

                # To get the context word pieces, we request the attention distribution for our word
                # for this parameter combination
                attention_distribution = embedding_retriever.get_attention_weights(0,
                                                                                   sentence["token_index"],
                                                                                   parameter_combination["layer_index"],
                                                                                   list(range(0, 12)))

                #print(max(attention_distribution))

                # Attach word pieces and indices to attention values 
                attention_distribution = list(zip(attention_distribution,
                                              embedding_retriever.word_pieces[0],
                                              list(range(0, len(attention_distribution)))))

                # Now filter only those items which are above the required attention threshold
                attention_distribution = list(filter(lambda attention_tuple: attention_tuple[0] >= self.attention_threshold,
                                                     attention_distribution))

                #print(len(attention_distribution))

                # For each relevant context word
                for attention_tuple in attention_distribution:
                    word_piece_index = attention_tuple[2]
                    actual_context_word = attention_tuple[1].replace("Ġ", "")
                    context_word = f"{model_name}/{i}/{actual_context_word}/{word_piece_index}"
                    context_word_lemma = f"{model_name}/{attention_tuple[1]}/{word_piece_index}"
                    word_piece_embedding = embedding_retriever.get_word_piece_vector(0,
                                                                                     word_piece_index,
                                                                                     parameter_combination["layer_index"])

                    # Add it to the context word collection of the current model
                    context_words[model_name].add(context_word, word_piece_embedding, i)

                    # Resolve the word index to the token index
                    # If SOS or EOS, there won't be any correspondence, so we override the lemma manually
                    if word_piece_index in [0, len(embedding_retriever.word_pieces[0]) - 1]:
                        context_word_lemma = "SOS" if word_piece_index == 0 else "EOS"
                    # Else, we resolve as usual
                    else:
                        context_word_index = embedding_retriever.get_token_index_from_word_piece_index(0, word_piece_index)
                        context_word_lemma = f"{embedding_retriever.tokens[0][context_word_index].lemma_}/" + \
                                             f"{embedding_retriever.tokens[0][context_word_index].pos_}"

                    # Also add the context word lemma form (vector not needed)
                    context_words_lemma[model_name].add(context_word_lemma, i)        

        # Register each model
        for model_name in models:
            model = Model(model_name, models_meta[model_name])
            model.hidden_states = models[model_name]

            if self.collect_context_words:
                model.context_words = context_words[model_name]
                model.context_words_lemma = context_words_lemma[model_name]

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

    def do_level_3_dimension_reduction_context(self):
        print("Applying dimension reduction (level 3, context words)...")

        # We go over each model
        for model_name in tqdm(self.model_names):
            # We create a numpy array
            # rows = tokens, columns = dimensions of the context word vector of that model
            model_matrix = np.array(self.model_collection.models[model_name].context_words.token_embeddings)

            for dimension_reduction_technique in self.dimension_reduction_techniques:
                self.model_collection.models[model_name].context_solutions[dimension_reduction_technique.name] = \
                    dimension_reduction_technique.reduce(model_matrix)
            
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

    def do_medoid_clustering(self, medoid_clusters):
        print(f"Applying K-medoid clustering with {medoid_clusters} clusters...")

        if medoid_clusters > 9:
            print(f"Warning: NephoVis can only inspect 9 models at a time. \
                    You will be unable to inspect all {medoid_clusters} medoids at once.")

        distance_matrix = []
        for model_name in self.model_names:
            row = list(map(lambda model_name_inner: self.model_collection.models[model_name].model_similarity_vector[model_name_inner],
                           self.model_collection.models[model_name].model_similarity_vector))

            distance_matrix.append(row)

        medoids = KMedoids(medoid_clusters, metric="precomputed", method="pam").fit(distance_matrix)
        
        self.medoids = list(map(lambda medoid_index: self.model_names[medoid_index],
                                medoids.medoid_indices_))

    def attach_variables(self, variables):
        self.variables = variables

        # NephoVis will crash if you have tokens for which no variables are available
        # Here, we'll make a list of token ids for which variables are available
        # There is no guideline for missing data yet, so this is more of a stopgap
        self.token_ids_variables_available = list(map(lambda row: row["_id"], self.variables))
        
    def export_models(self, output_path):
        for model_name in self.model_names:
            # We create a numpy array
            # rows = tokens, columns = dimensions of the hidden state of that layer
            layer_matrix = np.array(self.model_collection.models[model_name].hidden_states)
            np.save(f"{output_path}/{model_name}", self.soc_matrix)