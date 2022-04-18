class Model:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata
        
        # --- Original hidden state matrix ---
        self.hidden_states = None
        
        # --- Dimension-reduced hidden states ---
        # Will hold the *reduced* hidden states (in low-dimensional space)
        self.solutions = {}
        
        # --- Similarity matrices/vectors ---
        self.token_similarity_matrix = None
        self.model_similarity_vector = None

        # --- Context words information ---
        # Will hold the context words for this model, as well as their vectors
        self.context_words = None
        self.context_words_lemma = None
        # Will hold the *reduced* word vectors (in low-dimensional space)
        self.context_solutions = {}