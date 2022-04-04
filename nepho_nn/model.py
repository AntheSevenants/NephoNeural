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