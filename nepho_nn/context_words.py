class ContextWords:
	def __init__(self):
		self.words = []
		self.token_embeddings = []

	def add(self, word, token_embedding):
		if word not in self.words:
			self.words.append(word)
			self.token_embeddings.append(token_embedding)