class ContextWords:
	def __init__(self):
		self.words = []
		self.words_in_sentence = {}

		self.token_embeddings = []

	def add(self, word, token_embedding, sentence_id):
		if sentence_id not in self.words_in_sentence:
			self.words_in_sentence[sentence_id] = []

		self.words_in_sentence[sentence_id].append(word)

		if word not in self.words:
			self.words.append(word)
			self.token_embeddings.append(token_embedding)