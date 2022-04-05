class Corpus:
	def __init__(self, lemma):
		self.lemma = lemma
		self.sentences = {}

	# Sentence: the sentence where the token occurs (list!)
	# Token index: the zero-based index of that token
	# Token id: a unique identifier to give to the token
	def add_sentence(self, sentence, token_index, token_id, file=None):
		sentence_entry = { "sentence": sentence,
						   "token_index": token_index,
						   "token_id": token_id,
						   "file": file }

		self.sentences.append(sentence_entry)

	def get_sentence(self, sentence_id):
		return self.sentences[sentence_id]