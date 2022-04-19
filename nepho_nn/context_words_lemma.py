from .context_words import ContextWords

class ContextWordsLemma(ContextWords):
	def __init__(self):
		super()

	def add(self, word, sentence_id):
		# We do this first, because it'll allow us to have unique collections
		if sentence_id not in self.words_in_sentence:
			self.words_in_sentence[sentence_id] = set()

		super().add(self, word, None, sentence_id)