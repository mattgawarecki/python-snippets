from collections import Counter
import nltk
from nltk import tokenize
from nltk.util import ngrams
import numpy as np

class MarkovChainGenerator(object):
    def __init__(self, start_token):
        self.corpus = []
        self.start_token = start_token

    def load(self, text):
        # print('Tokenizing sentences')
        sents = tokenize.sent_tokenize(text)
        # print('{} tokens found'.format(len(sents)))

        # print('Tokenizing words')
        tokens = [
            [self.start_token] + tokenize.word_tokenize(s)
            for s in sents
        ]
        # print('{} tokens found'.format(sum(len(t) for t in tokens)))

        for t in tokens:
            self.corpus.extend(t)
        # print('Tokens added to corpus')
        # print('----------')

    def generate(self, *, ngram_size=3):
        # print('Starting output generation')
        # print('Corpus length: {} tokens'.format(len(self.corpus)))

        # print('Grouping ngrams')
        all_ngrams = Counter(ngrams(self.corpus, ngram_size))
        # print('{} groups of ngrams found'.format(len(all_ngrams)))

        # print('Building initial buffer')
        last_n = [self.start_token]
        for i in range(1, ngram_size):
            next_token = self._pick_next(last_n, all_ngrams)
            last_n.append(next_token)
            # print('Added token: {}'.format(next_token))

        # print('Buffer built. Beginning output loop')
        while True:
            output = last_n.pop(0)
            # if output != self.start_token:
            yield output

            next_token = self._pick_next(last_n, all_ngrams)
            last_n.append(next_token)

    def _pick_next(self, prev, all_ngrams):
        if prev[-1] == self.start_token:
            candidate_space = {
                k[1]: v
                for k, v in all_ngrams.items()
                if k[0] == self.start_token
            }
        else:
            next_index = len(prev)
            candidate_space = {
                k[next_index]: v
                for k, v in all_ngrams.items()
                if list(k)[:next_index] == list(prev)
            }


START_SENTENCE_TOKEN = '^^^'
markov = MarkovChainGenerator(START_SENTENCE_TOKEN)
# markov.load(text)

# for word in markov.generate():
#     if (word == START_SENTENCE_TOKEN):
#         print()
#     else:
#         print(word, end=' ', flush=True)
