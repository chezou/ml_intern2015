# -*- coding: utf-8 -*-
"""
given a word and visualize near words
original source code is https://github.com/nishio/mycorpus/blob/master/vis.py
"""
from gensim.models import word2vec
from sklearn.decomposition import PCA
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.font_manager

class visWord2Vec:
	def __init__(self, filename='vectors.bin'):
		font = matplotlib.font_manager.FontProperties(fname='/Library/Fonts/Osaka.ttf')
		FONT_SIZE = 20
		self.TEXT_KW = dict(fontsize=FONT_SIZE, fontweight='bold', fontproperties=font)

		self.model = word2vec.Word2Vec.load_word2vec_format(filename, binary=True)
		print 'loaded'

	def search(self, positive, negative=[]):
		results = self.model.most_similar(positive=positive, negative=negative)
		"rank\tword\tsimilarity"
		for index, (word, score) in enumerate(results):
			print "%i\t%s\t%s" %(index+1, word, score)

	def plot(self, positive, negative=[], nbest = 15):
		words = positive + [w for w, _ in self.model.most_similar(positive=positive, negative=negative, topn=nbest)]

		# do PCA
		X = [self.model[w] for w in words]
		pca = PCA(n_components=2)
		pca.fit(X)
		print pca.explained_variance_ratio_
		X = pca.transform(X)
		xs = X[:, 0]
		ys = X[:, 1]

		# draw
		plt.figure(figsize=(12,8))
		plt.scatter(xs, ys, marker = 'o')
		for i, w in enumerate(words):
			plt.annotate(
					w,
					xy = (xs[i], ys[i]), xytext = (3, 3),
					textcoords = 'offset points', ha = 'left', va = 'top',
					**self.TEXT_KW)

		plt.show()
