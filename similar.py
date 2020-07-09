# -*- coding: utf-8 -*-
import logging
import jieba
from gensim import corpora, models, similarities

logging.basicConfig(level=logging.DEBUG)
jieba.setLogLevel(logging.INFO)


class DocumentSimilar(object):
    def __init__(self, documents):
        self.documents = documents
        self.dictionary = None
        self.tfidf = None
        self.similar_matrix = None
        self.get_stop_word()
        self.calculate_similar_matrix()

    @staticmethod
    def split_word(doc):
        """
        分词，去除停用词
        """
        text = []
        stop_words = {":", " ", "#", "*", "=", "的", "，", "”", "？", "?"}
        doc = doc.replace('\n', '')
        for word in jieba.cut(doc):
            if word not in stop_words:
                text.append(word)
        return text

    def calculate_similar_matrix(self):
        """
        计算相似度矩阵及一些必要数据
        """
        words = [self.split_word(document) for document in self.documents]
        self.dictionary = corpora.Dictionary(words)
        corpus = [self.dictionary.doc2bow(word) for word in words]
        self.tfidf = models.TfidfModel(corpus)
        corpus_tfidf = self.tfidf[corpus]
        self.similar_matrix = similarities.MatrixSimilarity(corpus_tfidf)
        print('success')

    def get_similar(self, document):
        """
        计算要比较的文档与语料库中每篇文档的相似度
        """
        words = self.split_word(document)
        corpus = self.dictionary.doc2bow(words)
        corpus_tfidf = self.tfidf[corpus]
        return self.similar_matrix[corpus_tfidf]


if __name__ == '__main__':
    documents = open('haha.txt', 'r').readlines(-1)
    doc_similar = DocumentSimilar(documents)
    new_doc = input()
    while new_doc != "":
        print(new_doc)
        # 要比较的文档
        tmp = 0
        val = ''
        for value, document in zip(doc_similar.get_similar(new_doc), documents):
            if value > 0.5:
                tmp = value
                val = document
                break
            if value > tmp:
                tmp = value
                val = document
            else:
                continue
        print(tmp)
        print(val)
        new_doc = input()
