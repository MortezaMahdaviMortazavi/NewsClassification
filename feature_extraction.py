import math
import torch
import torch.nn as nn
from collections import Counter


class TFIDFVectorizer:
    def __init__(self):
        """Initialize the TFIDFVectorizer with empty attributes."""
        self.documents = []            # List to store input documents
        self.doc_count = 0             # Total number of documents
        self.term_freq = []            # List to store term frequencies for each document
        self.inverse_doc_freq = {}     # Dictionary to store inverse document frequencies for each term
        self.vocab = set()             # Set to store the unique vocabulary of all documents

    def _build_vocab(self):
        """Build the vocabulary from the documents."""
        for doc in self.documents:
            tokens = doc.split()
            self.vocab.update(tokens)

    def _calculate_term_frequency(self, document):
        """Calculate the term frequency (TF) for a given document."""
        tokens = document.split()
        term_freq = Counter(tokens)
        return term_freq

    def _calculate_inverse_doc_frequency(self):
        """Calculate the inverse document frequency (IDF) for each term in the vocabulary."""
        for term in self.vocab:
            doc_count_with_term = sum(1 for doc in self.documents if term in doc.split())
            self.inverse_doc_freq[term] = math.log(self.doc_count / (1 + doc_count_with_term))

    def fit_transform(self, new_documents):
        """
        Update the model with new documents and return the TF-IDF matrix for all documents.

        Parameters:
        - new_documents: List of new documents to update the model.

        Returns:
        - tfidf_matrix: TF-IDF matrix for all documents.
        """
        self.documents.extend(new_documents)
        self.doc_count += len(new_documents)

        self._build_vocab()

        for doc in new_documents:
            term_freq = self._calculate_term_frequency(doc)
            self.term_freq.append(term_freq)

        self._calculate_inverse_doc_frequency()

        tfidf_matrix = []
        for doc_term_freq in self.term_freq:
            tfidf_vector = [0.0] * len(self.vocab)
            for i, term in enumerate(self.vocab):
                tf = doc_term_freq[term]
                idf = self.inverse_doc_freq[term] if term in self.inverse_doc_freq else 0.0
                tfidf_vector[i] = tf * idf
            tfidf_matrix.append(tfidf_vector)

        return tfidf_matrix

    def transform(self, new_document):
        """
        Transform a new document into a TF-IDF vector using the existing model.

        Parameters:
        - new_document: The new document to transform.

        Returns:
        - tfidf_vector: TF-IDF vector for the new document.
        """
        term_freq = self._calculate_term_frequency(new_document)
        tfidf_vector = [0.0] * len(self.vocab)

        for i, term in enumerate(self.vocab):
            tf = term_freq[term]
            idf = self.inverse_doc_freq[term] if term in self.inverse_doc_freq else 0.0
            tfidf_vector[i] = tf * idf

        return tfidf_vector

    def update(self, new_documents):
        """
        Update the model with new documents and return the TF-IDF matrix for all documents.

        Parameters:
        - new_documents: List of new documents to update the model.

        Returns:
        - tfidf_matrix: TF-IDF matrix for all documents.
        """
        tfidf_matrix = self.fit_transform(new_documents)
        return tfidf_matrix



class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size,proj_size,dropout_rate=0.2, use_batch_norm=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(embed_size)

        self.projector = nn.Linear(in_features=embed_size, out_features=proj_size)

    def forward(self, tokens_tensor):
        embed_matrix = self.embedding(tokens_tensor)
        embed_matrix = self.dropout(embed_matrix)
        
        if self.use_batch_norm:
            embed_matrix = self.batch_norm(embed_matrix)

        proj = self.projector(embed_matrix)
        return proj



# if __name__ == "__main__":
#     tfidf_vectorizer = TFIDFVectorizer()
#     initial_documents = [
#         "This is the first document.",
#         "This document is the second document.",
#         "And this is the third one.",
#         "Is this the first document?",
#     ]
#     initial_tfidf_matrix = tfidf_vectorizer.fit_transform(initial_documents)
#     new_document = "A new document for testing."
#     tfidf_vector = tfidf_vectorizer.transform(new_document)
#     print(f"TF-IDF Vector for the New Document: {tfidf_vector}")
#     new_documents = ["Another new document for evaluation.", "Yet another new document."]
#     updated_tfidf_matrix = tfidf_vectorizer.update(new_documents)
#     for i, vector in enumerate(updated_tfidf_matrix):
#         print(f"Document {i + 1}: {vector}")
