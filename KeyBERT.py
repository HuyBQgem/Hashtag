import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def embedding(doc, n_gram_range, stop_words):
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
    candidates = count.get_feature_names_out()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)
    return doc_embedding, candidate_embeddings, candidates


def cos_sim_only(doc_embedding, candidate_embeddings, candidates, top):
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    return [candidates[index] for index in distances.argsort()[0][-top:]]


def max_sum_similarity(doc_embedding, candidate_embeddings, candidates, top, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings,
                                             candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


def maximal_marginal_relevance(doc_embedding, candidate_embeddings, candidates, top, diversity):
    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)
    word_similarity = cosine_similarity(candidate_embeddings)

    # Initialize candidates and already choose the best keyword/key phrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(candidates)) if i != keywords_idx[0]]

    for _ in range(top - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [candidates[idx] for idx in keywords_idx]
