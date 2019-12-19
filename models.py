# models.py

from optimizers import *
from nerdata import *
from utils import *

from collections import Counter
from typing import List

import numpy as np

import pdb
import string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.preprocessing import normalize

class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class FeatureBasedSequenceScorer(object):
    """
    Scoring function based on features


    """
    def __init__(self, weights: None, feature_cache: None, transition_log_probs: None):
        self.weights = weights
        self.feature_cache = feature_cache
        self.transition_log_probs = transition_log_probs

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, word_idx: int, tag_idx: int):
        features = self.feature_cache[word_idx][tag_idx]
        emission_potential = score_indexed_features(features, self.weights)

        return emission_potential
        # word = sentence_tokens[word_posn].word
        # word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        # return self.emission_log_probs[tag_idx, word_idx]


from sklearn.feature_extraction.text import TfidfVectorizer
def tfid_vector(sentences):
    tfid_sentences = []
    words = []
    for sentence_idx in range(0, len(sentences)):
        for word_idx in range(len(sentences[sentence_idx])):
            words.append(sentences[sentence_idx].tokens[word_idx].word)
        tfid_sentences.append(words)
        words = []
    tfid_sentences = [' '.join(i) for i in tfid_sentences]
    print(tfid_sentences[:2])    
    tfidf_vectorizer = TfidfVectorizer()
    values = tfidf_vectorizer.fit_transform(tfid_sentences)
    feature_names = tfidf_vectorizer.get_feature_names()

    return values, feature_names

class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """
    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def decode(self, sentence_tokens: List[Token]):
        """
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        ProbSequ = ProbabilisticSequenceScorer(self.tag_indexer, self.word_indexer, self.init_log_probs, self.transition_log_probs, self.emission_log_probs)
        v = np.zeros((len(sentence_tokens),9))
        bp = np.zeros((len(sentence_tokens),9))
        #print(len(sentence_tokens))
        #print(sentence_tokens)
        pred_tags = []
        for y in range(9):
            #v[0,y] = self.init_log_probs[y] + self.emission_log_probs[y,self.word_indexer.index_of(sentence_tokens[0].word)]
            v[0,y] = self.init_log_probs[y] + ProbSequ.score_emission(sentence_tokens, y,0)
            bp[0,y] = 0
        for i in range(1,len(sentence_tokens)):
            for y in range(9):
                # if y == 0: 
                #     v[i,y] = self.emission_log_probs[y,self.word_indexer.index_of(sentence_tokens[i].word)]
                #     bp[i,y] = 0
                # else:
                    max_values = np.zeros((9))
                    #v[i,y] = self.emission_log_probs[y,self.word_indexer.index_of(sentence_tokens[i].word)]
                    for index in range(9):
                        max_values[index] = self.transition_log_probs[index,y] + v[i-1,index] #+ ProbSequ.score_emission(sentence_tokens, y,i)
                    bp[i,y] = np.argmax(max_values)
                    v[i,y] = v[i,y] + np.max(max_values)
                    v[i,y] += ProbSequ.score_emission(sentence_tokens, y,i)

        col_idx = np.argmax(v[-1,:])
        pred_tags.append(self.tag_indexer.get_object(col_idx))

        # for i in range(1,len(sentence_tokens)):
        #   pred_tags.append(self.tag_indexer.get_object(np.argmax(v[i,:])))

        for i in range(len(sentence_tokens)-1,0,-1):
            pred_tag_idx = np.int(bp[i,col_idx])
            pred_tags.append(self.tag_indexer.get_object(pred_tag_idx))
            col_idx = np.int(bp[i,col_idx])
            
        pred_tags.reverse()
        #print(pred_tags)
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))
        #return LabeledSentence(sentence_tokens, pred_tags)

        raise Exception("IMPLEMENT ME")


def train_hmm_model(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:,word_indexer.add_and_get_index("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:,word_indexer.add_and_get_index("Phil")])
    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights, transition_log_probs, tfid_features, tfid_feature_names):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.transition_log_probs = transition_log_probs

        self.tfid_features = tfid_features
        self.tfid_feature_names = tfid_feature_names

    def decode(self, sentence_tokens):

        word_length = len(sentence_tokens)
        tags = 9
        feature_cache = [[[] for k in range(tags)] for j in range(word_length)]

        for curr_tag_index in range(tags):
            curr_tag = self.tag_indexer.get_object(curr_tag_index)
            for prev_tag_index in range(tags):
                prev_tag = self.tag_indexer.get_object(prev_tag_index)
                if prev_tag[0] == 'O' and curr_tag[0] == 'I':
                    self.transition_log_probs[prev_tag_index,curr_tag_index] = -np.inf
                elif curr_tag[0] == 'I':
                    if prev_tag[2:] != curr_tag[2:]:
                        self.transition_log_probs[prev_tag_index,curr_tag_index] = -np.inf


        for word_idx in range(0, word_length):
            #print("Extracting test features")
            for tag in range(tags):
                #print( word_idx, self.tag_indexer.get_object(tag))#, self.feature_indexer)
                #pdb.set_trace()
                #feature_cache[word_idx][tag] = extract_emission_features_additional(sentence_tokens, word_idx, self.tag_indexer.get_object(tag), self.feature_indexer, self.tfid_features, self.tfid_feature_names, None, add_to_indexer=False)
                feature_cache[word_idx][tag] = extract_emission_features(sentence_tokens, word_idx, self.tag_indexer.get_object(tag), self.feature_indexer, add_to_indexer=False)

        Fbs = FeatureBasedSequenceScorer(self.feature_weights, feature_cache, self.transition_log_probs)

        v = np.zeros((len(sentence_tokens),9))
        bp = np.zeros((len(sentence_tokens),9))

        pred_tags = []
        for y in range(9):
            v[0,y] =  Fbs.score_emission(0, y)
            bp[0,y] = 0

        for i in range(1,len(sentence_tokens)):
            for y in range(9):
                emission = Fbs.score_emission(i,y)
                for index in range(9):
                        current = v[i-1,index] + self.transition_log_probs[index,y] + emission
                        if index ==0:
                            v[i,y] = current
                            bp[i,y] = index
                        elif current > v[i,y]:
                            v[i,y] = current
                            bp[i,y] = index


        col_idx = np.argmax(v[-1,:])
        pred_tags.append(self.tag_indexer.get_object(col_idx))

        for i in range(len(sentence_tokens)-1,0,-1):
            pred_tag_idx = np.int(bp[i,col_idx])
            pred_tags.append(self.tag_indexer.get_object(pred_tag_idx))
            col_idx = np.int(bp[i,col_idx])

        pred_tags.reverse()
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))

        raise Exception("IMPLEMENT ME")


# Trains a CrfNerModel on the given corpus of sentences.

#def fwdbkw()



def train_crf_model(sentences, dev_sentences):
    tag_indexer = Indexer()


    #tfid_values, feature_names = tfid_vector(sentences)
    tfid_values = None
    feature_names = None

    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    for sentence_idx in range(0, len(sentences)):
        if sentence_idx % 100 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
            #break
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                #feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features_additional(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, tfid_values, feature_names, sentence_idx, add_to_indexer=True)
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
                #print("features[0][0]:",feature_cache[sentence_idx][word_idx][tag_idx])
    #pdb.set_trace()


    print("Training")

    ## debugging using crf_train_suit from scikit learn:
    # crf = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.1,c2=0.1,max_iterations=100,all_possible_transitions=True)
    # crf.fit(feature_cache, )

    hmm = train_hmm_model(sentences)
    transition_log_probs = hmm.transition_log_probs


    '''
    #feature_weights = np.random.normal(size=len(feature_indexer))
    feature_weights = np.zeros(len(feature_indexer))
    optimizer = UnregularizedAdagradTrainer(feature_weights)
    

    #print(np.shape(feature_weights))

    epoch = 10   
    for i in range(epoch):
        for sentence_idx in range(0, len(sentences)):
            if sentence_idx % 100 == 0:
                print("Ex %i/%i" % (sentence_idx, len(sentences)))
                #break
            alpha = np.zeros((len(sentences[sentence_idx]),9))

            for tag in range(9):
                alpha[0,tag] = optimizer.score(feature_cache[sentence_idx][0][tag])

            for word in range(1,len(sentences[sentence_idx])):
                for current_tag in range(9):
                    emission = optimizer.score(feature_cache[sentence_idx][word][current_tag])
                    for index in range(9):
                        if index==0:
                            alpha[word,current_tag] = emission + alpha[word-1,index]
                        else:
                            alpha[word,current_tag] = np.logaddexp(alpha[word,current_tag], emission + alpha[word-1,index])

            beta = np.zeros((len(sentences[sentence_idx]),9))

            ##check for beta[-1,tag]
            for tag in range(9):
                beta[-1,tag] = 0

            for word in range(len(sentences[sentence_idx])-2,0,-1):
                for curr_tag in range(9):
                    
                    for next_index in range(9):
                        emission = optimizer.score(feature_cache[sentence_idx][word+1][next_index])
                        if next_index == 0:
                            beta[word,curr_tag] = emission + beta[word + 1, next_index]
                        else:
                            beta[word,curr_tag] = np.logaddexp(beta[word,curr_tag], emission + beta[word + 1, next_index])

            marg_prob_log = np.zeros((len(sentences[sentence_idx]),9))
            marg_denom = np.zeros((len(sentences[sentence_idx])))

            for word in range(len(sentences[sentence_idx])):
                marg_denom[word] = alpha[word,0] + beta[word,0]
                for tag in range(1,9):
                    marg_denom[word] = np.logaddexp(marg_denom[word], alpha[word,tag] + beta[word,tag])


            for word in range(len(sentences[sentence_idx])):
                for tag in range(9):
                    marg_prob_log[word,tag] = alpha[word,tag] + beta[word,tag] - marg_denom[word]



            gradient_count = Counter()
            for word in range(len(sentences[sentence_idx])):
                golden_tag = sentences[sentence_idx].get_bio_tags()[word]
                golden_tag_index = tag_indexer.index_of(golden_tag)

                for feature in feature_cache[sentence_idx][word][golden_tag_index]:
                    gradient_count[feature]+=1.0

                for tag in range(9):
                    for feature in feature_cache[sentence_idx][word][tag]:
                        gradient_count[feature]+= -np.exp(marg_prob_log[word][tag])
            optimizer.apply_gradient_update(gradient_count,1)

    print(np.shape(optimizer.weights))
    np.save("optimizer_weights.npy", optimizer.weights)
    '''


    ## load a pre-trained weight
    weights = np.load("optimizer_weights_10epochs.npy")

    ## return function when using tfid features
    #return CrfNerModel(tag_indexer, feature_indexer, optimizer.weights, hmm.transition_log_probs, tfid_values, feature_names)
    return CrfNerModel(tag_indexer, feature_indexer, weights, hmm.transition_log_probs, tfid_values, feature_names)
    #return CrfNerModel(tag_indexer, feature_indexer, weights, transition_log_probs)

    raise Exception("IMPLEMENT THE REST OF ME")


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)


def extract_emission_features_additional(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, tfid_values: None, tfid_feature_names: None, sentence_idx: None, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 5
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape ## added punctuation
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        elif curr_word[i] in string.punctuation: 
            new_word +="!"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    #'''

    ## stop word indicator
    stop_word = []
    if curr_word in stopwords.words("english"):
        new_word = "Stop"
    else:
        new_word = "Go"

    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StopWord=" + repr(new_word))
    #'''


    ## POS tag clustering 
    #'''
    POS_clusters = {'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],\
                  'Adverb': ['RB', 'RBR', 'RBS', 'WRB'],\
                   'Adjective': ['JJ','JJR','JJS'],\
                    'NS': ['NN'],\
                     'NNS': ['NNS'],\
                      'NNP': ['NNP'],\
                       'NNPS': ['NNPS']}
    cluster_flag = 0
    for cluster in POS_clusters:
        # print ("cluster")
        # print (sentence_tokens[word_index].pos)
        # print (POS_clusters[cluster])
        if sentence_tokens[word_index].pos in POS_clusters[cluster]:
            maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":POS_clusters=" + repr(cluster))
            #print("cluster if")
            cluster_flag = 1
    if cluster_flag == 0:
            new_word = 'NotInCluster'
            maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":POS_clusters=" + repr(new_word))
            #print("cluster else")

    #'''
    ## TFID
    #'''

    #print("tfidf dimension:", np.shape(tfid_values))
    if add_to_indexer == True:
        max_tfidf_index = np.argmax(tfid_values[sentence_idx,:])
        #print(sentence_idx, word_index, max_tfidf_index)
        if curr_word == tfid_feature_names[max_tfidf_index]:
            #print(curr_word, max_tfidf_index)
            maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":TFID_max=" + repr(new_word))
        else:
            maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":TFID_not_max=" + repr(new_word))

    if add_to_indexer == False:
        new_word = curr_word
        if curr_word in tfid_feature_names:
            #pdb.set_trace()
            column_sum = np.sum(tfid_values, axis =0)
            column_sum_normalised = normalize(column_sum)
            #print (column_sum_normalised)
            feature_name_index = tfid_feature_names.index(curr_word)
            if column_sum_normalised[0,feature_name_index] > 0.5:
                maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":TFID_max=" + repr(new_word))
            else:
                maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":TFID_not_max=" + repr(new_word))
        else:
            maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":TFID_not_max=" + repr(new_word))
    #'''


    
    return np.asarray(feats, dtype=int)
