from collections import Counter
def clean_by_freq(tokenized_words, cut_off_count):
    vocab = Counter(tokenized_words)
    
    uncommon_words = {key for key, value in vocab.items() if value <= cut_off_count}
    
    #uncommon_words에 포함되지 않는 단어 리스트 생성
    cleaned_words = [word for word in tokenized_words if word not in uncommon_words]
    
    return cleaned_words

def clean_by_len(tokenized_words, cut_off_length):
    cleaned_by_freq_len = []
    
    for word in tokenized_words:
        if len(word) > cut_off_length:
            cleaned_by_freq_len.append(word)
            
    return cleaned_by_freq_len

def clean_by_stopwords(tokenized_words, stop_words_set):
    cleaned_words =[]
    
    for word in tokenized_words:
        if word not in stop_words_set:
            cleaned_words.append(word)
    return cleaned_words


from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def pos_tagger(tokenized_sents):
    pos_tagged_words =[]

    for sentence in tokenized_sents:
        tokenized_words = word_tokenize(sentence)

        pos_tagged = pos_tag(tokenized_words)
        pos_tagged_words.extend(pos_tagged)

    return pos_tagged_words

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    
def words_lemmatizer(pos_tagged_words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []

    for word, tag in pos_tagged_words:
        wn_tag = penn_to_wn(tag)

        if wn_tag in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
            lemmatized_words.append(lemmatizer.lemmatize(word, wn_tag))
        else:
            lemmatized_words.append(word)

    return lemmatized_words

from nltk.tokenize import sent_tokenize
nltk.download('punkt')
def sent_tokenize(text):
    tokenized_sents = sent_tokenize(text)
    return tokenized_sents