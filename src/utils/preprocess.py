from __future__ import absolute_import, division

import os
import re
import random
import gc
import bisect
from unicodedata import category, name, normalize
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Imputer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import text, sequence


def impute_missing(data, strategy):

    if data.isnull():
        if strategy == 'mean':
            miss_mean_imputer = Imputer(
                missing_values='NaN', strategy='mean', axis=0)
            miss_mean_imputer = miss_mean_imputer.fit(data)
            imputed_df = miss_mean_imputer.transform(data.values)
            return imputed_df
        elif strategy == 'median':
            miss_mean_imputer = Imputer(
                missing_values='NaN', strategy='mean', axis=0)
            miss_mean_imputer = miss_mean_imputer.fit(data)
            imputed_df = miss_mean_imputer.transform(data.values)
            return imputed_df


def apply_scale(data, strategy):

    if strategy == 'standard':
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

    if strategy == 'minmax':
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)


def remove_space(text):
    spaces = ["   "]
    for space in spaces:
        text = text.replace(space, ' ')
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    return text


def label_encoding(df, col):
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    le.fit(df[col])
    le_classes = le.classes_.tolist()
    bisect.insort_left(le_classes, '<missing>')
    le.classes_ = le_classes
    df[col] = le.transform(df[col])
    del le
    return df[col]


def clean_str(text, max_text_length):
    try:
        text = ' '.join([w for w in text.split()[:max_text_length]])
        text = text.lower()
        text = re.sub(u"é", u"e", text)
        text = re.sub(u"ē", u"e", text)
        text = re.sub(u"è", u"e", text)
        text = re.sub(u"ê", u"e", text)
        text = re.sub(u"à", u"a", text)
        text = re.sub(u"â", u"a", text)
        text = re.sub(u"ô", u"o", text)
        text = re.sub(u"ō", u"o", text)
        text = re.sub(u"ü", u"u", text)
        text = re.sub(u"ï", u"i", text)
        text = re.sub(u"ç", u"c", text)
        text = re.sub(u"\u2019", u"'", text)
        text = re.sub(u"\xed", u"i", text)
        text = re.sub(u"w\/", u" with ", text)

        text = re.sub(u"[^a-z0-9]", " ", text)
        text = u" ".join(re.split('(\d+)', text))
        text = re.sub(u"\s+", u" ", text).strip()
        text = ''.join(text)
    except:
        text = np.NaN
    return text


def remove_diacritics(s):
    return ''.join(c for c in normalize('NFKD', s.replace('ø', 'o').replace('Ø', 'O').replace('⁻', '-').replace('₋', '-'))
                   if category(c) != 'Mn')


def clean_special_punctuations(text, special_punc_mappings):
    for punc in special_punc_mappings:
        if punc in text:
            text = text.replace(punc, special_punc_mappings[punc])
    return text


def clean_number(text):
    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
    text = re.sub(r'(\d+)(e)(\d+)', '\g<1> \g<3>', text)

    return text


def pre_clean_rare_words(text, rare_words_mapping):
    for rare_word in rare_words_mapping:
        if rare_word in text:
            text = text.replace(rare_word, rare_words_mapping[rare_word])

    return text


def clean_misspell(text, mispell_dict):
    for bad_word in mispell_dict:
        if bad_word in text:
            text = text.replace(bad_word, mispell_dict[bad_word])
    return text


def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


def spacing_punctuation(text):
    regular_punct = list(string.punctuation)
    all_punct = list(set(regular_punct))
    all_punct.remove('-')
    all_punct.remove('.')
    for punc in all_punct:
        if punc in text:
            text = text.replace(punc, f' {punc} ')
    return text


def clean_bad_case_words(text, bad_case_words):
    for bad_word in bad_case_words:
        if bad_word in text:
            text = text.replace(bad_word, bad_case_words[bad_word])
    return text


def clean_special_text(x):

    x = str(x).replace(' s ', '').replace(
        '…', ' ').replace('—', '-').replace('•°•°•', '')
    for punct in "/-'":
        if punct in x:
            x = x.replace(punct, ' ')
    for punct in '&':
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    for punct in '?!-,"#$%\'()*+-/:;<=>@[\\]^_`{|}~–—✰«»§✈➤›☭✔½☺éïà-*' + '“”’':

        if punct in x:
            x = x.replace(punct, '')
    for punct in '.•':
        if punct in x:
            x = x.replace(punct, f' ')

    x = re.sub(r'[\x00-\x1f\x7f-\x9f\xad]', '', x)
    x = re.sub(r'(\d+)(e)(\d+)', r'\g<1> \g<3>', x)
    x = re.sub(r"(-+|\.+)\s?", "  ", x)
    x = re.sub("\s\s+", " ", x)
    x = re.sub(r'ᴵ+', '', x)

    return x


def remove_special(text):
    text = re.sub(r'[^\w]', ' ', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.replace('-', '').replace(' ', '')
    text = text.replace('%', '').replace(' ', '')
    text = text.replace('@', '').replace(' ', '')
    text = text.replace('$', '').replace(' ', '')
    text = text.replace('/', '').replace(' ', '')
    text = text.replace("'", "")
    text = text.lower()
    return text


def preprocess_dataframe(data, strategy, col):
    impute_missing(data, strategy)
    apply_scale(data, strategy)
    label_encoding(data, col)
    return text


def preprocess_text(text):

    text = clean_special_text(text)
    text = clean_number(text)
    text = remove_punct(text)
    text = spacing_punctuation(text)
    text = remove_special(text)
    return text
