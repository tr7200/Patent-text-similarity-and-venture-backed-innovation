#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:44:36 2019

@author: user

This command line script takes arguments for:
    - the number of patents to scrape (1-30)
    - the SDC Platinum industry category to scrape these patents from
    - the filename containing your patent's text (1 line CSV file)

Returns:
    - The average of the cosine similarities between the scraped patents 
    and your patent (choose number = 1 if you don't want an average)

`python get_cosine_similarity-3-21-20.py -n 5 -sdc biotech -filename test_cosine.csv`

should return:

'The similarity is 0.04'

"""


import re
import sys
import json
import nltk # You may have to run 'nltk.download('stopwords')' first
import random
import pandas as pd
import numpy as np
import urllib
import requests
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from bs4 import BeautifulSoup
from html.parser import HTMLParser
import argparse



Biotech_patents = [5149635, 5196320, 5338669, 5741682, 6329195, 6830913,
                   6267964, 6822075, 6380365, 5955316, 6080559, 6399570,
                   6589727, 6589727, 6589727, 6589727, 6541268, 4918011,
                   4973478, 4997814, 5171672, 5218093, 5086039, 5169837,
                   5169837, 5208041, 5208041, 5290920, 5344819, 5382658]

CommMedia_patents = [6396896, 6625763, 5715174, 5764892, 5802280, 6244758,
                     6300863, 6507914, 5991402, 6389409, 5491563, 6178243,
                     6373947, 6725249, 6532230, 6192344, 6292549, 6754202,
                     6766006, 6493439, 6738465, 6643321, 6731710, 3985392,
                     4043603, 6452565, 6768454, 6816118, 5645434, 5734842]

CompRel_patents = [6176582, 5845111, 5867715, 6754077, 6772394, 6192258, 
                   6192258, 6381637, 6622306, 6470381, 6675204, 5309348,
                   5349387, 5402178, 5411620, 5422589, 5434372, 5440252,
                   5442278, 5465927, 5483154, 5518216, 5633656, 5526196,
                   5526418, 5561653, 5574891, 5639336, 5506533, 5535285]

Medical_patents = [5912132, 5955281, 6358698, 6528529, 6627645, 6815458, 
                   6756393, 6316453, 6596719, 6720322, 6703392, 6815451,
                   6702146, 4784162, 4784162, 4784162, 4784162, 4827943,
                   4827943, 4827943, 4827943, 5922610, 6063027, 6038913, 
                   6149606, 6099480, 6183416, 6511425, 6626844, 6723056]

NonHT_patents = [5976068, 6185978, 6261392, 6352297, 6471292, 6640595,
                 6684505, 6726258, 6751998, 6331028, 6627018, 5862303,
                 6069997, 6128439, 6132532, 6467676, 6578754, 6365435,
                 6510976, 6550666, 6592019, 6599775, 6732913, 6734039,
                 6750082, 5980870, 6220172, 6622579, 4037086, 4069412]

semiconductor_patents = [5911316, 6813166, 6150999, 6160531, 6195073, 6198222, 
                         6113449, 6150767, 6172465, 6323830, 6344714, 6388643,
                         6433762, 6501453, 6507150, 6376813, 6479933, 6483490,
                         6621216, 6633126, 6489573, 6631565, 6166867, 6501726, 
                         6549507, 6785202, 6366551, 6483798, 6492889, 6654185]

sdc_dict ={'biotech': Biotech_patents,
           'comm': CommMedia_patents,
           'comprel': CompRel_patents,
           'medical': Medical_patents,
           'nonht': NonHT_patents,
           'semi': semiconductor_patents}



parser = argparse.ArgumentParser()
parser.add_argument("-n", 'number', type=int,
                    help="The number of patents for comparison (also rnd.seed)")
parser.add_argument("-sdc", 'sdc', type=str,
                    help="The SDC category to which your patent belongs. Can \
                    be either biotech, comm, comprel, medical, nonht, or semi")
parser.add_argument("-f", "filename", type=str,
                    help="The name of CSV file containing your text.")



args = parser.parse_args()
number = args.number
sdc = args.sdc
filename = args.filename

np.random.seed(number // 2)

patent_of_interest = pd.read_csv(pd.Series(filename), sep=',')
patent_list = random.choices(sdc_dict[sdc], k=number)
Patents = pd.DataFrame(data = patent_of_interest, columns=['Desc'])



class MLStripper(HTMLParser):
    """
    Helper class for the 
    strip_tags() function
    """
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    """
    HTML tag stripper

    Parameters
    ----------
    html : scraped HTML data

    Returns
    -------
    Tag-stripped HTML

    """
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def patent_scraper(patent_list=patent_list, 
                   Patents=Patents):
    """
    Scrapes patents according to the SDC platinum category chosen by the 
    user iand the number of patents to scrape in the argparse statements.
    There is a good chance that this code will not work by the time you use
    it. Github is a graveyard of patent scraping libraries and scripts that 
    have ceased to work due to API restrictions and HTML formatting changes.

    Parameters
    ----------
    patent_list : list of randomly chosen USPTO patent numbers
        from the category chosen by the user in the argparse.
    Patent_data : Empty dataframe for the scraped patent descriptionss
        from the patent_list. Position 0 has the user's patent description
        to which these scraped patents will be compared to obtain a cosine
        similarity measure.

    Returns
    -------
    Patent_data : Full dataframe with scraped (not cleaned) patent 
        descriptions below the patent of interest.

    """
    scraped_patents = pd.DataFrame(columns=['Desc'])

    for i in range(len(patent_list)): # my fav anti-pattern
        try:
            number = str(int(patent_list[i]))
            url = 'https://patents.google.com/patent/US' + number + '/en?oq=' + number
            page_response = requests.get(url, timeout=20)
            soup = BeautifulSoup(page_response.content, "html.parser")
            desc = soup.find('section', {"itemprop": "description"})
            tag_free = strip_tags(str(desc))
            tag_free = re.sub('\n','', tag_free)
            desc_list = []
            desc_list.append(tag_free)
            scraped_patents.loc[i+1,'Desc'] = str(desc_list)
        except:
          pass
    
    Patents = pd.concat([Patents,scraped_patents])
    
    return Patents


def lemmatize_text(text=text):
    """
    Lemmatizer    

    Parameters
    ----------
    text : list of words

    Returns
    -------
    lemmatized list

    """
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    return [w for w in w_tokenizer.tokenize(text)]


def prepare_text(Patents=Patents):
    """
    
    
    Parameters
    ----------
    Patent_data : Pandas Dataframe
        Single column 'Descriptions' has patent of interest at position 0 and 
        the text of the scraped patents below it.
    
    Returns
    -------
    Snowball-stemmed, tokenized, lemmatized lower-case text with stop words 
        removed and words less than two characters removed.
    
    """
    stop_words = stopwords.words('english')
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    snowballer = SnowballStemmer('english')
    
    patent_specific_stop_words = ['Fig.', 'fig.' 'step', '(', ')', 'no.', 'i.e.', 'e.g.', 'pp.', 'ip', '(step )' '(step ).']
  
    Patents['Desc'] = Patents['Desc'].str.lower()
    Patents['Desc'] = Patents['Desc'].apply(lambda x: re.sub(r'\d+', '', str(x)))
    Patents['Desc'] = Patents['Desc'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in stop_words]))
    Patents['Desc'] = Patents['Desc'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in patent_specific_stop_words]))
    # Remove words less than 2 chars
    Patents['Desc'] = Patents['Desc'].apply(lambda x: re.sub(r'\b\w{1,2}\b', '', str(x)))
    Patents['Desc'] = Patents['Desc'].apply(lambda x: tokenizer.tokenize(str(x)))
    Patents['Desc'] = Patents['Desc'].apply(lambda x: lemmatize_text(str(x)))
    Patents['Desc'] = Patents['Desc'].apply(lambda x: " ".join([snowballer.stem(word) for word in x]))
  
    return Patents


def get_cosine(Patents=Patents):
    """
    Get the average cosine similarity for the patent of interest.

    Parameters
    ----------
    Patent_data : DataFrame with a single column 'Desc' with patent text that 
        has been cleaned and prepared by the prepare_text() function.

    Returns
    -------
    Cosine_similarity : a single cosine similarity value between the patent of 
        interest andd number averaged over the number of patents scraped.

    """
    length = len(Patents) - 1
    tfidf_vectorizer = TfidfVectorizer()
  
    summary_tfidf_matrix = tfidf_vectorizer.fit_transform(Patents['Desc'])
    cosine_matrix = cosine_similarity(summary_tfidf_matrix)
    average_of_cosines = np.mean(cosine_matrix[0][1:])
  
    cosine = str('The similarity is %.2f' % average_of_cosines)
  
    return cosine



if '__name__' == main:
    get_cosine(prepare_text(patent_scraper(patent_list, Patents)))

