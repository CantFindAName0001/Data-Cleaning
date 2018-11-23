#author Sahil Malik
#Time 12:34 AM
#Location New Delhi

import nltk #for removing conjunctions pronouns and prepositions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
#reading input file
file1=open('text.txt','r')
text=file1.read()
file1.close()

# split into words and lower case every word
tokens = word_tokenize(text)
tokens = [w.lower() for w in tokens]

#remove punctuations from each word
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]

#remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]

#filter stop words
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
#print(words[:100])

# stemming of words
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in words]
print(stemmed[:100])



"""
This criterion advantages sentences that contain keywords. To determine the web
page keywords, we suggest to use the tf.idf technique (Term Frequency Inverse Document
Frequency times) used in information retrieval to assign weights to the terms
(words) of a document.
According to the tf.idf technique a word is important if it is relatively common in the
web page and relatively rare in a large collection consisting of web pages linked by
hypertext links to that web page.
We propose to ignore the "empty" words (e.g. conjunctions, pronouns, prepositions,
etc.) that figure in a fixed list and calculate the tf.idf for the remaining terms using the
following formula:
n
w tf N i j ij = ×log
where wij is the weight of the term Tj in the page Pi ; tfij is the frequency of the term Tj
in the page Pi ; N is the number of pages linked by hypertext links to the page Pi ; n is
the number of pages where the term Tj occurs at least once.
We calculate for each word of the web page its tf.idf and we retain as keywords only
those which tf.idf is above the average tf.idf.
The retained keywords are then enriched with the keywords that are in the keywords
Meta tag (if it is available in the HTML file of the Web page).
The score of a sentence s, according to this criterion is the number of keywords contained
in the processed sentence: C3(s) = number of keywords of the sentence s.
"""
