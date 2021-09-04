# NLP tfidf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from icecream import ic
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

TRAIN_CORPUS = 'train_after_analysis.csv'
STOP_WORDS = 'stopwords.txt'
WORDS_COLUMN = 'words_keep'

content = pd.read_csv(TRAIN_CORPUS)
corpus = content[WORDS_COLUMN].values


stop_words_size = 100
WORDS_LONG_TAIL_BEGIN = 10000
WORDS_SIZE = WORDS_LONG_TAIL_BEGIN - stop_words_size

stop_words = open(STOP_WORDS).read().split()[:stop_words_size]

tfidf = TfidfVectorizer(max_features=WORDS_SIZE, stop_words=stop_words)
text_vectors = tfidf.fit_transform(corpus)
print(text_vectors.shape)

targets = content['label']

x_train, x_test, y_train, y_test = train_test_split(text_vectors, targets, test_size=0.2, random_state=0)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
accuracy = accuracy_score(rf.predict(x_test), y_test)
ic(accuracy)


