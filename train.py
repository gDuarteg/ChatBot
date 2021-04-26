import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle
from util import clean
from sklearn.metrics import accuracy_score


df = pd.read_csv('sentencas.csv').drop(columns=['Timestamp'])
df["Sentença"] = df["Sentença"].apply(clean)

train, test = train_test_split(df, test_size=0.15)
vectorizer = CountVectorizer()
textos = train['Sentença'].tolist()
counts = vectorizer.fit_transform(textos)
y = train['Intenção']
model = MultinomialNB()
model.fit(counts, y)

score = accuracy_score(train["Intenção"], model.predict(vectorizer.transform(train["Sentença"])))

print("Score:",score)

with open("model.bin","wb+") as f:
    pickle.dump(model, f)

with open("vectorizer.bin","wb+") as f:
    pickle.dump(vectorizer, f)