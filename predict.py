import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

french_stop_words = {"dans", "pour", "est", "sur", "avec", "plus", "les", "par", "que", "aux", "ses", "des", "cet", "ce", "cette", "le", "la", "un", "une", "et", "à"}

def preprocess_text(text):
    tokens = text.split()
    tokens = [word for word in tokens if word.lower() not in french_stop_words]
    return " ".join(tokens)

train_data['Text'] = train_data['Text'].apply(preprocess_text)
test_data['Text'] = test_data['Text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['Text'])
y_train = train_data['Label']
X_test = vectorizer.transform(test_data['Text'])

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_split, y_train_split)

y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)

print("Accuracy:", accuracy)

test_data['Label'] = model.predict(X_test)

test_data.to_csv('test.csv', index=False)
print(f"test.csv updated.")
