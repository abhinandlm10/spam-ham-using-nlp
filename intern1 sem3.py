# Importing Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords

# Loading the Dataset
data = pd.read_csv("spam_or_not_spam.csv")
print("Examples of text before preprocessing:")
print(data.columns)

# Preprocessing
data["email"] = data["email"].apply(lambda x: str(x).lower())
data["email"] = data["email"].apply(lambda x: " ".join([word for word in x.split() if word not in stopwords.words("english")]))

# Print column names (headers)
print("Column names (headers):")
print(data.columns)

# Vectorization
vectorizer = CountVectorizer(min_df=1, stop_words='english')
X = vectorizer.fit_transform(data["email"])
y = data["label"]

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Evaluating the Model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Predict labels for the entire dataset
y_all_pred = model.predict(X)

# Calculate the total count of spam and ham emails
total_spam_count = sum(y_all_pred == 1)
total_ham_count = sum(y_all_pred == 0)

print(f"Total spam emails: {total_spam_count}")
print(f"Total ham emails: {total_ham_count}")