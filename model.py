import pandas as pd 


# importing the datset 

df = pd.read_csv('spam_assassin.csv')
df['label'] = df['target']
import re 
# pre-processing text 
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]*>','',text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]',  '', text)
    text = ' '.join(text.split())
    return text 
df['preprocessed_text'] = df['text'].apply(preprocess_text)

from sklearn.model_selection import train_test_split 
# 3 splits: train, validation, test 
X_train, X_temp, y_train, y_temp = train_test_split(df['preprocessed_text'], df['label'], test_size=0.2, random_state=42, stratify = df['label']) 

X_val, X_test, y_val, y_test = train_test_split(X_temp , y_temp, test_size = 0.5, random_state = 42, stratify = y_temp)

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression  
from sklearn.pipeline import make_pipeline 
from sklearn.metrics import classification_report

# Model: Logistical Regression with a conneecting pipeline that helps in weighing the words 
model = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df =5, max_df = 0.80 ), 
        LogisticRegression(class_weight = 'balanced'))
# fitting the training model and testing validation set
model.fit(X_train,y_train)
y_val_pred = model.predict(X_val) 
print(classification_report(y_val,y_val_pred, target_names=["Spam","Ham"], digits = 3 ))
# test case 
y_test_pred = model.predict(X_test)
print(classification_report(y_test, y_test_pred, target_names=["Spam","Ham"], digits = 3 ))

