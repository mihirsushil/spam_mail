import pandas as pd 


# importing the datset 

df = pd.read_csv('spam_assassin.csv')
df['label'] = df['target']
print(df.shape)
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




# bigger data set(~ 75,000)

df2 = pd.read_csv('processed_data.csv')
cols = ['label', 'message']
df2 = df2[cols]
print(df2.shape)
df2 = df2.dropna(axis =1, how = 'all')
# pre processing 
def preprocess_text2(text):
    text = str(text)  
    text = re.sub(r'<[^>]*>','',text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]',  '', text)
    text = ' '.join(text.split())
    return text 
df2['preprocessed_text'] = df2['message'].apply(preprocess_text2)

X_train2, X_temp2, y_train2, y_temp2 = train_test_split(df2['preprocessed_text'], df2['label'], test_size=0.2, random_state=42, stratify = df2['label']) 

X_val2, X_test2, y_val2, y_test2 = train_test_split(X_temp2 , y_temp2, test_size = 0.5, random_state = 42, stratify = y_temp2)

model.fit(X_train2,y_train2)
y_val_pred2 = model.predict(X_val2) 
print(classification_report(y_val2,y_val_pred2, target_names=["Spam","Ham"], digits = 3 ))

y_test_pred2 = model.predict(X_test2)
print(classification_report(y_test2, y_test_pred2, target_names=["Spam","Ham"], digits = 3 ))
