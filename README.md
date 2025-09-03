Spam Mail Classifier 
- Detects if an email is spam or ham

Data Set: 
- SpamAssassin dataset (https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset)

Process: 
- Preprocess text: lowercase, remove HTML tags, URLs, email addresses, punctuation, and numbers
- Split data into train (80%), validation (10%), and test (10%)
- Vectorize text using TF-IDF with unigrams and bigrams 
- Trained using a Logistic Regression model 
- Evaluate on validation and test sets using precision, recall, F1-score

Results:
- Validation Accuracy: 99.1%
- Test Accuracy: 99.5%

Future Improvements:
- Experiment with other models (Naive Bayes, SVM, Neural Networks)
- Try different feature extraction methods (character n-grams, embeddings)
- Test on modern spam datasets (with obfuscation, emojis, etc.)
- Deploy as a web app (Flask/Streamlit)
