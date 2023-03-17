import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import pandas as pd

stemport = PorterStemmer()

# we'll do all above steps using a single function:
def sms_transform(sms):
    sms=sms.lower()    #for lower-----
    sms=nltk.word_tokenize(sms)  #for tokenize the words
    alnum=[]
    for i in sms:
        if i.isalnum():
            alnum.append(i)     # remove special characters
            
            
    sms=alnum[:]
    alnum.clear()
    for i in sms:
        if i not in stopwords.words('english') and i not in string.punctuation:
            alnum.append(i)
                    
    sms=alnum[:]
    alnum.clear()     # removing stop words and punctuation
                     
    for i in sms:
        alnum.append(stemport.stem(i))
            
    return " ".join(alnum) #stemming



tfidf = pickle.load(open('Tdif_Vec.pkl','rb'))
model = pickle.load(open('MOdel.pkl','rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Write a SMS: ")

if st.button('Predict'):

    transformed_sms = sms_transform(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
