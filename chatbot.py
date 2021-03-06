#Imports Libraries
import io
import random
import string
import warnings
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

#TF-IDF: Term Frequency – Inverse Document Frequency

#Downloads Auxiliar Vocabulary
nltk.download('popular', quiet = True)
nltk.download('punkt', quiet = True)
nltk.download('wordnet', quiet = True)

#Reads ChatBot Content from File
with open('banking.txt', 'r', encoding = 'utf8', errors = 'ignore') as fin:
    raw = fin.read().lower()

#Tokenize Content By Sentences and By Words
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

#Lematize Data
def LemTokens(tokens):
    lemmer = WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

#Normalize Data
def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Keyword Matching for Greeting
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

#Return a Greeting if User Sent a Greeting
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

#Process User Input, Get Response and Retuns it
def response(user_response):
    robo_response = ''

    #Add User Response to Sent Tokens List to Process Similarity
    sent_tokens.append(user_response)

    #Creates and Train a Tf-Idf Vectorizer Model
    TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words = 'english')
    tfidf = TfidfVec.fit_transform(sent_tokens)

    #Gets The Most Similar Values Using Cosine Similarity Method ([-1]: Last Item)
    vals = cosine_similarity(tfidf[-1], tfidf)
    
    #Gets The Answer Index to Pick From the Array ([-2]: Last 2 Items)
    idx = vals.argsort()[0][-2]

    #Get a Copy of Vals Array Dimentionally Reduced
    flat = vals.flatten()
    flat.sort()

    #Sets the Bot Answer to the Response String 
    if(flat[-2] == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

flag = True
print("VBot: My name is VBot. I will answer your queries about Banking and Financial Services. If you want to exit, type Bye!")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you' ):
            flag = False
            print("VBot: You are welcome..")
        else:
            if(greeting(user_response) != None):
                print("VBot: " + greeting(user_response))
            else:
                print("VBot: ", end = "")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("VBot: Bye! take care..")