# Imports Libraries
import random
import string
import warnings
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# TF-IDF: Term Frequency – Inverse Document Frequency

# Downloads Auxiliar Vocabulary
# nltk.download() => Opens the UI to View and Download the Vocabulary
nltk.download('popular', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Chat Bot Variables
BOT_NAME = "B-Bot"

# Keyword Matching for Greeting
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad you are chatting with me"]

# Reads ChatBot Content from File
with open('banking.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenize Content By Sentences
sent_tokens = nltk.sent_tokenize(raw)

# Lematize Tokens
def LemTokens(tokens):
    lemmer = WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

# Lematize and Normalize Text
def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None)
                             for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Return a Greeting if User Sent a Greeting
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Process User Input, Get Response and Retuns it
def response(user_response):
    bot_response = ''

    # Add User Response to Sent Tokens List to Process Similarity
    sent_tokens.append(user_response)

    # Creates and Train a Tf-Idf Vectorizer Model
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)

    # Gets The Most Similar Values Using Cosine Similarity Method ([-1]: Last Item)
    vals = cosine_similarity(tfidf[-1], tfidf)

    # Gets The Answer Index to Pick From the Array ([-2]: Last 2 Items)
    idx = vals.argsort()[0][-2]

    # Get a Copy of Vals Array Dimentionally Reduced
    flat = vals.flatten()
    flat.sort()

    # Sets the Bot Answer to the Response String
    if(flat[-2] == 0):
        bot_response = bot_response + "I am sorry, but i don't Understand your Question"
    else:
        bot_response = bot_response+sent_tokens[idx]
    
    # Removes the User Response from the Sent Tokens List
    sent_tokens.remove(user_response)

    # Returns the Response from Bot
    return bot_response

# Write Text to User
def talk_to_client(message):
    print(f"{BOT_NAME}: " + message)

# Executes the Chat Bot
if __name__ == '__main__':
    flag = True
    talk_to_client(f"My name is {BOT_NAME}. I will answer your queries about Banking.")
    while(flag == True):
        talk_to_client(
            "Please type a Question about Banking. If you want to exit, type Bye!")
        user_response = input()
        if("bye" in user_response.lower()):
            flag = False
            talk_to_client("Bye! take care..")
        elif ("thank" in user_response.lower()):
            flag = False
            talk_to_client("You are welcome..")
        elif (greeting(user_response) != None):
            talk_to_client(greeting(user_response))
        else:
            talk_to_client(response(user_response))
