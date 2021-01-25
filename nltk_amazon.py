import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pickle
import json
import os.path
from os import path
import html
import argparse

def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict

def load_reviews(review_file_name):
    f = open(review_file_name, "r")
    for review in tqdm(f):
        reviews.append(json.loads(review))

    print("Loaded " + str(len(reviews)) + " reviews")

    negative_reviews = []
    positive_reviews = []
    mixed_reviews = []
    #Use truncated list for testing
    # for review in tqdm(reviews[1:5000]):    
    for review in tqdm(reviews):
        words = [review.strip() for review in html.unescape(review['reviewText']).split()]
        if review['overall'] <= 2.0:
            negative_reviews.append((create_word_features(words), "negative"))
        if review['overall'] >= 4.0:
            positive_reviews.append((create_word_features(words), "positive"))
        if review['overall'] == 3.0:
            mixed_reviews.append((create_word_features(words), "mixed"))    

    print("Loaded " + str(len(positive_reviews)) + " positive reviews")
    print("Loaded " + str(len(mixed_reviews)) + " mixed reviews")
    print("Loaded " + str(len(negative_reviews)) + " negative reviews") 

if __name__ == '__main__':
    desc = 'nltk_amazon is a script to parse amazon reviews and provide helpful information. Here are some helpful parameters:'
    parser = argparse.ArgumentParser(desc)
    parser.add_argument("-V", "--verbose", help="Display most informative features and a word cloud for positive, negative, and mixed reviews", action="store_true")
    parser.add_argument("-m", "--mode", type=str, help="Mode of script: interactive, input")
    args = parser.parse_args()


    if not path.exists('amazon.pickle'):
        reviews = load_reviews("Sports_and_Outdoors_5.json")

    try:
        #Attempt to load classifier
        f = open('amazon.pickle', 'rb')
        classifier = pickle.load(f)
    except Exception as e:
        print("No classifier model to load.")
        train_set = positive_reviews + negative_reviews + mixed_reviews
        classifier = NaiveBayesClassifier.train(train_set)

        #Save classifier
        f = open('amazon.pickle', 'wb')
        pickle.dump(classifier, f)
        f.close()

    #Example negative review
    print("Here's an example review:")
    scathing_review = "Piece of crap. I will never buy this brand again. All in all I would say don't waste your money."
    print(scathing_review)
    words = word_tokenize(scathing_review)
    words = create_word_features(words)
    print("Label: " + classifier.classify(words))

    if args.mode == "interactive":
        user_review=""
        while user_review != "exit":
            print("Entering interactive mode, please enter \"exit\" to terminate this process")
            user_review = input("Please enter your review: ")
            user_words = word_tokenize(user_review)
            user_features = create_word_features(user_words)
            print("Your review would most likely be: " + classifier.classify(user_features))            

    if args.verbose:
        print(classifier.most_informative_features(500))