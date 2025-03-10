import spacy
import pandas as pd
import time
import re
from collections import Counter
from langdetect import DetectorFactory
from nltk.corpus import words


def review_lemmatize(review):
    doc = nlp(review)
    lemmatized_words = [token.lemma_ for token in doc]
    lemmatized_sentence = " ".join(lemmatized_words)
    return lemmatized_sentence


def clean_review(review):
    replace_dict = {
        "won't": "will not",
        "don't": "do not",
        "didn't": "did not",
        "can't": "can not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "couldn't": "could not",
        "'m": " am",
        "'re": " are",
        "'s": " is",
    }
    pattern = r'(' + '|'.join(map(re.escape, replace_dict.keys())) + r')'

    def replace_func(match):
        return replace_dict[match.group(0)]

    review = re.sub(pattern, replace_func, review)
    review = re.sub(r'[^\w\s]', ' ', review)
    review = re.sub('_', ' ', review)
    review = re.sub(r'\d+', '', review)
    review = ' '.join(review.split())
    return review.lower()


def meaningful_sentence(text):
    word_list = text.split()
    if not word_list:
        return False
    valid_words = sum(1 for word in word_list if word.lower() in english_words)
    if valid_words / len(word_list) >= 0.5:
        return True
    return False


def remove_url(data):
    url_pattern = r'http[s]?'
    data = data[~data[data.columns[2]].str.contains(url_pattern, na=False, regex=True)]
    data = data[data[data.columns[2]] != "early access review"]
    data = data[data[data.columns[2]] != " "]
    return data


def reduce_data(dataset):
    df = pd.DataFrame(columns=['app_id', 'app_name', 'review_text', 'review_score', 'review_votes'])
    count = Counter(dataset['app_name'])
    for key, value in count.items():
        if value > 1000:
            game = dataset[dataset['app_name'] == key]
            positive_like = game[(game['review_score'] == 1) & (game['review_votes'] == 1)]
            negative_like = game[(game['review_score'] == -1) & (game['review_votes'] == 1)]
            positive_unlike = game[(game['review_score'] == 1) & (game['review_votes'] == 0)]
            negative_unlike = game[(game['review_score'] == -1) & (game['review_votes'] == 0)]
            pl = int(800 * len(positive_like) / len(game)) if int(800 * len(positive_like) / len(game)) > 10 else len(positive_like)
            nl = int(800 * len(negative_like) / len(game)) if int(800 * len(negative_like) / len(game)) > 10 else len(negative_like)
            pu = int(800 * len(positive_unlike) / len(game)) if int(800 * len(positive_unlike) / len(game)) > 10 else len(positive_unlike)
            nu = int(800 * len(negative_unlike) / len(game)) if int(800 * len(negative_unlike) / len(game)) > 10 else len(negative_unlike)
            pl_data = positive_like.sample(pl)
            nl_data = negative_like.sample(nl)
            pu_data = positive_unlike.sample(pu)
            nu_data = negative_unlike.sample(nu)
            df = pd.concat([df, pl_data, nl_data, pu_data, nu_data], ignore_index=True)
        else:
            game = dataset[dataset['app_name'] == key]
            df = pd.concat([df, game], ignore_index=True)
        print("game", key, "has been added.")
        print(df.shape)
    return df


if __name__ == "__main__":
    # dataset column name:['app_id', 'app_name', 'review_text', 'review_score', 'review_votes']
    DetectorFactory.seed = 0
    english_words = set(words.words())
    nlp = spacy.load("en_core_web_sm")
    # train_data = pd.read_csv("results/top1000data.csv")
    # print(train_data.shape)
    # train_data = train_data.dropna(subset=[train_data.columns[2]])
    # print(train_data.shape)
    # train_data['review_text'] = [clean_review(text) for text in train_data['review_text']]
    # train_data = remove_url(train_data)
    # print(train_data.shape)
    # train_data = train_data.dropna(subset=[train_data.columns[2]])
    # print(train_data.shape)
    # train_data['review_text'] = [text for text in train_data['review_text'] if meaningful_sentence(text)]
    # print(train_data.shape)
    # train_data.to_csv("results/cleaned_top_1000.csv",index=False)
    # exit()
    # train_data = pd.read_csv('results/cleaned_top_1000.csv')
    # train_data = train_data.dropna(subset=['review_text'])
    # train_data = reduce_data(train_data)
    # train_data.to_csv("results/reduced_top1000.csv", index=False)
    # exit()
    # train_data = pd.read_csv("results/reduced_top1000.csv")
    # print(train_data.shape)
    # train_data = train_data.dropna(subset=['review_text'])
    # print(train_data.shape)
    # exit()
    # train_data = pd.read_csv("results/reduced_top1000.csv")
    # train_data['review_text'] = [review_lemmatize(text) for text in train_data['review_text']]
    # train_data.to_csv("results/final_top1000.csv", index=False)
    # exit()
    train_data = pd.read_csv("results/final_top1000.csv")
    count_top1000 = Counter(train_data[train_data.columns[1]])
    print(count_top1000)
