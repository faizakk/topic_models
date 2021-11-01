"""
    File name: main.py
    Author: Faiza Khan Khattak
"""

# import sys
# sys.path.append(".")
import string
from datetime import datetime
from nltk.corpus import stopwords
import pandas as pd
from docs_preprocessor import DocsPreprocessor
from vectorize_doc import VectorizeDoc
from topic_model import TopicModels
from visualization import TopicVisulaization


def main():
    """
    Loads data
    Cleans data
    Creates topic using topic modelling
    Stores topic numbers and topic words in a csv
    Creates word clouds for each topic and saves them

        stop_word_list: list of words to be removed from the text (list)
        punctuation_list: list of punctuation to be removed from the text (list)
        min_token_len: min acceptable length of the tokens (int)
        nums_topics: Number of topics (int)
        topic_word_display: Number of words to be displayed for each topic (int)
        max_features: number of top tfidf features to be selected (int)
        tfidf_category: category for tfidf i.e., 'word', 'ngram', or 'char' (str)
        tfidf_max_word_freq: token max frequency (float)
        tfidf_min_word_freq: token min frequency (float)
    """
    stop_word_list = stopwords.words("english")
    punctuation_list = set(string.punctuation)
    min_token_len = 3
    nums_topics = 2
    topic_word_display = 3
    max_features = 4
    tfidf_category = "word"
    tfidf_max_word_freq = 0.8
    tfidf_min_word_freq = 0.1

    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Load  data
    path = "/Users/faizakhankhattak/Documents/hack/veri_code/topic_models/"
    file_name = "dummy_data.csv"
    text_df = pd.read_csv(path + file_name)
    documents = text_df.text.tolist()
    processed_documents = DocsPreprocessor.cleaning_pipeline(
        documents,
        min_token_len,
        punct_list=punctuation_list,
        stop_list=stop_word_list,
        lemmatized=True,
        stemmed=False,
    )

    # Vectorize data
    doc_tf, features = VectorizeDoc.tf_converter(
        processed_documents,
        tfidf_category,
        max_features,
        tfidf_max_word_freq,
        tfidf_min_word_freq,
    )

    # Topic modelling
    topic_words = TopicModels.topic_modelling(
        doc_tf, features, "lda", nums_topics, topic_word_display
    )

    topic_df = pd.DataFrame(
        {
            "Topic_number": list(topic_words.keys()),
            "Topic_words": list(topic_words.values()),
        }
    )

    # Save topic modelling results
    topic_df.to_csv(
        path + "predicted_" + str(nums_topics) + "_topics_" + str(now) + ".csv"
    )

    # Word clouds
    for i, j in topic_words.items():
        print("Saving word cloud for topic", i, ".......")
        TopicVisulaization.word_cloud(j).to_file(
            path + "word_cloud_vis_topic_" + str(i) + "_" + str(now) + ".png"
        )


if __name__ == "__main__":
    main()
