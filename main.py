import sys
import string
from datetime import datetime
from nltk.corpus import stopwords
import pandas as pd
from docs_preprocessor import DocsPreprocessor
from vectorize_doc import VectorizeDoc
from topic_model import TopicModels
from visualization import TopicVisulaization


sys.path.append(".")


def main():
    """
    Loads data
    Cleans data
    Creates topic using topic modelling
    Stores the results in a csv
    Creates word clouds for each topic and saves them
    """
    STOP_LIST = stopwords.words("english")
    PUNCT = set(string.punctuation)
    min_token_len = 3
    nums_topics = 2
    topic_word_display = 3
    max_features = 4
    tfidf_category = "word"
    tfidf_max_word_freq = 0.8
    tfidf_min_word_freq = 0.1

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = "/Users/faizakhankhattak/Documents/hack/veri_code/topic_models/"

    documents = [
        "This is a test of nerves for those who have done amazing things.",
        "Never hide and never be intimidated",
    ]
    # documents1 = docs_preprocessor.tokenize_doc(documents)
    # print(documents1)

    # print(docs_preprocessor.clean_doc((documents1), 2, punct_list = PUNCT, stop_list=STOP_LIST))
    documents1 = DocsPreprocessor.cleaning_pipeline(
        documents,
        min_token_len,
        punct_list=PUNCT,
        stop_list=STOP_LIST,
        lemmatized=True,
        stemmed=False,
    )

    doct, features = VectorizeDoc.tf_converter(
        documents1,
        tfidf_category,
        max_features,
        tfidf_max_word_freq,
        tfidf_min_word_freq,
    )
    print(doct)
    print("features", features)

    topic_words = TopicModels.topic_modelling(
        doct, features, "lda", nums_topics, topic_word_display
    )
    print(type(topic_words), topic_words)

    topic_df = pd.DataFrame(
        {
            "Topic_number": list(topic_words.keys()),
            "Topic_words": list(topic_words.values()),
        }
    )
    print(topic_df)

    topic_df.to_csv(
        path + "predicted_" + str(nums_topics) + "_topics_" + str(now) + ".csv"
    )

    for i, j in topic_words.items():
        print("Saving word cloud for topic", i, ".......")
        TopicVisulaization.word_cloud(j).to_file(
            path + "word_cloud_vis_topic_" + str(i) + "_" + str(now) + ".png"
        )


if __name__ == "__main__":
    main()
