from sklearn.feature_extraction.text import TfidfVectorizer


class VectorizeDoc:
    """
    This class vectorizes the text.

    Methods:
        tf_converter: Converts the cleaned text to tfidf

    """

    def __init__(self):
        """ """
        pass

    @staticmethod
    def tf_converter(txt, cat, max_feat, max_freq, min_freq):
        """
        tf_converter: Converts the cleaned text to tfidf
        Parametrs:
            txt: text data (list)
            cat: category for tfidf i.e., 'word', 'ngram', or 'char' (str)
            max_feat: number of top tfidf features to be selected (int)
            max_freq: token max frequency (float)
            min_freq: token min frequency (float)
        Returns:
            final_doc: tfidf of the input text
            vocab: vocaulary
        """

        if cat == "word":
            # word level tf-idf
            vectorizer = TfidfVectorizer(
                analyzer="word",
                lowercase=True,
                stop_words="english",
                max_df=max_freq,
                min_df=min_freq,
                token_pattern=r"\w{1,}",
                max_features=max_feat,
                use_idf=True,
            )
        elif cat == "ngram":
            # ngram level tf-idf
            vectorizer = TfidfVectorizer(
                analyzer="word",
                lowercase=True,
                max_df=max_freq,
                min_df=min_freq,
                token_pattern=r"\w{1,}",
                ngram_range=(2, 3),
                max_features=max_feat,
                use_idf=True,
            )
        elif cat == "char":
            # characters level tf-idf
            vectorizer = TfidfVectorizer(
                analyzer="char",
                lowercase=True,
                token_pattern=r"\w{1,}",
                ngram_range=(3, 6),
                max_features=max_feat,
                use_idf=True,
            )

        final_doc = vectorizer.fit_transform(txt)
        vocab = vectorizer.get_feature_names()
        # final_doc_features = pd.DataFrame(final_doc.toarray(), columns=vw.get_feature_names())

        return final_doc, vocab
