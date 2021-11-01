from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer


class DocsPreprocessor:
    """
    Tokenize text, clean text,  lemmatize  and/or stem the text.

        Methods:

            tokenize_doc: Tokenizes text

            clean_doc: Cleans text by removing stopwords, punctuation and short tokens.

            lemmatize_doc: Lemmatization  of text.

            stem_doc: Stemming text

            cleaning_pipeline: Combines all above functions into a pipeline

    """

    def __init__(self):
        """ Empty """
        pass

    @staticmethod
    def tokenize_doc(text_data):
        """
        Tokenizes text
            Parametrs: Text data(list)
            Returns: Tokenized text (list)

        """
        return [word_tokenize(doc) for doc in text_data]

    @staticmethod
    def clean_doc(tokenized_text, min_token_len, punct_list, stop_list):
        """
        Cleans text by removing stopwords, punctuation and short tokens

            Parametrs: Tokenized text (list),  minimum acceptable token length (int),
                    list of punction (list), list of stop-words (list)
            Returns: Cleaned text   (list)

        """
        return [
            [
                token.lower()
                for token in doc
                if (not token.isdigit())
                and (len(token) > min_token_len)
                and (token not in punct_list)
                and (token not in stop_list)
            ]
            for doc in tokenized_text
        ]

    @staticmethod
    def lemmatize_doc(cleaned_text):
        """
        Lemmatizes the cleaned tokenized text.
            Parametrs: Cleaned text (list)
            Returns: Lemmatized text (list)

        """
        lemmatizer = WordNetLemmatizer()
        return [[lemmatizer.lemmatize(token) for token in doc] for doc in cleaned_text]

    @staticmethod
    def stem_doc(cleaned_text):
        """
        Stems the cleaned tokenized text.
            Parametrs: Cleaned/Lemmatized text (list)
            Returns: Stemmed text (list)

        """
        ps = PorterStemmer()
        return [[ps.stem(token) for token in doc] for doc in cleaned_text]

    @staticmethod
    def cleaning_pipeline(
        text_data, min_token_len, punct_list, stop_list, lemmatized=True, stemmed=False
    ):
        """
        Combines all above functions into a pipeline
            Parametrs: Text data (list)
            Returns: Cleaned and lemmatized/stemmed text (list)

        """

        updated_text = DocsPreprocessor.tokenize_doc(text_data)
        updated_text = DocsPreprocessor.clean_doc(
            updated_text, min_token_len, punct_list, stop_list
        )

        if lemmatized:
            updated_text = DocsPreprocessor.lemmatize_doc(updated_text)

        if stemmed:
            updated_text = DocsPreprocessor.stem_doc(updated_text)

        return [" ".join(txt) for txt in updated_text]
