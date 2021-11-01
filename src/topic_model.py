from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


class TopicModels:
    """
    This class applies with topic modeling  algorithm
        methods:
          topic_modelling: Applies topic modelling and does visualization

    """

    @staticmethod
    def topic_modelling(
        doc_ready, vocab, tm_method, num_topic, num_top_words, topic_vis=True
    ):
        # calculate_perplexity=False,
        # calculate_coherence=True):
        """
         Parametrs:
            doc_ready: Cleaned and vectorized text
            vocab: Vaculary ,
            tm_method: the topic method to be applied,
            num_topic: Number of topics ,
            num_top_words: Number of top words to be displayed for each topic,
            topic_vis: Variable if true will show the visualization
        Returns:
            Topic along with topic words
        """

        if tm_method == "lda":
            lda = LatentDirichletAllocation(n_components=num_topic, random_state=1)
            id_topic = lda.fit_transform(doc_ready)
            topic_words = {}

            for topic, comp in enumerate(lda.components_):
                word_ids = np.argsort(comp)[::-1][:num_top_words]
                topic_words[topic] = [vocab[i] for i in word_ids]


        return topic_words
