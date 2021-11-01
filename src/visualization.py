import matplotlib.pyplot as plt
from wordcloud import WordCloud

class TopicVisulaization():
    
    @staticmethod
    def word_cloud(word_list):
        wordcloud = WordCloud(background_color='white').generate(' '.join(word_list))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
        return wordcloud



