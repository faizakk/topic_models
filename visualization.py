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



'''
doc0 = ['focus','sleeping','depressed','asleep','attention','mind','cymbalta','appetite', 'psychiatrist','energy']
#doc1 =' '.join(doc0)
wcloud = TopicVisulaization.word_cloud(doc0)

path = '/Users/faizakhankhattak/Documents/hack/veri_code/topic_models/'
wcloud.to_file(path+'word_cloud_vis.png')
'''