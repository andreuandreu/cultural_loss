
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np


def dataframe_to_dic(df):
    # Create dictionaries out of the dataframe
    records = df.to_dict(orient='records')
    data = {x['word']: x['number'] for x in records}
    #colors = {x['concept']: x['colour'] for x in records}
    return data


def print_wordcloud(ax, text, height=200, bac_color='white'):

    min_font_size = 8 #threshold, below this words will not be displayed, so is linked to scale and relative_scaling
    # Generate word cloud from frequencies
    wc = WordCloud(background_color=bac_color,
                width= 400,
                height =height,
                prefer_horizontal=0.99, #ratio of times to try horizontal fitting as opposed to vertical
                #stopwords=stopwords,
                max_words=200,
                #max_font_size=40,
                min_font_size=min_font_size,#  is linked to and relative_scaling and font_step
                relative_scaling=0.2,  # relative_scaling around .5 often looks good
                font_step=1,  # step on how the scale grows
                scale=11,  # Scaling between computation and drawing. For large word-cloud images,
                   #using scale instead of larger canvas size is significantly faster,
                margin=1,
                #colormap=colormap,
                #collocations=False,  # False, cloud doesn’t contain duplicate words
                #mode="RGBA",
                #repeat=True,
                random_state=1,  # 5 red, 13 blue  ensure reproducibility of the exact same word cloud
                #regexp=None,
                #collocation_threshold=30#Bigrams must have a Dunning likelihood greater than this to be counted as bigrams
                #color_func=lambda *args, **kwargs: (255, 0, 0)
                   )
                
    wc.generate_from_frequencies(data)
    #wc.recolor(color_func=color_func)
    #wc = WordCloud()
    # Show final result
    ax.imshow(wc, interpolation="bilinear")  #
    ax.axis("off")
    
name = 'top_10_fields'
df = pd.read_csv(name, sep=',')
data = dataframe_to_dic(df)


fig, ax = plt.subplots(1,1, constrained_layout=False, figsize=(5.8, 3.1))

text = 'Sociology Environmental Science Psychology Psychological Assessment Sociology Climatic Change Sociology Political Science Sociology Education History History Political Science'
print_wordcloud(ax,  data, bac_color='white')
plt.show()
