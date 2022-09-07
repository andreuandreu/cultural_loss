import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np



def dataframe_to_dic(df):
    # Create dictionaries out of the dataframe
    records = df.to_dict(orient='records')
    data = {x['concept']: x['number'] for x in records}
    #colors = {x['concept']: x['colour'] for x in records}
    return data

def print_wordcloud(data, bac_color= 'white', colormap = 'Pastel1', height = 200):

    # Generate word cloud from frequencies
    wc = WordCloud(background_color=bac_color,
                width= 400,
                height =height,
                prefer_horizontal=0.95, #ratio of times to try horizontal fitting as opposed to vertical
                #stopwords=stopwords,
                max_words=200,
                #max_font_size=40,
                min_font_size=8,
                scale=11,
                margin=1,
                colormap=colormap,
                font_step=1,
                #collocations=False,  # False, cloud doesn’t contain duplicate words
                mode="RGBA",
                #repeat=True,
                random_state=11,  # 5 red, 13 blue  ensure reproducibility of the exact same word cloud
                #regexp=None,
                relative_scaling=0.2,  # relative_scaling around .5 often looks good
                #collocation_threshold=30#Bigrams must have a Dunning likelihood greater than this to be counted as bigrams
                #color_func=lambda *args, **kwargs: (255, 0, 0)
                )
    #when the dictionary contains names and frequencies
    wc.generate_from_frequencies(data)

    #mask = np.random.rand(len(data))
    #image_colors = ImageColorGenerator(mask)

    def grey_color_func(word, font_size, position, orientation, random_state=None,
                        **kwargs):
        return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 100)

    def red_color_func(word, font_size, position, orientation, random_state=None,
                        **kwargs):
        return "hsl(0, 230%%, %d%%)" % np.random.randint(25, 80)

    def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return ("hsl(230,100%%, %d%%)" % np.random.randint(25, 80))
    
    def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return ("hsl(100,100%%, %d%%)" % np.random.randint(25, 80))

    # Color words depending on the colors dictionary
    def color_func(word, **kwargs):
        if colors.get(word) == 'g':
            return "rgb(0, 255, 0)"
        else:
            return "rgb(255, 0, 0)"

    #wc.recolor(color_func=color_func)
    wc.recolor(color_func=green_color_func)

    # Show final result
    plt.imshow(wc, interpolation="bilinear")  # 
    plt.axis("off")

    #plot_name = name[:-4]+".png"
    #WordCloud.to_file(plot_name, plot_name)
    #wordcloud.to_file("wordcloud.png")
    


    # Load data as pandas dataframe
name = './root_concepts.csv'
name = './change_concepts.csv'
name = './keep_concepts.csv'
bac_color =  'white'
#bac_color = 'red'
#bac_color = 'green'
colormap = 'Blues'
#colormap = 'Reds'
#colormap = 'Greens'

#colormap = 'YlOrRd'

df = pd.read_csv(name, sep=',')
print('ddffdfd', df)
data = dataframe_to_dic(df)
print_wordcloud(data, bac_color, colormap)  # height = 404

plt.show()
plt.savefig(name[:-4]+'.svg')
