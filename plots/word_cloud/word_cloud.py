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


def print_wordcloud(ax, df, data, color_dic,  key, height=200, bac_color='white'):

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
                random_state=11,  # 5 red, 13 blue  ensure reproducibility of the exact same word cloud
                #regexp=None,
                #collocation_threshold=30#Bigrams must have a Dunning likelihood greater than this to be counted as bigrams
                #color_func=lambda *args, **kwargs: (255, 0, 0)
                )
    
    #when the dictionary contains names and frequencies
    wc.generate_from_frequencies(data)

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return color_dic[key] % np.random.randint(25, 80)#(int(font_size)*99/min_font_size)

    wc.recolor(color_func=color_func)

    # Show final result
    ax.imshow(wc, interpolation="bilinear")  # 
    ax.axis("off")

    #plot_name = name[:-4]+".png"
    #WordCloud.to_file(plot_name, plot_name)
    #wordcloud.to_file("wordcloud.png")


mossaic_keys = [['left', 'upper right'],
                ['left', 'lower right']]

#fig, axes = plt.subplot_mosaic(mossaic_keys, sharex=True, sharey=True, constrained_layout=True,
#                               figsize=(5.5, 3.5), gridspec_kw={'hspace': 0, 'wspace': 0})

fig = plt.figure(constrained_layout=False, figsize=(5.8, 3.1))  # facecolor='0.9'

gs = fig.add_gridspec(nrows=2, ncols=2, left=0.0, right=0.99, top=0.99, bottom=0.0,
                     hspace=0.0, wspace=0.0)

ax0 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, -1])
ax2 = fig.add_subplot(gs[-1, -1])

axes = [ax0, ax1, ax2]
    
   


color_dic = {'blue': "hsl(230,100%%, %d%%)",
              'red': "hsl(20,230%%, %d%%)",
            'green': "hsl(100,100%%, %d%%)"}

colors_names = ['blue', 'red', 'green']
names = ['./root_concepts.csv',
        './change_concepts.csv',
         './keep_concepts.csv']
bac_color =  'white'


#colormap = 'YlOrRd'

# Load data as pandas dataframe


def load_plot_wordclouds(axes, names, colors_names, bac_color):

    for ax, n, k, in zip(axes, names, colors_names):
        height = 190
        print('nananana', ax, n, k)
        df = pd.read_csv(n, sep=',')
        data = dataframe_to_dic(df)
        if k == 'blue':
            print('blue', 'blue')
            height = 420
        # height = 404
        print_wordcloud(ax, df, data, color_dic, k,
                        height, bac_color)  # axes[ax_key]


load_plot_wordclouds(axes, names, colors_names, bac_color)

plt.show()
plt.savefig('Wordcloud_v3'+'.svg')

'''
   def grey_color_func(word, font_size, position, orientation, random_state=None,
                        **kwargs):
        return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 100)

    def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return ("hsl(230,100%%, %d%%)" % np.random.randint(25, 80))

    def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return ("hsl(100,100%%, %d%%)" % int(data['number']/np.max(data['number']*99)))

            # Color words depending on the colors dictionary
    def color_func(word, **kwargs):
        if colors.get(word) == 'g':
            return "rgb(0, 255, 0)"
        else:
            return "rgb(255, 0, 0)"
'''
