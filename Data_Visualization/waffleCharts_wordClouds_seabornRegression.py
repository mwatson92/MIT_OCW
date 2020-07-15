import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # needed for waffle charts
import numpy as np
import pandas as pd
from PIL import Image # converting images into arrays

mpl.style.use("ggplot")

df_canada = pd.read_excel('Canada.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2)

df_canada.drop(['AREA','REG','DEV','Type','Coverage',],
                  axis=1, inplace=True)

df_canada.rename(columns={'OdName':'Country','AreaName':'Continent',
                       'RegName':'Region'}, inplace=True)

df_canada['Total'] = df_canada.sum(axis=1)
df_canada.set_index('Country', inplace=True)
df_canada.sort_values(['Total'], ascending = False, axis = 0,
                      inplace = True)
df_canada.columns = list(map(str, df_canada.columns))
years = list(map(str, range(1980,2014)))
print(df_canada.shape)

#======================================================================
"""
=============
Waffle Charts
=============
"""

df_dsn = df_canada.loc[["Denmark", "Norway", "Sweden"], :]
print(df_dsn)

def create_waffle_chart(categories, values, height, width,
                        colormap, value_sign=""):

    # Compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values)
                            for value in values]

    # Compute the total number of tiles
    total_num_tiles = width * height
    print("Total number of tiles is", total_num_tiles)

    # Compute the number of tiles for each category
    tiles_per_category = [round(proportion * total_num_tiles)
                          for proportion in category_proportions]

    # Print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print(df_dsn.index.values[i] + ": " + str(tiles))

    # Initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # Define indices to loop through waffle chart
    category_index = 0
    title_index = 0

    # Populate the waffle chart
    for col in range(width):
        for row in range(height):
            title_index += 1

            # if the number of tiles populated for the current category
            # is equal to its corresponding allocated tiles...
            if title_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1

            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index

    # Instantiate a new figure object
    fig = plt.figure()

    # Use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # Get the axis
    ax = plt.gca()

    # Set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

    # Add gridlines based on minor ticks
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # Compute cumulative sum of individual categories to match
    # color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # Create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == "%":
            label_str = category + " (" + str(values[i]) + value_sign + ")"
        else:
            label_str = category + " (" + value_sign + str(values[i]) + ")"

        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # Add legend to chart
    plt.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1))


width = 40
height = 10
categories = df_dsn.index.values
values = df_dsn["Total"]
colormap = plt.cm.coolwarm

create_waffle_chart(categories, values, height, width, colormap)

#======================================================================
"""
============
Word Clouds
============
"""

from wordcloud import WordCloud, STOPWORDS

alice_novel = open("alice_novel.txt", "r").read()
stopwords = set(STOPWORDS)
stopwords.add("said")

# instantiate a word cloud object
alice_wc = WordCloud(
    background_color="white",
    max_words=2000,
    stopwords=stopwords
)


alice_mask = np.array(Image.open("alice_mask.png"))

alice_wc = WordCloud(background_color="white",
                     max_words=2000,
                     mask=alice_mask,
                     stopwords=stopwords)

# generate the word cloud
alice_wc.generate(alice_novel)

# display the word cloud
fig = plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)

plt.imshow(alice_wc, interpolation="bilinear")
plt.axis("off")
plt.show()


# Word cloud for immigration data
total_immigration = df_canada["Total"].sum()
print(total_immigration)
max_words = 90
word_string = ""
for country in df_canada.index.values:
    # check if country's name is a single-word name
    if len(country.split(' ')) == 1:
        repeat_num_times = int(df_canada.loc[country, "Total"]/float(total_immigration)*max_words)
        word_string = word_string + ((country + ' ') * repeat_num_times)

# Display the generated text
print(word_string)

# Create the word cloud
wordcloud = WordCloud(background_color="white").generate(word_string)

# Display the cloud
fig = plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#======================================================================
"""
================
Regression Plots
================
"""

import seaborn as sns

# Create a new dataframe that stores the total number of landed
# immigrants to Canada per year from 1980 to 2013.

df_tot = pd.DataFrame(df_canada[years].sum(axis=0))
df_tot.index = map(float, df_tot.index)
df_tot.reset_index(inplace=True)
df_tot.columns = ["year", "total"]
print(df_tot.head())

plt.figure(figsize=(15, 10))
sns.set(font_scale=1.5)
ax = sns.regplot(x="year", y="total",
                 data=df_tot, color="green", marker="+",
                 scatter_kws={"s": 200})

ax.set(xlabel="Year", ylabel="Total Immigration")
ax.set_title("Total Immigration to Canada From 1980 - 2013")
plt.show()
