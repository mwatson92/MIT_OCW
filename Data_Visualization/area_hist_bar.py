"""
Area, Histogram, and Bar Charts
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
#print(df_canada.head())

years = list(map(str, range(1980,2014)))
df_top5 = df_canada.head(5)
df_top5 = df_top5[years].transpose()
print(df_top5.head())

#======================================================================
"""
=========
Area Plot
=========
"""

#change the indices to int for plotting
df_top5.index = df_top5.index.map(int)

#Note that we can produce a stacked or unstacked plot by
#changing the 'stacked' parameter.
df_top5.plot(kind='area', stacked=True)
plt.title("Immigration trend of top 5 countries")
plt.ylabel("Number of immigrants")
plt.xlabel("Years")
plt.show()

#The unstacked plot has a default transparency (alpha value)
#at 0.5. We can modify this with the 'alpha' parameter.
df_top5.plot(kind='area', alpha=0.25, stacked=False, figsize=(20,10))
plt.show()
#======================================================================
"""
==========
Histograms
==========
"""

"""Initially, the bars in a histogram do not align centrally
with the ticks on the x-axis. To fix this, we use
the numpy library and calling plot() with the xticks parameter."""

#np.histogram() by default separates the data into 10 bins.
count, bin_edges = np.histogram(df_canada['2013'])
#df_canada['2013'].plot(kind='hist')
df_canada['2013'].plot(kind='hist', xticks = bin_edges)
plt.title("Histogram of immigration from 195 countries in 2013")
plt.ylabel("Number of Countries")
plt.xlabel("Number of Immigrants")
plt.show()

#We can also plot multiple histograms on the same plot.
print(df_canada.loc[['Denmark','Norway','Sweden'], years])
#df_canada.loc[['Denmark','Norway','Sweden'], years].plot.hist()

df_t = df_canada.loc[['Denmark','Norway','Sweden'], years].transpose()
print(df_t.head())

count, bin_edges = np.histogram(df_t, 15)
df_t.plot(kind='hist',
          figsize=(10, 6),
          bins=15,
          alpha=0.6,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen'])

plt.title("Histogram of Immigration from Denmark, Norway, and Sweden")
plt.ylabel("Number of Years")
plt.xlabel("Number of Immigrants")
plt.show()

#If we don't want the plots to overlap, we can stack them
#with the 'stacked' parameter. We can also remove the extra gaps
#on the edges of the plot by adjusting the min and max x-axis labels.

xmin = bin_edges[0] - 10   #first bin value is 31.0, adding buffer of 10
                           #for aesthetic purposes.
xmax = bin_edges[-1] + 10  #last bin value is 308.0, adding buffer of 10
                           #for aesthetic purposes.

df_t.plot(kind='hist',
          figsize=(10, 6),
          bins=15,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen'],
          stacked=True,
          xlim=(xmin, xmax))

plt.title("Histogram of Immigration from Denmark, Norway, and Sweden")
plt.ylabel("Number of Years")
plt.xlabel("Number of Immigrants")
plt.show()


#======================================================================
"""
===========
Bar Charts
===========
"""

df_iceland = df_canada.loc['Iceland',years]
print(df_iceland)

df_iceland.plot(kind='bar', figsize=(10,6))
plt.title("Icelandic immigrants to Canada from 1980 to 2013")
plt.xlabel("Year")
plt.ylabel("Number of immigrants")

#Annotate arrow:
plt.annotate('',                #s: str. Will leave it blank for no text
             xy=(32, 70),       #put head of arrw @ pt(year2012, pop 70)
             xytext=(28,20),    #put base of arrw @ pt(yr2008, pop 20)
             xycoords='data',   #use coord sys of object being annot.
             arrowprops=dict(arrowstyle='->',
                             connectionstyle='arc3',
                             color='blue',
                             lw=2))

#Annotate text
plt.annotate('2008 - 2011 Financial Crisis',
             xy=(28, 30),
             rotation=72.5,
             va='bottom',   #vertically bottom aligned
             ha='left')     #horizontally left aligned

plt.show()

