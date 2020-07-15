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

df_canada.set_index('Country', inplace=True)
df_canada['Total'] = df_canada.sum(axis=1)
df_canada.sort_values(['Total'], ascending = False, axis = 0,
                      inplace = True)
df_canada.columns = list(map(str, df_canada.columns))
years = list(map(str, range(1980,2014)))

mpl.style.use("ggplot")
#======================================================================
"""
==========
Pie Charts
==========
"""

#'autopct' - a string function used to label the wedges with their
#numeric value. The label will be placed inside the wedge.
#If it is a format string, the label well be fmt%pct.

#'startangle' - rotates the start of the pie chart by angle degrees
#counterclockwise from the x-axis.

#'shadow' - draws a shadow beneath the pie (to give a 3D feel)

#'explode' - emphasize the lowest three continents

df_continents = df_canada.groupby("Continent", axis = 0).sum()
print(df_continents)

colors_list = ["gold", "yellowgreen", "lightcoral", "lightskyblue", "lightgreen", "pink"]
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to
                                        # offset each wedge.
df_continents["Total"].plot(kind="pie",
                            figsize=(5,6),
                            autopct='%1.1f%%', # add in percentages
                            startangle=90,     # start angle 90 (Africa)
                            shadow=True,
                            labels=None,
                            pctdistance=1.12,
                            colors=colors_list,
                            explode=explode_list)

#scale the title up by 12% to match pctdistance
plt.title("Immigration to Canada by Continent [1980 - 2013]", y=1.12)
plt.axis("equal") # Sets the pie chart to look like a circle
plt.show()

#======================================================================
"""
==========
Box Plots
==========
"""

"""A box plot is a way of statistically representing the distribution
of the data through five main dimensions:
1.) Minimum
2.) First quartile: Middle number between the minimum and the median
3.) Second quartile (Median): middle number of the (sorted) dataset
4.) Third quartile: Middle number between median and maximum
5.) Maximum"""

df_japan = df_canada.loc[["Japan"],years].transpose()
df_japan.plot(kind="box", figsize=(8, 6))
plt.title("Box Plot of Japanese Immigrants From 1980 - 2013")
plt.ylabel("Number of Immigrants")
plt.show()
print(df_japan.describe())


#Compare the distribution of the number of new immigrants from
#India and China for the period 1980 - 2013.

df_CI = df_canada.loc[["China", "India"], years].transpose()
df_CI.plot(kind="box", figsize=(8, 6), color="blue", vert=False)
plt.title("Chinese and Indian Immigration From 1980 - 2013")
plt.xlabel("Number of Immigrants")
plt.show()
print(df_CI.describe())

"""
=========
Sub Plots
=========
nrows and ncols are used to notioinally split the figure into
(nrows * ncols) sub-axes.

'plot_number' is used to identify the particular subplot that
this function is to create within the notional grid.
'plot_number' starts at 1, increments across rows first and has
a maximum of nrows * ncols.

We can then specify which subplot to place each plot by passing
in the 'ax' parameter in plot()
"""

fig = plt.figure()
ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1, 2, 2)

#Subplot 1: Box plot
df_CI.plot(kind="box", color="blue", vert=False, figsize=(20, 6), ax=ax0)
ax0.set_title("Box Plots of Immigrants From China and India (1980 - 2013)")
ax0.set_xlabel("Number of Immigrants")
ax0.set_ylabel("Countries")

#Subplot 2: Line plot
df_CI.plot(kind="line", figsize=(20,6), ax=ax1)
ax1.set_title("Line Plots of Immigrants from China and India (1980 - 2013)")
ax1.set_ylabel("Number of Immigrants")
ax1.set_xlabel("Years")

plt.show()

#Create a box plot to visualize the distribution of the top 15 countries
#(based on total immigration) grouped by the decades:
#1980's, 1990's, and 2000's.]

df_top15 = df_canada.sort_values(["Total"], ascending=False, axis=0).head(15)
years_80s = list(map(str, range(1980, 1990)))
years_90s = list(map(str, range(1990, 2000)))
years_00s = list(map(str, range(2000, 2010)))

df_80s = df_top15.loc[:, years_80s].sum(axis=1)
df_90s = df_top15.loc[:, years_90s].sum(axis=1)
df_00s = df_top15.loc[:, years_00s].sum(axis=1)

new_df = pd.DataFrame({"1980s": df_80s, "1990s": df_90s, "2000s": df_00s})
print(new_df.head())

new_df.plot(kind="box", figsize=(10,6))
plt.title("Immigration from top 15 countries for decades 80s, 90s, and 2000s")
plt.show()

#======================================================================
"""
=============
Scatter Plots
=============
"""

df_tot = pd.DataFrame(df_canada[years].sum(axis=0))
print(df_tot.head())
df_tot.index = map(int, df_tot.index)
df_tot.reset_index(inplace = True)
df_tot.columns = ["year", "total"]
print(df_tot.head())

df_tot.plot(kind="scatter", x="year", y="total", figsize=(10, 6), color="darkblue")

plt.title("Total Immigration to Canada From 1980 - 2013")
plt.xlabel("Year")
plt.ylabel("Number of Immigrants")

#Linear Regression
x = df_tot["year"]
y = df_tot["total"]
fit = np.polyfit(x, y, deg=1)
plt.plot(x, fit[0] * x + fit[1], color="red")
plt.annotate("y={0:.0f} x + {1:.0f}".format(fit[0], fit[1]), xy=(2000, 150000))
plt.show()
print(fit)
print("No. Immigrants = {0:.0f} * Year + {1:.0f}".format(fit[0], fit[1]))


#Create a scatter plot of total immigration from Denmark, Norway, and Sweden
#to Canada from 1980 to 2013.

df_countries = df_canada.loc[["Denmark", "Norway", "Sweden"], years].transpose()
df_total = pd.DataFrame(df_countries.sum(axis=1))
df_total.reset_index(inplace=True)
df_total.columns = ["year", "total"]
df_total["year"] = df_total["year"].astype(int)
print(df_total.head())

df_total.plot(kind="scatter", x="year", y="total", figsize=(10, 6), color="darkblue")
plt.title("Immigration from Denmark, Norway, and Sweden to Canada from 1980 - 2013")
plt.xlabel("Year")
plt.ylabel("Number of Immigrants")
plt.show()

#======================================================================
"""
=============
Bubble Plots
=============
"""

df_can_t = df_canada[years].transpose()
df_can_t.index = map(int, df_can_t.index)
df_can_t.index.name = "Year"
df_can_t.reset_index(inplace=True)
print(df_can_t.head())

#Normalization of data:
norm_brazil = (df_can_t["Brazil"] - df_can_t["Brazil"].min()) / (df_can_t["Brazil"].max() - df_can_t["Brazil"].min())

norm_argentina = (df_can_t["Argentina"] - df_can_t["Argentina"].min()) / (df_can_t["Argentina"].min())

#Plotting two different scatter plots on one plot

#Brazil
ax0 = df_can_t.plot(kind="scatter",
                    x="Year",
                    y="Brazil",
                    figsize=(14, 8),
                    alpha=0.5,
                    color="green",
                    s=norm_brazil * 2000 + 10, #pass in weights
                    xlim=(1975, 2015))

#Argentina
ax1 = df_can_t.plot(kind="scatter",
                    x="Year",
                    y="Argentina",
                    alpha=0.5,
                    color="blue",
                    s=norm_argentina * 2000 + 10,
                    ax=ax0)

ax0.set_ylabel("Number of Immigrants")
ax0.set_title("Immigration From Brazil and Argentina From 1980 - 2013")
ax0.legend(["Brazil", "Argentina"], loc="upper left", fontsize="x-large")
plt.show()



