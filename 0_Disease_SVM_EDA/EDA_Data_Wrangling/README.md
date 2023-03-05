# Data Science - EDA & Data Wrangling

# 1) Data Wrangling 

Data manipulation is another way to refer to this process is data manipulation but there is no set list or order of operations. However, there are three common tasks involved in the data wrangling process such as:
  - **Data cleaning**
  - **Data transformation**
  - **Data enrichment**

Useful techniques used to clean and process the data is with **Pandas library**. Pandas is a powerful toolkit for analyzing initial data and introducing dataset structures in Python. Activating Pandas is very easy in python. As one library to do the initial process of data analysis. 
Let's explore data and its types.

- **The overview of the dataset**

```python 
import pandas as pd
import seaborn as sns

df = pd.read_csv('/disease.data')

# to look at a small sample of the dataset at the top
df.head()

# to look at a small sample of the dataset at the end
df.tail()

# have a look at a subset of the rows or columns fx: select the first 10 columns
df.iloc[:,:10].head()

# shows the data type for each column, among other things
df.info()

# shows the data type for each column
df.dtypes()

# describe() gives the insights about the data and some useful statistics about the data such as mean, min and max etc.
df.describe()
```
- **Missing and duplicate value**

The dataset may consist of a lot of missing and duplicate values, so let's deal with them before applying any machine learning algorithms on them. If you have identified the missing values in the dataset, now you have a couple of options to deal with them, either we can drop those rows which consist missing values or calculate the mean, min, max and median etc.
  
```python
# dealing with missing values
df.isna().sum()

# Visualize the missing values using Missingno library. 
msno.matrix(df)
```

<div align="center">
 Visualize the missing values
</div>

<p align="center">
  <img width="400" height="200" src="https://github.com/sulova/Data_Science_Disease_SVM/blob/main/Sk%C3%A6rmbillede%202021-03-04%20212910.png ">
</p>

```python

# fill in the missing values in 'ColumnName'
ColumnName_mean_value = df['ColumnName'].mean()
df['ColumnName'] = df['ColumnName'].fillna(ColumnName_mean_value)

Replace the NaN with median.

# Remove 'ColumnName' column
df.drop("ColumnName",axis = 1,inplace=True)
# Drops all columns in the DataFrame that have more than 10% Null values
df.dropna(1,thresh=len(df.index)*0.9,inplace=True)

# List down all the duplicated rows in the dataframe
duplicate_rows_df = df[df.duplicated()]
print('number of duplicate rows:', duplicate_rows_df.shape)
# Remove those rows 
df.drop_duplicates(inplace=False) 


# Get rid of all non-unique columns in a dataset
nunique = df.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
df.drop(cols_to_drop, axis=1,inplace=True)

# Check if any existing column has all null values
df.isnull().values.all(axis=0)
# Replace all the null values in the column with a zero
df['ColumnName'].fillna(0, inplace = True)

# Define in which columns to look for missing values
df.dropna(subset=['ColumnName_1', 'ColumnName_1'])
# Drop the rows where all elements are missing
df.dropna(how='all')
# Keep only the rows with at least 2 non-NA values
df.dropna(thresh=2)
```

- **Correct Data type for a column**

```python
# Conversion between data types for a column. 
 df['ColumnName'].astype()

# Change the labeling for better visualization and interpretation.**
df['target'] = df.target.replace({1: "Disease", 0: "No_disease"})
df['sex'] = df.sex.replace({1: "Male", 0: "Female"})

``` 

- **Filtering Data**

 ```python
# The value of the Country column is not USA
df[df[‘Country’]!=’USA’]

# The following piece of code filters the entire dataset for age greater than 40
filtered_age = df[df.Age>40]
filtered_age

# Grouping/Aggregating the values of the Customer column based on the Country column
df.groupby([‘Country’]).CustomerNo.count()

#  Merging the Customer with the Location DataFrame using a left join
pd.merge(customer,location,how=’left’,on=[‘city’])

``` 
A good data wrangler knows how to integrate information from multiple data sources, solving common transformation problems, and resolve data cleansing and quality issues.

# 2) EDA - Exploratory Data Analysis
Exploratory Data Analysis (EDA) is a pre-processing step to understand the data. There are numerous methods and steps in performing EDA, however, most of them are specific, focusing on either visualization or distribution, and are incomplete. 
It is a good practice to understand the data first and EDA refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.

**Seaborn Versus Matplotlib** 
*Matplotlib* - classic plot formatting,  looks a bit old-fashioned in the context of 21st-century data visualization.
*Seaborn* - high-level plotting routines, to produce vastly superior output. 

Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. We can set the style by calling Seaborn's *set()* method. By convention, Seaborn is imported as *sns*:

Now we will make necessary imports and try to load the dataset to jupyter notebook.

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```

- **Scatter Plot**
The scatter plot is useful when we want to show the relation between two features or a feature and the label. It is useful as we can also describe the size of each data point, color them differently and use different markers

```python
# Scatter plot showingrelation between two features 
sns.scatterplot(x = 'room', y = 'bedroom', data = df)

# Using figsize- increasing size of plot and scatterplot based on differen values. 
plt.figure(figsize = (12, 8))
sns.scatterplot(data = df,
               x = 'rooms', 
               y = 'bedrooms',
               hue = 'ocean_proximity', 
               style = 'ocean_proximity')
plt.title("Rooms vs Bedrooms")
plt.xlabel("Total rooms")
plt.ylabel("Total bedrooms")
```

- **Bar plot - Count Plot**

```python
# Count the data points based on a certain categorical column
plt.figure(figsize = (12, 8))
ocean_plot = sns.countplot(x = 'ocean_proximity', data = dataset)
for p in ocean_plot.patches:
    ocean_plot.annotate(p.get_height(), 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                        ha = 'center', 
                        va = 'center', 
                        xytext = (0, 5),
                        textcoords = 'offset points')

plt.title("Count of houses based on their proximity to ocean")
plt.xlabel("Proximity to the ocean")
plt.ylabel("Count of houses")
```
- **Histograms - Count Plot**
Histograms are an effective way to show *CONTINUOUS points* of data and see how they are distributed. we could see if most values are to the lower side, or to the higher side or evenly distributed. We want is to plot histograms and joint distributions of variables. 

```python
plt.figure(figsize = (12, 8))
sns.distplot(a = dataset['median_house_value'], bins = 10, hist = True)
plt.title("Density and histogram plot for Median house value")
plt.xlabel("Median house value")
plt.ylabel("Value")
```

- **Violin Plot**
Violin plots are quite similar to box plots and depict the **width** based on the density to reflect the data distribution.
```python
plt.figure(figsize = (12, 8))
sns.violinplot(x = 'ocean_proximity', y = 'median_house_value', data = dataset)
plt.title("Box plots of house values based on ocean proximity")
plt.xlabel("Ocean proximity")
plt.ylabel("Median house value")
```

- **Joint Plot**
A joint plot is a combination of scatter plot along with the density plots (histograms) for both features we’re trying to plot. The seaborn’s joint plot allows us to even plot a linear regression all by itself using kind as reg. I defined the square dimensions using height as 8 and color as green.
The green line depicts the linear regression based on the data points.

```python
sns.jointplot(x = "total_rooms", y = "total_bedrooms", data=dataset, kind="reg", height = 8, color = 'g')
plt.xlabel("Total rooms")
plt.ylabel("Total bedrooms")
```
- **Box Plot with Swarm Plots**
The box plots present the information into separate quartiles as well as the median. When overlapped with swarm plot, the data points are spread across their location such that there is no overlapping at all.

```python
plt.figure(figsize = (12, 8))
sns.boxplot(x = 'ocean_proximity', y = 'median_house_value', data = dataset)
sns.swarmplot(x = 'ocean_proximity', y = 'median_house_value', data = dataset)
plt.title("Box plots of house values based on ocean proximity")
plt.xlabel("Ocean proximity")
plt.ylabel("Median house value")
```

- **Swarmplots**
jitter can be used to randomly provide displacements along the horizontal axis, which is useful when there are large clusters of datapoints
```python
p = sns.stripplot(data=df,
                  x='player_name',
                  y='SHOT_DIST',
                  hue='SHOT_RESULT',
                  order=sorted(players_to_use),
                  jitter=0.25,
                  dodge=True,
                  palette=sns.husl_palette(2, l=0.5, s=.95))
```
Swarmplots look good when overlaid on top of another categorical plot, like boxplot.

```python
params = dict(data=df,
              x='player_name',
              y='SHOT_DIST',
              hue='SHOT_RESULT',
              #jitter=0.25,
              order=sorted(players_to_use),
              dodge=True)
p = sns.stripplot(size=8,
                  jitter=0.35,
                  palette=['#91bfdb','#fc8d59'],
                  edgecolor='black',
                  linewidth=1,
                  **params)
p_box = sns.boxplot(palette=['#BBBBBB','#DDDDDD'],linewidth=6,**params)
```
