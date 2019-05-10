
## Machine Learning with Spark - Lab

## Introduction

In the previous lecture, you were shown how to manipulate data with Spark DataFrames as well as create machine learning models. In this lab, you're going to practice loading data, manipulating it, and fitting it into the Spark Framework. Afterwords, you're going to make use of different visualizations to see if you can get any insights from the model. Let's get started!

### Objectives

* Create machine learning pipeline with pyspark
* Evaluate a model with pyspark
* Create and interpret visualizations with pyspark


```python
import pyspark
```


```python
from pyspark import SparkContext
from pyspark.sql import SparkSession

```


```python
sc = SparkContext('local[*]')
spark = SparkSession(sc)
```


```python
spark_df = spark.read.csv('./credit_card_default.csv',header='true',inferSchema='true')
```


```python
spark_df.dtypes
```




    [('ID', 'int'),
     ('LIMIT_BAL', 'double'),
     ('SEX', 'string'),
     ('EDUCATION', 'string'),
     ('MARRIAGE', 'string'),
     ('AGE', 'int'),
     ('PAY_0', 'int'),
     ('PAY_2', 'int'),
     ('PAY_3', 'int'),
     ('PAY_4', 'int'),
     ('PAY_5', 'int'),
     ('PAY_6', 'int'),
     ('BILL_AMT1', 'double'),
     ('BILL_AMT2', 'double'),
     ('BILL_AMT3', 'double'),
     ('BILL_AMT4', 'double'),
     ('BILL_AMT5', 'double'),
     ('BILL_AMT6', 'double'),
     ('PAY_AMT1', 'double'),
     ('PAY_AMT2', 'double'),
     ('PAY_AMT3', 'double'),
     ('PAY_AMT4', 'double'),
     ('PAY_AMT5', 'double'),
     ('PAY_AMT6', 'double'),
     ('default', 'int')]




```python
from pyspark.sql import functions
```


```python
for col in spark_df.columns:
    print('column',col,spark_df.filter(spark_df[col].isNull()).count())
```

    column ID 0
    column LIMIT_BAL 0
    column SEX 0
    column EDUCATION 0
    column MARRIAGE 0
    column AGE 0
    column PAY_0 0
    column PAY_2 0
    column PAY_3 0
    column PAY_4 0
    column PAY_5 0
    column PAY_6 0
    column BILL_AMT1 0
    column BILL_AMT2 0
    column BILL_AMT3 0
    column BILL_AMT4 0
    column BILL_AMT5 0
    column BILL_AMT6 0
    column PAY_AMT1 0
    column PAY_AMT2 0
    column PAY_AMT3 0
    column PAY_AMT4 0
    column PAY_AMT5 0
    column PAY_AMT6 0
    column default 0



```python
for column , data_type in spark_df.dtypes:
    if data_type == 'string':
        print('Feature ',column,' has: ', spark_df.select(column).distinct().collect())
```

    Feature  SEX  has:  [Row(SEX='Female'), Row(SEX='Male')]
    Feature  EDUCATION  has:  [Row(EDUCATION='High School'), Row(EDUCATION='0'), Row(EDUCATION='5'), Row(EDUCATION='6'), Row(EDUCATION='Other'), Row(EDUCATION='Graduate'), Row(EDUCATION='College')]
    Feature  MARRIAGE  has:  [Row(MARRIAGE='0'), Row(MARRIAGE='Other'), Row(MARRIAGE='Married'), Row(MARRIAGE='Single')]


Interesting... it looks like we have some extraneous values in each of our categories. Let's look at some visualizations of each of these to determine just how many of them there are. Create histograms of the variables EDUCATION and MARRIAGE to see how many of the undefined values there are. After doing so, come up with a strategy for accounting for the extra value.


```python
education_cats = spark_df.groupBy('EDUCATION').count().collect()
sns.barplot(x=bar_plot_values(0,education_cats),y=bar_plot_values(1,education_cats))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1960d7b8>




![png](index_files/index_10_1.png)



```python
marriage_cats =  spark_df.groupby('MARRIAGE').count().collect()
sns.barplot(x=bar_plot_values(0, marriage_cats), y=bar_plot_values(1, marriage_cats))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a17c63cc0>




![png](index_files/index_11_1.png)


It looks like there are barely any of the categories of 0 and 5 categories. We can go ahead and throw them into the "Other" category since it's already operating as a catchall here. Similarly, the category "0" looks small, so let's throw it in with the "Other" values.


```python
x = spark_df.select('EDUCATION') == '0'
x
```




    False




```python
from pyspark.sql.functions import when

# conditional = spark_df['EDUCATION'] == '0' |  spark_df['EDUCATION'] == '5' | spark_df['EDUCATION'] == '6'
spark_df_2 = spark_df.withColumn("EDUCATION",
                    when(spark_df.EDUCATION == '0','Other')\
                    .when(spark_df.EDUCATION == '5','Other')\
                    .when(spark_df.EDUCATION == '6','Other')\
                    .otherwise(spark_df['EDUCATION']))

spark_df_done = spark_df_2.withColumn("MARRIAGE",
                                   when(spark_df.MARRIAGE == '0','Other')\
                                   .otherwise(spark_df['MARRIAGE']))
```


```python
for column , data_type in spark_df_done.dtypes:
    if data_type == 'string':
        print('Feature ',column,' has: ', spark_df_done.select(column).distinct().collect())
```

    Feature  SEX  has:  [Row(SEX='Female'), Row(SEX='Male')]
    Feature  EDUCATION  has:  [Row(EDUCATION='High School'), Row(EDUCATION='Other'), Row(EDUCATION='Graduate'), Row(EDUCATION='College')]
    Feature  MARRIAGE  has:  [Row(MARRIAGE='Other'), Row(MARRIAGE='Married'), Row(MARRIAGE='Single')]


Much better. Now, let's do a little more EDA before diving into the machine learning aspect of this project.

## EDA

Let's first look at the overall distribution of class imbalance to determine if there is a need for each one of the different things here.


```python
number_of_defaults = spark_df.groupBy('default').count().collect()

```


```python
default = [x[0] for x in number_of_defaults]
num_defaults = [x[1] for x in number_of_defaults]
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.barplot(default,num_defaults,tick_label=['No Default','Default'])
# ax.xlabel('Default')
# ax.ylabel('Number of Defaults')
ax.set_xlabel('Default vs. Not Default')
ax.set_ylabel('Number of Defaults')
```




    Text(0,0.5,'Number of Defaults')




![png](index_files/index_20_1.png)



```python
## a reasonable class balance, no real issues with class imbalances
```


```python
results = spark_df.groupBy(['SEX','default']).count().collect()
```


```python
results
```




    [Row(SEX=1, default=0, count=9015),
     Row(SEX=1, default=1, count=2873),
     Row(SEX=2, default=1, count=3762),
     Row(SEX=2, default=0, count=14349)]




```python
def bar_plot_values(idx,group):
    return [x[idx] for x in group]
```


```python
defaulted = results[:2]
not_defaulted = results[2:]

```


```python
bar_plot_values(1,defaulted)
```




    [1, 1]




```python
f, axes = plt.subplots(1,2)
f.set_figwidth(10)
sns.barplot(x= bar_plot_values(1,defaulted),y=bar_plot_values(2,defaulted),ax=axes[0])
sns.barplot(x= bar_plot_values(1,not_defaulted),y=bar_plot_values(2,not_defaulted),ax=axes[1])


```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1818dc50>




![png](index_files/index_27_1.png)



```python
results = spark_df.groupBy(['SEX','default']).count().collect()
```


```python
import seaborn as sns
sns.categorical.boxplot()
```

## Onto the Machine Learning!


```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StringIndexerModel
```


```python
l = [3,4,5,]
t = [4,5,3]
c = [6,2,8]
l += [t,c]

l
```




    [3, 4, 5, [4, 5, 3], [6, 2, 8]]




```python
stages = []
indexers = []

for col in ['EDUCATION','SEX','MARRIAGE']:
    indexers.append(StringIndexer(inputCol =col,outputCol=col+'_',handleInvalid='keep'))
    
input_columns = [indexer.getOutputCol() for indexer in indexers]

one_hot_encoder = OneHotEncoderEstimator(inputCols=input_columns,outputCols=[col + 'ohe' for col in input_columns])

features = ['LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3',
            'PAY_4','PAY_5','PAY_6', 'BILL_AMT1','BILL_AMT2',
            'BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']

features.extend(one_hot_encoder.getOutputCols())

vector_assember = VectorAssembler(inputCols= features , outputCol='features')

stages.extend(indexers)
stages.extend([one_hot_encoder,vector_assember])
print(stages)
```

    [StringIndexer_4a1ebab6194451f0ef0e, StringIndexer_441fb14e4ffbca71828c, StringIndexer_4814b38309b2c37240d2, OneHotEncoderEstimator_422a926e5bfe35ae5a07, VectorAssembler_47d0a99f6577b8dcce9a]


Now let's create all the stages here to make 


```python
x.getOutputCol()
```




    'EDUCATION_'




```python
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler

ohe = OneHotEncoderEstimator(inputCols=['SEX','EDUCATION','MARRIAGE'],outputCols=['SEX_c','EDUCATION_c','MARRIAGE_c'])
```


```python
ohe.fit(spark_df)
```




    OneHotEncoderEstimator_4cdd8e04d5de9b43dc2e




```python
import pandas as pd
```


```python
df = pd.read_csv('./credit_default.csv')
```


```python
df.loc[df['SEX'] == 2 ,'SEX'] = 'Female'
df.loc[df['SEX'] == 1 ,'SEX'] = 'Male'
```


```python
df.loc[df['EDUCATION'] == 1,'EDUCATION'] = 'Graduate'
df.loc[df['EDUCATION'] == 2,'EDUCATION'] = 'College'
df.loc[df['EDUCATION'] == 3,'EDUCATION'] = 'High School'
df.loc[df['EDUCATION'] == 4,'EDUCATION'] = 'Other'
```


```python
df.loc[df['MARRIAGE'] == 1,'MARRIAGE'] = 'Married'
df.loc[df['MARRIAGE'] == 2,'MARRIAGE'] = 'Single'
df.loc[df['MARRIAGE'] == 3,'MARRIAGE'] = 'Other'


```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>120000.0</td>
      <td>Female</td>
      <td>College</td>
      <td>Single</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272.0</td>
      <td>3455.0</td>
      <td>3261.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>90000.0</td>
      <td>Female</td>
      <td>College</td>
      <td>Single</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331.0</td>
      <td>14948.0</td>
      <td>15549.0</td>
      <td>1518.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>5000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>50000.0</td>
      <td>Female</td>
      <td>College</td>
      <td>Married</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314.0</td>
      <td>28959.0</td>
      <td>29547.0</td>
      <td>2000.0</td>
      <td>2019.0</td>
      <td>1200.0</td>
      <td>1100.0</td>
      <td>1069.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>50000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940.0</td>
      <td>19146.0</td>
      <td>19131.0</td>
      <td>2000.0</td>
      <td>36681.0</td>
      <td>10000.0</td>
      <td>9000.0</td>
      <td>689.0</td>
      <td>679.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>50000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>19394.0</td>
      <td>19619.0</td>
      <td>20024.0</td>
      <td>2500.0</td>
      <td>1815.0</td>
      <td>657.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>800.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
df.to_csv('credit_card_default.csv')
```


```python
pd.read_csv('./credit_card_default.csv')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>120000.0</td>
      <td>Female</td>
      <td>College</td>
      <td>Single</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272.0</td>
      <td>3455.0</td>
      <td>3261.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>90000.0</td>
      <td>Female</td>
      <td>College</td>
      <td>Single</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331.0</td>
      <td>14948.0</td>
      <td>15549.0</td>
      <td>1518.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>5000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>50000.0</td>
      <td>Female</td>
      <td>College</td>
      <td>Married</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314.0</td>
      <td>28959.0</td>
      <td>29547.0</td>
      <td>2000.0</td>
      <td>2019.0</td>
      <td>1200.0</td>
      <td>1100.0</td>
      <td>1069.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>50000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940.0</td>
      <td>19146.0</td>
      <td>19131.0</td>
      <td>2000.0</td>
      <td>36681.0</td>
      <td>10000.0</td>
      <td>9000.0</td>
      <td>689.0</td>
      <td>679.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>50000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>19394.0</td>
      <td>19619.0</td>
      <td>20024.0</td>
      <td>2500.0</td>
      <td>1815.0</td>
      <td>657.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>800.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>500000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>542653.0</td>
      <td>483003.0</td>
      <td>473944.0</td>
      <td>55000.0</td>
      <td>40000.0</td>
      <td>38000.0</td>
      <td>20239.0</td>
      <td>13750.0</td>
      <td>13770.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>100000.0</td>
      <td>Female</td>
      <td>College</td>
      <td>Single</td>
      <td>23</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>221.0</td>
      <td>-159.0</td>
      <td>567.0</td>
      <td>380.0</td>
      <td>601.0</td>
      <td>0.0</td>
      <td>581.0</td>
      <td>1687.0</td>
      <td>1542.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>140000.0</td>
      <td>Female</td>
      <td>High School</td>
      <td>Married</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>12211.0</td>
      <td>11793.0</td>
      <td>3719.0</td>
      <td>3329.0</td>
      <td>0.0</td>
      <td>432.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>20000.0</td>
      <td>Male</td>
      <td>High School</td>
      <td>Single</td>
      <td>35</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>...</td>
      <td>0.0</td>
      <td>13007.0</td>
      <td>13912.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13007.0</td>
      <td>1122.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>200000.0</td>
      <td>Female</td>
      <td>High School</td>
      <td>Single</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2513.0</td>
      <td>1828.0</td>
      <td>3731.0</td>
      <td>2306.0</td>
      <td>12.0</td>
      <td>50.0</td>
      <td>300.0</td>
      <td>3738.0</td>
      <td>66.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12</td>
      <td>260000.0</td>
      <td>Female</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>51</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>8517.0</td>
      <td>22287.0</td>
      <td>13668.0</td>
      <td>21818.0</td>
      <td>9966.0</td>
      <td>8583.0</td>
      <td>22301.0</td>
      <td>0.0</td>
      <td>3640.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13</td>
      <td>630000.0</td>
      <td>Female</td>
      <td>College</td>
      <td>Single</td>
      <td>41</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>6500.0</td>
      <td>6500.0</td>
      <td>2870.0</td>
      <td>1000.0</td>
      <td>6500.0</td>
      <td>6500.0</td>
      <td>6500.0</td>
      <td>2870.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>14</td>
      <td>70000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Single</td>
      <td>30</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>66782.0</td>
      <td>36137.0</td>
      <td>36894.0</td>
      <td>3200.0</td>
      <td>0.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>1500.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>15</td>
      <td>250000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>59696.0</td>
      <td>56875.0</td>
      <td>55512.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>16</td>
      <td>50000.0</td>
      <td>Female</td>
      <td>High School</td>
      <td>Other</td>
      <td>23</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28771.0</td>
      <td>29531.0</td>
      <td>30211.0</td>
      <td>0.0</td>
      <td>1500.0</td>
      <td>1100.0</td>
      <td>1200.0</td>
      <td>1300.0</td>
      <td>1100.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>17</td>
      <td>20000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>18338.0</td>
      <td>17905.0</td>
      <td>19104.0</td>
      <td>3200.0</td>
      <td>0.0</td>
      <td>1500.0</td>
      <td>0.0</td>
      <td>1650.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>18</td>
      <td>320000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Married</td>
      <td>49</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>...</td>
      <td>70074.0</td>
      <td>5856.0</td>
      <td>195599.0</td>
      <td>10358.0</td>
      <td>10000.0</td>
      <td>75940.0</td>
      <td>20000.0</td>
      <td>195599.0</td>
      <td>50000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>19</td>
      <td>360000.0</td>
      <td>Female</td>
      <td>Graduate</td>
      <td>Married</td>
      <td>49</td>
      <td>1</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>20</td>
      <td>180000.0</td>
      <td>Female</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>29</td>
      <td>1</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>21</td>
      <td>130000.0</td>
      <td>Female</td>
      <td>High School</td>
      <td>Single</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>20616.0</td>
      <td>11802.0</td>
      <td>930.0</td>
      <td>3000.0</td>
      <td>1537.0</td>
      <td>1000.0</td>
      <td>2000.0</td>
      <td>930.0</td>
      <td>33764.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>22</td>
      <td>120000.0</td>
      <td>Female</td>
      <td>College</td>
      <td>Married</td>
      <td>39</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0.0</td>
      <td>632.0</td>
      <td>316.0</td>
      <td>316.0</td>
      <td>316.0</td>
      <td>0.0</td>
      <td>632.0</td>
      <td>316.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>23</td>
      <td>70000.0</td>
      <td>Female</td>
      <td>College</td>
      <td>Single</td>
      <td>26</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>44006.0</td>
      <td>46905.0</td>
      <td>46012.0</td>
      <td>2007.0</td>
      <td>3582.0</td>
      <td>0.0</td>
      <td>3601.0</td>
      <td>0.0</td>
      <td>1820.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>24</td>
      <td>450000.0</td>
      <td>Female</td>
      <td>Graduate</td>
      <td>Married</td>
      <td>40</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>...</td>
      <td>560.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19428.0</td>
      <td>1473.0</td>
      <td>560.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1128.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>25</td>
      <td>90000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>...</td>
      <td>5398.0</td>
      <td>6360.0</td>
      <td>8292.0</td>
      <td>5757.0</td>
      <td>0.0</td>
      <td>5398.0</td>
      <td>1200.0</td>
      <td>2045.0</td>
      <td>2000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>26</td>
      <td>50000.0</td>
      <td>Male</td>
      <td>High School</td>
      <td>Single</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28967.0</td>
      <td>29829.0</td>
      <td>30046.0</td>
      <td>1973.0</td>
      <td>1426.0</td>
      <td>1001.0</td>
      <td>1432.0</td>
      <td>1062.0</td>
      <td>997.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>27</td>
      <td>60000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>27</td>
      <td>1</td>
      <td>-2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>-57.0</td>
      <td>127.0</td>
      <td>-189.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>500.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>28</td>
      <td>50000.0</td>
      <td>Female</td>
      <td>High School</td>
      <td>Single</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>17878.0</td>
      <td>18931.0</td>
      <td>19617.0</td>
      <td>1300.0</td>
      <td>1300.0</td>
      <td>1000.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1012.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>29</td>
      <td>50000.0</td>
      <td>Female</td>
      <td>High School</td>
      <td>Married</td>
      <td>47</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>2040.0</td>
      <td>30430.0</td>
      <td>257.0</td>
      <td>3415.0</td>
      <td>3421.0</td>
      <td>2044.0</td>
      <td>30430.0</td>
      <td>257.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>30</td>
      <td>50000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>17907.0</td>
      <td>18375.0</td>
      <td>11400.0</td>
      <td>1500.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1600.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>31</td>
      <td>230000.0</td>
      <td>Female</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>27</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>15339.0</td>
      <td>14307.0</td>
      <td>36923.0</td>
      <td>17270.0</td>
      <td>13281.0</td>
      <td>15339.0</td>
      <td>14307.0</td>
      <td>37292.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29969</th>
      <td>29971</td>
      <td>360000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Married</td>
      <td>34</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>49005.0</td>
      <td>8676.0</td>
      <td>19487.0</td>
      <td>52951.0</td>
      <td>64535.0</td>
      <td>8907.0</td>
      <td>53.0</td>
      <td>19584.0</td>
      <td>16080.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29970</th>
      <td>29972</td>
      <td>80000.0</td>
      <td>Male</td>
      <td>High School</td>
      <td>Married</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>69674.0</td>
      <td>71070.0</td>
      <td>73612.0</td>
      <td>2395.0</td>
      <td>2500.0</td>
      <td>2530.0</td>
      <td>2556.0</td>
      <td>3700.0</td>
      <td>3000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29971</th>
      <td>29973</td>
      <td>190000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Married</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>29223.0</td>
      <td>19616.0</td>
      <td>148482.0</td>
      <td>2000.0</td>
      <td>3869.0</td>
      <td>25128.0</td>
      <td>10115.0</td>
      <td>148482.0</td>
      <td>4800.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29972</th>
      <td>29974</td>
      <td>230000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>35</td>
      <td>1</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29973</th>
      <td>29975</td>
      <td>50000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2846.0</td>
      <td>1585.0</td>
      <td>1324.0</td>
      <td>0.0</td>
      <td>3000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29974</th>
      <td>29976</td>
      <td>220000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>5924.0</td>
      <td>1759.0</td>
      <td>1824.0</td>
      <td>8840.0</td>
      <td>6643.0</td>
      <td>5924.0</td>
      <td>1759.0</td>
      <td>1824.0</td>
      <td>7022.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29975</th>
      <td>29977</td>
      <td>40000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Single</td>
      <td>47</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>51259.0</td>
      <td>47151.0</td>
      <td>46934.0</td>
      <td>4000.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>0.0</td>
      <td>3520.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29976</th>
      <td>29978</td>
      <td>420000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>141695.0</td>
      <td>144839.0</td>
      <td>147954.0</td>
      <td>7000.0</td>
      <td>7000.0</td>
      <td>5500.0</td>
      <td>5500.0</td>
      <td>5600.0</td>
      <td>5000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29977</th>
      <td>29979</td>
      <td>310000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>219409.0</td>
      <td>216540.0</td>
      <td>210675.0</td>
      <td>10029.0</td>
      <td>9218.0</td>
      <td>10029.0</td>
      <td>8049.0</td>
      <td>8040.0</td>
      <td>10059.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29978</th>
      <td>29980</td>
      <td>180000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Married</td>
      <td>32</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29979</th>
      <td>29981</td>
      <td>50000.0</td>
      <td>Male</td>
      <td>High School</td>
      <td>Single</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>50360.0</td>
      <td>19971.0</td>
      <td>19694.0</td>
      <td>10000.0</td>
      <td>4000.0</td>
      <td>5000.0</td>
      <td>3000.0</td>
      <td>4500.0</td>
      <td>2000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29980</th>
      <td>29982</td>
      <td>50000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>44</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>28192.0</td>
      <td>22676.0</td>
      <td>14647.0</td>
      <td>2300.0</td>
      <td>1700.0</td>
      <td>0.0</td>
      <td>517.0</td>
      <td>503.0</td>
      <td>585.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29981</th>
      <td>29983</td>
      <td>90000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>11328.0</td>
      <td>12036.0</td>
      <td>14329.0</td>
      <td>1500.0</td>
      <td>1500.0</td>
      <td>1500.0</td>
      <td>1200.0</td>
      <td>2500.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29982</th>
      <td>29984</td>
      <td>20000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>44</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>...</td>
      <td>2882.0</td>
      <td>9235.0</td>
      <td>1719.0</td>
      <td>2890.0</td>
      <td>2720.0</td>
      <td>2890.0</td>
      <td>9263.0</td>
      <td>1824.0</td>
      <td>1701.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29983</th>
      <td>29985</td>
      <td>30000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Single</td>
      <td>38</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-1</td>
      <td>...</td>
      <td>1993.0</td>
      <td>1907.0</td>
      <td>3319.0</td>
      <td>923.0</td>
      <td>2977.0</td>
      <td>1999.0</td>
      <td>3057.0</td>
      <td>3319.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29984</th>
      <td>29986</td>
      <td>240000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>30</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29985</th>
      <td>29987</td>
      <td>360000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>35</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29986</th>
      <td>29988</td>
      <td>130000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>108047.0</td>
      <td>93708.0</td>
      <td>97353.0</td>
      <td>3000.0</td>
      <td>2000.0</td>
      <td>93000.0</td>
      <td>4000.0</td>
      <td>5027.0</td>
      <td>4005.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29987</th>
      <td>29989</td>
      <td>250000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Married</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>245750.0</td>
      <td>175005.0</td>
      <td>179687.0</td>
      <td>65000.0</td>
      <td>8800.0</td>
      <td>9011.0</td>
      <td>6000.0</td>
      <td>7000.0</td>
      <td>6009.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29988</th>
      <td>29990</td>
      <td>150000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>35</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>780.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9054.0</td>
      <td>0.0</td>
      <td>783.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29989</th>
      <td>29991</td>
      <td>140000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>138262.0</td>
      <td>49675.0</td>
      <td>46121.0</td>
      <td>6000.0</td>
      <td>7000.0</td>
      <td>4228.0</td>
      <td>1505.0</td>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29990</th>
      <td>29992</td>
      <td>210000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>34</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29991</th>
      <td>29993</td>
      <td>10000.0</td>
      <td>Male</td>
      <td>High School</td>
      <td>Married</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29992</th>
      <td>29994</td>
      <td>100000.0</td>
      <td>Male</td>
      <td>Graduate</td>
      <td>Single</td>
      <td>38</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>70626.0</td>
      <td>69473.0</td>
      <td>55004.0</td>
      <td>2000.0</td>
      <td>111784.0</td>
      <td>4000.0</td>
      <td>3000.0</td>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29993</th>
      <td>29995</td>
      <td>80000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Single</td>
      <td>34</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>77519.0</td>
      <td>82607.0</td>
      <td>81158.0</td>
      <td>7000.0</td>
      <td>3500.0</td>
      <td>0.0</td>
      <td>7000.0</td>
      <td>0.0</td>
      <td>4000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29994</th>
      <td>29996</td>
      <td>220000.0</td>
      <td>Male</td>
      <td>High School</td>
      <td>Married</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>88004.0</td>
      <td>31237.0</td>
      <td>15980.0</td>
      <td>8500.0</td>
      <td>20000.0</td>
      <td>5003.0</td>
      <td>3047.0</td>
      <td>5000.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29995</th>
      <td>29997</td>
      <td>150000.0</td>
      <td>Male</td>
      <td>High School</td>
      <td>Single</td>
      <td>43</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>8979.0</td>
      <td>5190.0</td>
      <td>0.0</td>
      <td>1837.0</td>
      <td>3526.0</td>
      <td>8998.0</td>
      <td>129.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29996</th>
      <td>29998</td>
      <td>30000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Single</td>
      <td>37</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>-1</td>
      <td>...</td>
      <td>20878.0</td>
      <td>20582.0</td>
      <td>19357.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22000.0</td>
      <td>4200.0</td>
      <td>2000.0</td>
      <td>3100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29997</th>
      <td>29999</td>
      <td>80000.0</td>
      <td>Male</td>
      <td>High School</td>
      <td>Married</td>
      <td>41</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>52774.0</td>
      <td>11855.0</td>
      <td>48944.0</td>
      <td>85900.0</td>
      <td>3409.0</td>
      <td>1178.0</td>
      <td>1926.0</td>
      <td>52964.0</td>
      <td>1804.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29998</th>
      <td>30000</td>
      <td>50000.0</td>
      <td>Male</td>
      <td>College</td>
      <td>Married</td>
      <td>46</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>36535.0</td>
      <td>32428.0</td>
      <td>15313.0</td>
      <td>2078.0</td>
      <td>1800.0</td>
      <td>1430.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>29999 rows × 25 columns</p>
</div>


