
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
import seaborn as sns

def bar_plot_values(idx,group):
    return [x[idx] for x in group]


education_cats = spark_df.groupBy('EDUCATION').count().collect()
sns.barplot(x=bar_plot_values(0,education_cats),y=bar_plot_values(1,education_cats))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a12a0d5c0>




```python
marriage_cats =  spark_df.groupby('MARRIAGE').count().collect()
sns.barplot(x=bar_plot_values(0, marriage_cats), y=bar_plot_values(1, marriage_cats))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1e14f630>




![png](index_files/index_11_1.png)


It looks like there are barely any of the categories of 0 and 5 categories. We can go ahead and throw them into the "Other" category since it's already operating as a catchall here. Similarly, the category "0" looks small, so let's throw it in with the "Other" values.


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
spark_df
```

Now let's take a look at the values.


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
number_of_defaults = spark_df_done.groupBy('default').count().collect()

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




![png](index_files/index_21_1.png)



```python
## a reasonable class balance, no real issues with class imbalances
```


```python
results = spark_df_done.groupBy(['SEX','default']).count().collect()
```


```python
results
```




    [Row(SEX='Male', default=1, count=2873),
     Row(SEX='Female', default=0, count=14349),
     Row(SEX='Male', default=0, count=9015),
     Row(SEX='Female', default=1, count=3762)]




```python
defaulted = results[:2]
not_defaulted = results[2:]

```


```python
bar_plot_values(1,defaulted)
```




    [1, 0]




```python
f, axes = plt.subplots(1,2)
f.set_figwidth(10)
sns.barplot(x= bar_plot_values(1,defaulted),y=bar_plot_values(2,defaulted),ax=axes[0])
sns.barplot(x= bar_plot_values(1,not_defaulted),y=bar_plot_values(2,not_defaulted),ax=axes[1])


```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1e314be0>




![png](index_files/index_27_1.png)



```python
results = spark_df.groupBy(['SEX','default']).count().collect()
```


```python
import seaborn as sns

```

## Onto the Machine Learning!


```python
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, StringIndexerModel
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

one_hot_encoder = OneHotEncoderEstimator(inputCols=input_columns,outputCols=[col + 'ohe' for col in input_columns],dropLast=True)

features = ['LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3',
            'PAY_4','PAY_5','PAY_6', 'BILL_AMT1','BILL_AMT2',
            'BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']

features.extend(one_hot_encoder.getOutputCols())

vector_assember = VectorAssembler(inputCols= features , outputCol='features')

stages.extend(indexers)
stages.extend([one_hot_encoder,vector_assember])
print(stages)
```

    [StringIndexer_40d38a1255d92550c417, StringIndexer_41478663d2855607f4d5, StringIndexer_4f83b58450346b3e4366, OneHotEncoderEstimator_44b7a30776829ea2baad, VectorAssembler_4343937965100214c801]



```python
features
```




    ['LIMIT_BAL',
     'AGE',
     'PAY_0',
     'PAY_2',
     'PAY_3',
     'PAY_4',
     'PAY_5',
     'PAY_6',
     'BILL_AMT1',
     'BILL_AMT2',
     'BILL_AMT3',
     'BILL_AMT4',
     'BILL_AMT5',
     'BILL_AMT6',
     'EDUCATION_ohe',
     'SEX_ohe',
     'MARRIAGE_ohe']



Alright! Now let's see if that worked. Let's investigate how it transforms our dataset.


```python
from pyspark.ml.pipeline import Pipeline
pipe = Pipeline(stages=stages)
```


```python
data_transformer = pipe.fit(spark_df_done)
transformed_data = data_transformer.transform(spark_df_done)
p = transformed_data.select('features')
```


```python
spark_df_done.columns
```




    ['ID',
     'LIMIT_BAL',
     'SEX',
     'EDUCATION',
     'MARRIAGE',
     'AGE',
     'PAY_0',
     'PAY_2',
     'PAY_3',
     'PAY_4',
     'PAY_5',
     'PAY_6',
     'BILL_AMT1',
     'BILL_AMT2',
     'BILL_AMT3',
     'BILL_AMT4',
     'BILL_AMT5',
     'BILL_AMT6',
     'PAY_AMT1',
     'PAY_AMT2',
     'PAY_AMT3',
     'PAY_AMT4',
     'PAY_AMT5',
     'PAY_AMT6',
     'default']




```python
ohe_bleh= pipe.getStages()[3]
```


```python
ohe_bleh.getInputCols()
```




    ['EDUCATION_', 'SEX_', 'MARRIAGE_']




```python
len(p.head()[0])
```




    23



That looks good! Now let's go ahead and fit it to an ML pipeline. Try whichever machine learning model you would like. 


```python
from pyspark.ml.classification import GBTClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```

### Logistic Regression

First, we'll try with a simple Logistic Regression Model:


```python
lr = LogisticRegression(featuresCol='features',labelCol='default')
```


```python
p = Pipeline(stages=stages + [lr])
model = p.fit(spark_df_done)
```


```python
trained = model.transform(spark_df_done)
```


```python
trained.select('prediction').take(20)
```




    [Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0),
     Row(prediction=0.0)]




```python
evaluation = BinaryClassificationEvaluator(labelCol = 'default')
```


```python
evaluation.evaluate(trained)
```




    0.7193835007502917




```python
lr_params = ParamGridBuilder().addGrid(lr.regParam,[0.0,0.2,0.5,1.0])\
.addGrid(lr.standardization,[True,False])\
.build()

# cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params,evaluator=reg_evaluator)
```


```python
print(lr_params)
```

### Making a reusable function to run models


```python
import numpy as np
import time
def create_model(ml_model,
                 preprocessing_stages,
                 param_grid,
                 parallel = 4,
                 evaluation_metric = 'areaUnderROC',
                 parafeaturesCol = 'features',
                 label='default'):
    start = time.time()
    stage_with_ml = preprocessing_stages + [ml_model]
    pipe = Pipeline(stages=stage_with_ml)
    
    evaluation = BinaryClassificationEvaluator(labelCol = label,metricName=evaluation_metric)
    cv = CrossValidator(estimator = pipe,
                        estimatorParamMaps=param_grid,
                        evaluator = evaluation,
                       parallelism = parallel).fit(spark_df_done)
    print(np.mean(cv.avgMetrics))
    end = time.time()
    print(end-start)
    return cv


```


```python
spark_df_done.dtypes
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



Create an ROC curve comparing the effectiveness of your optimal three models. Which one performed the best? Use AUC of the ROC to make your decision.


```python
cross_val_model_lr = create_model(lr,stages,param_grid=lr_params,parallel = 4)
```

    0.7047069286355365
    29.316229104995728



```python
evaluate= cross_val_model_lr.getEvaluator()
estimate = cross_val_model_lr.bestModel
```


```python
evaluate.evaluate(estimate.transform(spark_df_done))
```




    0.7193835007502917



### Gradient Boosting


```python
GBTParams??
```


```python
gb = GBTClassifier(featuresCol='features',labelCol='default')
param_gb = ParamGridBuilder().addGrid(gb.maxDepth,[1,5]).addGrid(gb.maxIter,[20,100]).addGrid(gb.stepSize,[0.1,0.01]).build()


# params = ParamGridBuilder()\
# .addGrid(random_forest.maxDepth, [5,10,15])\
# .addGrid(random_forest.numTrees, [20,50,100])\
# .build()
```


```python
cross_val_gb = create_model(gb,stages, param_grid=param_gb, parallel=4)
```

    0.7637664085212773
    452.124135017395



```python
# evaluate= cross_val_model_lr.getEvaluator()
# estimate = cross_val_model_lr.bestModel
```


```python
evaluate_gb = cross_val_gb.getEvaluator()
estimator = cross_val_gb.bestModel
evaluate_gb.evaluate(estimator.transform(spark_df_done))
```




    0.7992496265323968




```python
gbmodel = estimator.stages[-1]
```


```python
gbmodel.featureImportances.toArray()
```




    array([0.1217847 , 0.09038265, 0.12095581, 0.04057928, 0.09497802,
           0.03388622, 0.03667418, 0.04226211, 0.11547971, 0.06103341,
           0.03749708, 0.04312199, 0.04007098, 0.0405966 , 0.00921013,
           0.01554876, 0.00478494, 0.01917409, 0.01031815, 0.        ,
           0.01060254, 0.00758431, 0.00347433])




```python
gbmodel.featuresCol
```




    Param(parent='GBTClassifier_489880420cdc491b1a97', name='featuresCol', doc='features column name')




```python
len(['LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3',
            'PAY_4','PAY_5','PAY_6', 'BILL_AMT1','BILL_AMT2',
            'BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6', 'SEX','SEX2','GRAD_SCHOOL','COLLEGE','HIGH_SCHOOL','OTHER','MARRIED','NOT MARRIED','OTHER'])

```




    23



## Make an ROC curve for each of your models

## Summary

If you've made it thus far, congratulations! pyspark is not an easy to use language any way you approach it. It's quite new, and as a result the documentation can be lacking and there is not as much support online as their is for more established libraries like sci-kit learn or pandas.
