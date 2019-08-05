
## Machine Learning with Spark - Lab

## Introduction

In the previous lecture, you were shown how to manipulate data with Spark DataFrames as well as create machine learning models. In this lab, you're going to practice loading data, manipulating it, and fitting it into the Spark Framework. Afterward, you're going to make use of different visualizations to see if you can get any insights from the model. This dataset is from a Taiwanese financial company, and the task is to determine which individuals are going to default on their credit card based off of characteristics such as limit balance, past payment history, age, marriage status, and sex. Let's get started!

### Objectives

* Create machine learning pipeline with pyspark
* Evaluate a model with pyspark
* Create and interpret visualizations with pyspark

To begin with create a SparkSession and read in 'credit_card_default.csv' to a PySpark DataFrame. 


```python
# import necessary libraries
from pyspark import SparkContext
from pyspark.sql import SparkSession
# initialize Spark Session


# read in csv to a spark dataframe
spark_df = None
```


```python
# __SOLUTION__ 
# import necessary libraries
from pyspark import SparkContext
from pyspark.sql import SparkSession
# initialize Spark Session
sc = SparkContext('local[*]')
spark = SparkSession(sc)

# read in csv to a spark dataframe
spark_df = spark.read.csv('./credit_card_default.csv',header='true',inferSchema='true')
```

Check the datatypes to ensure that all columns are the datatype you expect.


```python

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
# __SOLUTION__ 
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



Check to see how many null values are in the dataset. This will require using the `filter` , `isNull`, and `count` method.


```python
for col in spark_df.columns:
    # your code here
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
# __SOLUTION__ 
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


Now, determine how many categories there are in each of the categorical columns.


```python
for column , data_type in spark_df.dtypes:
   # your code here
```

    Feature  SEX  has:  [Row(SEX='Female'), Row(SEX='Male')]
    Feature  EDUCATION  has:  [Row(EDUCATION='High School'), Row(EDUCATION='0'), Row(EDUCATION='5'), Row(EDUCATION='6'), Row(EDUCATION='Other'), Row(EDUCATION='Graduate'), Row(EDUCATION='College')]
    Feature  MARRIAGE  has:  [Row(MARRIAGE='0'), Row(MARRIAGE='Other'), Row(MARRIAGE='Married'), Row(MARRIAGE='Single')]



```python
# __SOLUTION__ 
for column , data_type in spark_df.dtypes:
    if data_type == 'string':
        print('Feature ',column,' has: ', spark_df.select(column).distinct().collect())
```

    Feature  SEX  has:  [Row(SEX='Female'), Row(SEX='Male')]
    Feature  EDUCATION  has:  [Row(EDUCATION='High School'), Row(EDUCATION='0'), Row(EDUCATION='5'), Row(EDUCATION='6'), Row(EDUCATION='Other'), Row(EDUCATION='Graduate'), Row(EDUCATION='College')]
    Feature  MARRIAGE  has:  [Row(MARRIAGE='0'), Row(MARRIAGE='Other'), Row(MARRIAGE='Married'), Row(MARRIAGE='Single')]


Interesting... it looks like we have some extraneous values in each of our categories. Let's look at some visualizations of each of these to determine just how many of them there are. Create bar plots of the variables EDUCATION and MARRIAGE to see how many of the undefined values there are. After doing so, come up with a strategy for accounting for the extra value.


```python
import seaborn as sns
import matplotlib.pyplot as plt



## plotting the categories for education

```


    <Figure size 640x480 with 1 Axes>



```python
# __SOLUTION__ 
import seaborn as sns
import matplotlib.pyplot as plt

def bar_plot_values(idx,group):
    return [x[idx] for x in group]

## plotting the categories for education
education_cats = spark_df.groupBy('EDUCATION').count().collect()
sns.barplot(x=bar_plot_values(0,education_cats),y=bar_plot_values(1,education_cats))
plt.show()
```


    <Figure size 640x480 with 1 Axes>



```python
## plotting the categories for marriage

```


![png](index_files/index_16_0.png)



```python
# __SOLUTION__ 
## plotting the categories for marriage
marriage_cats =  spark_df.groupby('MARRIAGE').count().collect()
sns.barplot(x=bar_plot_values(0, marriage_cats), y=bar_plot_values(1, marriage_cats))
plt.show()
```


![png](index_files/index_17_0.png)


It looks like there are barely any of the categories of 0 and 5 categories. We can go ahead and throw them into the "Other" category since it's already operating as a catchall here. Similarly, the category "0" looks small, so let's throw it in with the "Other" values. You can do this by using a function called `when` from pyspark in conjunction with `withColumn` and `otherwise` 


```python
from pyspark.sql.functions import when

## changing the values in the education column

## changing the values in the marriage column

spark_df_done = None
```


```python
# __SOLUTION__ 
from pyspark.sql.functions import when

## changing the values in the education column
spark_df_2 = spark_df.withColumn("EDUCATION",
                    when(spark_df.EDUCATION == '0','Other')\
                    .when(spark_df.EDUCATION == '5','Other')\
                    .when(spark_df.EDUCATION == '6','Other')\
                    .otherwise(spark_df['EDUCATION']))

## chaning the values in the marriage column
spark_df_done = spark_df_2.withColumn("MARRIAGE",
                                   when(spark_df.MARRIAGE == '0','Other')\
                                   .otherwise(spark_df['MARRIAGE']))
```


```python
spark_df_done.head()
```




    Row(ID=2, LIMIT_BAL=120000.0, SEX='Female', EDUCATION='College', MARRIAGE='Single', AGE=26, PAY_0=-1, PAY_2=2, PAY_3=0, PAY_4=0, PAY_5=0, PAY_6=2, BILL_AMT1=2682.0, BILL_AMT2=1725.0, BILL_AMT3=2682.0, BILL_AMT4=3272.0, BILL_AMT5=3455.0, BILL_AMT6=3261.0, PAY_AMT1=0.0, PAY_AMT2=1000.0, PAY_AMT3=1000.0, PAY_AMT4=1000.0, PAY_AMT5=0.0, PAY_AMT6=2000.0, default=1)




```python
# __SOLUTION__ 
spark_df_done.head()
```




    Row(ID=2, LIMIT_BAL=120000.0, SEX='Female', EDUCATION='College', MARRIAGE='Single', AGE=26, PAY_0=-1, PAY_2=2, PAY_3=0, PAY_4=0, PAY_5=0, PAY_6=2, BILL_AMT1=2682.0, BILL_AMT2=1725.0, BILL_AMT3=2682.0, BILL_AMT4=3272.0, BILL_AMT5=3455.0, BILL_AMT6=3261.0, PAY_AMT1=0.0, PAY_AMT2=1000.0, PAY_AMT3=1000.0, PAY_AMT4=1000.0, PAY_AMT5=0.0, PAY_AMT6=2000.0, default=1)



Now let's take a look at all the values contained in the categorical columns of the DataFrame.


```python
for column , data_type in spark_df_done.dtypes:
    # your code here
```

    Feature  SEX  has:  [Row(SEX='Female'), Row(SEX='Male')]
    Feature  EDUCATION  has:  [Row(EDUCATION='High School'), Row(EDUCATION='Other'), Row(EDUCATION='Graduate'), Row(EDUCATION='College')]
    Feature  MARRIAGE  has:  [Row(MARRIAGE='Other'), Row(MARRIAGE='Married'), Row(MARRIAGE='Single')]



```python
# __SOLUTION__ 
for column , data_type in spark_df_done.dtypes:
    if data_type == 'string':
        print('Feature ',column,' has: ', spark_df_done.select(column).distinct().collect())
```

    Feature  SEX  has:  [Row(SEX='Female'), Row(SEX='Male')]
    Feature  EDUCATION  has:  [Row(EDUCATION='High School'), Row(EDUCATION='Other'), Row(EDUCATION='Graduate'), Row(EDUCATION='College')]
    Feature  MARRIAGE  has:  [Row(MARRIAGE='Other'), Row(MARRIAGE='Married'), Row(MARRIAGE='Single')]


Much better. Now, let's do a little more investigation into our target variable before diving into the machine learning aspect of this project.

##  EDA

Let's first look at the overall distribution of class balance of the default and not default label to determine if there is a need for each one of the different things here. Create a barplot to compare the number of defaults vs. non-defaults. This will require using groupBy as well as an aggregation method.


```python

```




    [Text(0,0,'No Default (0)'), Text(0,0,'Default (1)')]




![png](index_files/index_28_1.png)



```python
# __SOLUTION__ 
number_of_defaults = spark_df_done.groupBy('default').count().collect()
default = [x[0] for x in number_of_defaults]
num_defaults = [x[1] for x in number_of_defaults]
ax = sns.barplot(default,num_defaults)
ax.set_ylabel('Number of Defaults')
ax.set_xticklabels(['No Default (0)','Default (1)'])

```




    [Text(0,0,'No Default (0)'), Text(0,0,'Default (1)')]




![png](index_files/index_29_1.png)


Let's also visualize the difference in default rate between males and females in this dataset.


```python
# perform a groupby for default and sex

```




    [Row(default=1, SEX='Female', count=3762),
     Row(default=0, SEX='Male', count=9015),
     Row(default=1, SEX='Male', count=2873),
     Row(default=0, SEX='Female', count=14349)]




```python
# __SOLUTION__ 
# perform a groupby for default and sex
results = spark_df_done.groupBy(['default','SEX']).count().collect()
results
```




    [Row(default=1, SEX='Female', count=3762),
     Row(default=0, SEX='Male', count=9015),
     Row(default=1, SEX='Male', count=2873),
     Row(default=0, SEX='Female', count=14349)]




```python
# make barplot for female and male default v no default rate
```




    [Text(0,0,'No Default (0)'), Text(0,0,'Default (1)')]




![png](index_files/index_33_1.png)



```python
# __SOLUTION__ 
female =  [results[0],results[-1]]
male = [results[1],results[2]]
```


```python
# __SOLUTION__ 
f, axes = plt.subplots(1,2)
f.set_figwidth(10)
sns.barplot(x= bar_plot_values(0,female),y=bar_plot_values(2,female),ax=axes[0])
sns.barplot(x= bar_plot_values(0,male),y=bar_plot_values(2,male),ax=axes[1])
axes[0].set_title('Female Default Rate')
axes[1].set_title('Male Default Rate')
axes[0].set_ylabel('Number of Defaults')
axes[0].set_xticklabels(['No Default (0)','Default (1)'])
axes[1].set_xticklabels(['No Default (0)','Default (1)'])
```




    [Text(0,0,'No Default (0)'), Text(0,0,'Default (1)')]




![png](index_files/index_35_1.png)


It looks like males have an ever so slightly higher default rate than females.

## Onto the Machine Learning!

Now, it's time to fit the data to the pyspark machine learning model pipeline.  You will need:

* 3 StringIndexers (for each categorical feature)
* A OneHotEncoderEstimator (to encode the newly indexed strings into categorical variables)
* A VectorAssembler (to combine all features into one SparseVector)

All of these initialized estimators should be stored in a list.


```python
# importing the necessary modules


# creating the string indexers


# features to be included in the model 

# adding the categorical features

# putting all of the features into a single vector


```

    [StringIndexer_47acb457ea747325b362, StringIndexer_476d9333661023960df9, StringIndexer_4477bba86fbbf3c44e9b, OneHotEncoderEstimator_41e99280e73030b62fe5, VectorAssembler_45a59eab105a2278079b]



```python
# __SOLUTION__ 
# importing the necessary modules
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, StringIndexerModel
stages = []
indexers = []

# creating the string indexers
for col in ['EDUCATION','SEX','MARRIAGE']:
    indexers.append(StringIndexer(inputCol =col,outputCol=col+'_',handleInvalid='keep'))
    
input_columns = [indexer.getOutputCol() for indexer in indexers]

one_hot_encoder = OneHotEncoderEstimator(inputCols=input_columns,outputCols=[col + 'ohe' for col in input_columns],dropLast=True)


# features to be included in the model 
features = ['LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3',
            'PAY_4','PAY_5','PAY_6', 'BILL_AMT1','BILL_AMT2',
            'BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']

# adding the categorical features
features.extend(one_hot_encoder.getOutputCols())

# putting all of the features into a single vector
vector_assember = VectorAssembler(inputCols= features , outputCol='features')

stages.extend(indexers)
stages.extend([one_hot_encoder,vector_assember])
print(stages)
```

    [StringIndexer_47acb457ea747325b362, StringIndexer_476d9333661023960df9, StringIndexer_4477bba86fbbf3c44e9b, OneHotEncoderEstimator_41e99280e73030b62fe5, VectorAssembler_45a59eab105a2278079b]


Alright! Now let's see if that worked. Let's investigate how it transforms your dataset. Put all of the stages in a Pipeline and fit it to your data. Look at the features column. Did you obtain the number of features you expected?


```python
from pyspark.ml.pipeline import Pipeline


# 17 numerical features and 6 categorical ones (the argument dropLast = True makes us have Sex, 3 Edu variables and 2 marriage)
```




    Row(features=SparseVector(23, {0: 120000.0, 1: 26.0, 2: -1.0, 3: 2.0, 7: 2.0, 8: 2682.0, 9: 1725.0, 10: 2682.0, 11: 3272.0, 12: 3455.0, 13: 3261.0, 14: 1.0, 18: 1.0, 20: 1.0}))




```python
# __SOLUTION__ 
from pyspark.ml.pipeline import Pipeline
pipe = Pipeline(stages=stages)
data_transformer = pipe.fit(spark_df_done)
transformed_data = data_transformer.transform(spark_df_done)
p = transformed_data.select('features')
p.head()

# 17 numerical features and 6 categorical ones (the argument dropLast = True makes us have Sex, 3 Edu variables and 2 marriage)
```




    Row(features=SparseVector(23, {0: 120000.0, 1: 26.0, 2: -1.0, 3: 2.0, 7: 2.0, 8: 2682.0, 9: 1725.0, 10: 2682.0, 11: 3272.0, 12: 3455.0, 13: 3261.0, 14: 1.0, 18: 1.0, 20: 1.0}))



## Fitting Machine Learning Models
That looks good! Now let's go ahead and fit data to different machine learning models. To evaluate these models, you should use the `BinaryClassificationEvaluator`. Below is an import of all the classes and libraries you'll need in the remainder of this lab.


```python
from pyspark.ml.classification import GBTClassifier, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np
```


```python
# __SOLUTION__ 
from pyspark.ml.classification import GBTClassifier, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np
```

### Logistic Regression

First, we'll try with a simple Logistic Regression Model:

* instantiate a logistic regression model
* add it to the stages list
* instantiate a new Pipeline estimator (not fit) with all of the stages
* instantiate an `BinaryClassificationEvaluator`
* create parameters to gridsearch through using `ParamGridBuilder`
* Instantiate and fit a `CrossValidator` 


```python
# your code here

```


```python
# __SOLUTION__ 
lr = LogisticRegression(featuresCol='features',labelCol='default')
p = Pipeline(stages=stages + [lr])
evaluation = BinaryClassificationEvaluator(labelCol = 'default',metricName='areaUnderROC')

lr_params = ParamGridBuilder().addGrid(lr.regParam,[0.0,0.2,0.5,1.0])\
.addGrid(lr.standardization,[True,False])\
.build()

cv = CrossValidator(estimator=p, estimatorParamMaps=lr_params,evaluator=evaluation,parallelism=4)
model = cv.fit(spark_df_done)
```

Determine how well your model performed by looking at the evaluator metrics. If you tried multiple parameters, which performed best?


```python
# print out the AUC of your best model as well as the parameters of your best model

```

    0.7183252301096683 AUC
    best parameters :  {Param(parent='LogisticRegression_4e12b7861559618c2aa6', name='regParam', doc='regularization parameter (>= 0).'): 0.0, Param(parent='LogisticRegression_4e12b7861559618c2aa6', name='standardization', doc='whether to standardize the training features before fitting the model.'): True}



```python
# __SOLUTION__ 
index_best_model = np.argmax(model.avgMetrics)
print(model.avgMetrics[index_best_model],'AUC')
print('best parameters : ',lr_params[index_best_model])
```

    0.7183252301096683 AUC
    best parameters :  {Param(parent='LogisticRegression_4e12b7861559618c2aa6', name='regParam', doc='regularization parameter (>= 0).'): 0.0, Param(parent='LogisticRegression_4e12b7861559618c2aa6', name='standardization', doc='whether to standardize the training features before fitting the model.'): True}


#### Now try this again with other classifiers. Try and create a function that will allow you to easily test different models with different parameters. This function is optional, but it should allow for your code to be far more D.R.Y. The function should return the fitted cross-validated model as well as print out the performance metrics of the best performing model and the best parameters.


```python
# create function to cross validate models with different parameters


```


```python
# __SOLUTION__ 
def create_model(ml_model,
                 preprocessing_stages,
                 param_grid,
                 parallel = 4,
                 evaluation_metric = 'areaUnderROC',
                 parafeaturesCol = 'features',
                 label='default'):
    
    stage_with_ml = preprocessing_stages + [ml_model]
    pipe = Pipeline(stages=stage_with_ml)
    
    evaluation = BinaryClassificationEvaluator(labelCol = label,metricName=evaluation_metric)
    model = CrossValidator(estimator = pipe,
                        estimatorParamMaps=param_grid,
                        evaluator = evaluation,
                       parallelism = parallel).fit(spark_df_done)

    index_best_model = np.argmax(model.avgMetrics)
    print('best performing model: ', model.avgMetrics[index_best_model],'AUC')
    print('best parameters: ',param_grid[index_best_model])
    return model


```

Train a Random Forest classifier and determine the best performing model with the best parameters. This might take a while! Be smart about how you use parallelization here.


```python
# code to train Random Forest Classifier

```

    best performing model:  0.7826543113276045 AUC
    best parameters:  {Param(parent='RandomForestClassifier_4ddb8a67eee4da94132a', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 10, Param(parent='RandomForestClassifier_4ddb8a67eee4da94132a', name='numTrees', doc='Number of trees to train (>= 1).'): 200}



```python
# __SOLUTION__ 
rf = RandomForestClassifier(featuresCol='features',labelCol='default')
rf_params = ParamGridBuilder()\
.addGrid(rf.maxDepth, [5,10])\
 .addGrid(rf.numTrees, [20,50,100,200])\
 .build()

rf_model = create_model(rf,stages,rf_params)
```

    best performing model:  0.7826543113276045 AUC
    best parameters:  {Param(parent='RandomForestClassifier_4ddb8a67eee4da94132a', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 10, Param(parent='RandomForestClassifier_4ddb8a67eee4da94132a', name='numTrees', doc='Number of trees to train (>= 1).'): 200}


Now train a Gradient Boosting Classifier. **This might take a very long time depending on the number of parameters you are training**


```python
# code to train Gradient Boosting Classifier


```

    best performing model:  0.7798494380533647 AUC
    best parameters:  {Param(parent='GBTClassifier_42569733d04d7a375612', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5, Param(parent='GBTClassifier_42569733d04d7a375612', name='maxIter', doc='max number of iterations (>= 0).'): 50}



```python
# __SOLUTION__ 
gb = GBTClassifier(featuresCol='features',labelCol='default')
param_gb = ParamGridBuilder().addGrid(gb.maxDepth,[1,5]).addGrid(gb.maxIter,[20,50,100]).build()

gb_model = create_model(gb,stages, param_grid=param_gb, parallel=4)



```

    best performing model:  0.7798494380533647 AUC
    best parameters:  {Param(parent='GBTClassifier_42569733d04d7a375612', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 5, Param(parent='GBTClassifier_42569733d04d7a375612', name='maxIter', doc='max number of iterations (>= 0).'): 50}


It looks like the optimal performing model is the Random Forest Classifier Model because it has the highest AUC!

## Level Up

* Create ROC curves for each of these models
* Try the Multi-Layer Perceptron classifier algorithm. You will soon learn about what this means in the neural network section!

## Summary

If you've made it thus far, congratulations! Spark is an in-demand skill, but it is not particularly easy to master. In this lesson, you fit multiple different machine learning pipelines for a classification problem. If you want to boost your spark skills to the next level, connect to a distributed cluster using a service like AWS or Databricks and perform these Spark operations on the cloud.
