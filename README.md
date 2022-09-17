<div class="cell markdown">

# <center> Credit Risk Modeling <center>

<center>

Hieu Nguyen <center>

</div>

<div class="cell markdown">

<div class="alert alert-block alert-info">
The project focuses on the probability of loan defaults in the banking industry. It is my practice regarding a predictive probablity-related problem using some machine learning models. There are three main credit risk models on theory (exposure at default, probability of default, and loss given default). In my analysis, I will focus solely on the probability of default. <br> <br> The goal is to build/review some models that lenders can use to help make the best financial decisions on dealing with high-risk borrowers. In particularly, I will try to predict the probability that somebody will experience financial distress in the next two years. <br><br> Later, I will compare these models with non-ML models (hopefully I have the time and effort)
</div>

</div>

<div class="cell markdown">

Content: <br>

  - Libraries
  - Data description
  - Exploratory data analysis
  - Oversampling
  - Data evaluation (correlation matrix)
  - Modeling
      - K-Means
      - Logistics regression
      - Random forest
      - Gradient boosting
  - Model performance
      - Confusion matrix
      - Accuracy
      - Precision, recall, and F-measure
      - Receiver Operating Characteristics Curve (ROC), Precision-Recall
        Curve, and AUC
  - Post analysis
      - Feature analysis
      - Model analysis and discussion

</div>

<div class="cell markdown">

## Libraries

</div>

<div class="cell code" data-execution_count="8">

``` python
import pandas as pd
from pathlib import Path
import os
```

</div>

<div class="cell code">

``` python
# set seeds
seed = 3001
```

</div>

<div class="cell markdown">

## Data description

</div>

<div class="cell markdown">

The four used datasets were from:
<https://www.kaggle.com/competitions/GiveMeSomeCredit/data>. <br>

  - Data dictionary (xls)
  - Credit score training data (csv)
  - Credit score test data (csv)
  - Sample entry (?) (csv)

I downloaded all of these data and uploaded to my github.

</div>

<div class="cell code" data-execution_count="18">

``` python
base_dir = os.getcwd()

#data dictionary (since it is an Excel file, I had to download it and process it locally in my machine)
dict_path = "data/Data Dictionary.xls"
dic = pd.read_excel(dict_path, skiprows=1)
dic
```

<div class="output execute_result" data-execution_count="18">

``` 
                           Variable Name  \
0                       SeriousDlqin2yrs   
1   RevolvingUtilizationOfUnsecuredLines   
2                                    age   
3   NumberOfTime30-59DaysPastDueNotWorse   
4                              DebtRatio   
5                          MonthlyIncome   
6        NumberOfOpenCreditLinesAndLoans   
7                NumberOfTimes90DaysLate   
8           NumberRealEstateLoansOrLines   
9   NumberOfTime60-89DaysPastDueNotWorse   
10                    NumberOfDependents   

                                          Description        Type  
0   Person experienced 90 days past due delinquenc...         Y/N  
1   Total balance on credit cards and personal lin...  percentage  
2                            Age of borrower in years     integer  
3   Number of times borrower has been 30-59 days p...     integer  
4   Monthly debt payments, alimony,living costs di...  percentage  
5                                      Monthly income        real  
6   Number of Open loans (installment like car loa...     integer  
7   Number of times borrower has been 90 days or m...     integer  
8   Number of mortgage and real estate loans inclu...     integer  
9   Number of times borrower has been 60-89 days p...     integer  
10  Number of dependents in family excluding thems...     integer  
```

</div>

</div>

<div class="cell code" data-execution_count="15">

``` python
# import credit data directly from my github
train_url = "https://raw.githubusercontent.com/quanghieu31/credit-risk-modeling/main/data/cs-training.csv?token=GHSAT0AAAAAABVGOACGNSFTT45LDTUSXEFGYZF3QEQ"
train_data = pd.read_csv(train_url)
test_url = "https://raw.githubusercontent.com/quanghieu31/credit-risk-modeling/main/data/cs-test.csv?token=GHSAT0AAAAAABVGOACH6TI5MZFFII5BFQWEYZF3AXA"
test_data = pd.read_csv(test_url)
```

</div>

<div class="cell markdown">

We can see that our label is the variable *SeriousDlqin2yrs* or Person
experienced 90 days past due delinquency with binary values (Yes-1 and
No-0). The first thought might be that running a logistics regression
makes sense here which is true, and I will also utilize other models to
tackle this. But first, let's explore and clean the data.

</div>

<div class="cell markdown">

## Exploratory data analysis

</div>

<div class="cell code" data-execution_count="29">

``` python
train_data.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150000 entries, 0 to 149999
    Data columns (total 12 columns):
     #   Column                                Non-Null Count   Dtype  
    ---  ------                                --------------   -----  
     0   ID                                    150000 non-null  int64  
     1   SeriousDlqin2yrs                      150000 non-null  int64  
     2   RevolvingUtilizationOfUnsecuredLines  150000 non-null  float64
     3   age                                   150000 non-null  int64  
     4   NumberOfTime30-59DaysPastDueNotWorse  150000 non-null  int64  
     5   DebtRatio                             150000 non-null  float64
     6   MonthlyIncome                         120269 non-null  float64
     7   NumberOfOpenCreditLinesAndLoans       150000 non-null  int64  
     8   NumberOfTimes90DaysLate               150000 non-null  int64  
     9   NumberRealEstateLoansOrLines          150000 non-null  int64  
     10  NumberOfTime60-89DaysPastDueNotWorse  150000 non-null  int64  
     11  NumberOfDependents                    146076 non-null  float64
    dtypes: float64(4), int64(8)
    memory usage: 13.7 MB

</div>

</div>

<div class="cell markdown">

Observation: *NumberOfDependents* is in float64 type (shouldn't it be
integer?). The first column *Unnamed: 0*'s name is not very pleasing to
my eyes, so I will change it to *ID* (ID of recorded people who were
having loans). There are no other columns with categorical values
(except for our label column), they all have continuous values.

</div>

<div class="cell code" data-execution_count="25">

``` python
train_data = train_data.rename(columns={'Unnamed: 0': 'ID'})
train_data.describe()
```

<div class="output execute_result" data-execution_count="25">

``` 
                  ID  SeriousDlqin2yrs  RevolvingUtilizationOfUnsecuredLines  \
count  150000.000000     150000.000000                         150000.000000   
mean    75000.500000          0.066840                              6.048438   
std     43301.414527          0.249746                            249.755371   
min         1.000000          0.000000                              0.000000   
25%     37500.750000          0.000000                              0.029867   
50%     75000.500000          0.000000                              0.154181   
75%    112500.250000          0.000000                              0.559046   
max    150000.000000          1.000000                          50708.000000   

                 age  NumberOfTime30-59DaysPastDueNotWorse      DebtRatio  \
count  150000.000000                         150000.000000  150000.000000   
mean       52.295207                              0.421033     353.005076   
std        14.771866                              4.192781    2037.818523   
min         0.000000                              0.000000       0.000000   
25%        41.000000                              0.000000       0.175074   
50%        52.000000                              0.000000       0.366508   
75%        63.000000                              0.000000       0.868254   
max       109.000000                             98.000000  329664.000000   

       MonthlyIncome  NumberOfOpenCreditLinesAndLoans  \
count   1.202690e+05                    150000.000000   
mean    6.670221e+03                         8.452760   
std     1.438467e+04                         5.145951   
min     0.000000e+00                         0.000000   
25%     3.400000e+03                         5.000000   
50%     5.400000e+03                         8.000000   
75%     8.249000e+03                        11.000000   
max     3.008750e+06                        58.000000   

       NumberOfTimes90DaysLate  NumberRealEstateLoansOrLines  \
count            150000.000000                 150000.000000   
mean                  0.265973                      1.018240   
std                   4.169304                      1.129771   
min                   0.000000                      0.000000   
25%                   0.000000                      0.000000   
50%                   0.000000                      1.000000   
75%                   0.000000                      2.000000   
max                  98.000000                     54.000000   

       NumberOfTime60-89DaysPastDueNotWorse  NumberOfDependents  
count                         150000.000000       146076.000000  
mean                               0.240387            0.757222  
std                                4.155179            1.115086  
min                                0.000000            0.000000  
25%                                0.000000            0.000000  
50%                                0.000000            0.000000  
75%                                0.000000            1.000000  
max                               98.000000           20.000000  
```

</div>

</div>

<div class="cell markdown">

Observation: ... (to be continued)

</div>

<div class="cell code">

``` python
```

</div>
