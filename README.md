# Amazon Food Reviews Analysis and Modelling Using Various Machine Learning Models


#### Performed Exploratory Data Analysis, Data Cleaning, Data Visualization and Text Featurization(BOW, tfidf, Word2Vec). Build several ML models like KNN, Naive Bayes, Logistic Regression, SVM, Random Forest, GBDT, LSTM(RNNs) etc.

### Objective:
Given a text review, determine the sentiment of the review whether its positive or negative.

Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews

#### About Dataset

The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.<br>

Number of reviews: 568,454<br>
Number of users: 256,059<br>
Number of products: 74,258<br>
Timespan: Oct 1999 - Oct 2012<br>
Number of Attributes/Columns in data: 10 

Attribute Information:

1. Id
2. ProductId - unique identifier for the product
3. UserId - unqiue identifier for the user
4. ProfileName
5. HelpfulnessNumerator - number of users who found the review helpful
6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
7. Score - rating between 1 and 5
8. Time - timestamp for the review
9. Summary - brief summary of the review
10. Text - text of the review
<hr>

### Amazon Food Reviews EDA, NLP, Text Preprocessing and Visualization using TSNE
1. Performed Exploratory Data Analysis(EDA) on Amazon Fine Food Reviews Dataset plotted Word Clouds, Distplots, Histograms, etc.
2. Performed Data Cleaning & Data Preprocessing by removing unneccesary and duplicates rows and for text reviews removed html tags, punctuations, Stopwords and Stemmed the words using Porter Stemmer 
3. Plotted TSNE plots for Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec
<hr>

### KNN
1. Applied K-Nearest Neighbour on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec 
2. Used both brute & kd-tree implementation of KNN 
3. Evaluated the test data on various performance metrics like accuracy also plotted Confusion matrix 
using seaborne

**K-nn with BOW,TF-IDF,AVG-W2v,TF-IDF W2V text classifications and Optimal-k ,Brute Force algorithms:**

|          |sample size|Optimal_K|Brute Force   |              |              |            |                            
|----------|-----------|---------|--------------|--------------|--------------|------------| 
|          |           |         |Accuracy      |Precision     |Recall        |F1-score    |
|BOW       |  60k      |   15    |   81.40%     |    0.815     |    0.994     |  0.896     |
|TF-IDF    |  60k      |   10    |   81.72%     |    0.823     |    0.985     |  0.897     |
|Avg-W2V   |  60k      |   45    |   80.23%     |    0.805     |    0.996     |  0.890     |
|TF-IDF W2V|  60k      |   40    |   80.45%     |    0.805     |    0.999     |  0.892     |
    
    
**K-nn with BOW,TF-IDF,AVG-W2v,TF-IDF W2V text classifications and Optimal-k ,Brute Force algorithms:**

|          |sample size|Optimal_K|kd-tree       |              |              |            |                            
|----------|-----------|---------|--------------|--------------|--------------|------------| 
|          |           |         |Accuracy      |Precision     |Recall        |F1-score    |
|BOW       | 60k       |   45    |   79.59%     |  0.804       |  0.987       |0.866       |
|TF-IDF    | 60k       |   45    |   80.42%     |  0.805       |  0.999       |0.892       |
|Avg-W2V   | 60k       |   45    |   80.23%     |  0.805       |  0.996       |0.890       |
|TF-IDF W2V| 60k       |   40    |   80.45%     |  0.805       |  0.999       |0.892       |


**observation:**
              By comparing above table, for our data Avg-w2v with knn is working better.
    

###### Conclusions:
1.  KNN is a very slow Algorithm takes very long time to train.
2.  Best Accuracy  is achieved by Avg Word2Vec Featurization which is of 89.38%.
3.  Both kd-tree and brute algorithms of KNN gives comparatively similar results.
4.  Overall KNN was not that good for this dataset.
<hr>


### Naive Bayes
1. Applied Naive Bayes using Bernoulli NB and Multinomial NB on Different Featurization of Data viz. BOW(uni-gram), tfidf. 
2. Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne
3. Printed Top 25 Important Features for both Negative and Positive Reviews


**Naive-Bayes with different text classification:**

|          |sample size|Optimal_alpha|Test accuracy |precision     |recall        |f1-score    |                         
|----------|-----------|-------------|--------------|--------------|--------------|------------| 
|          |           |             |              |              |              |            |
|BOW       | 100k      |   0.01      |   82.90%     |    0.828     |    0.995     |  0.904     |
|TF-IDF    | 100k      |   0.05      |   83.37%     |    0.834     |    0.992     |  0.906     |

**Observation:**
              By comparing above table, for our data TFIDF with Naive-bayes is working better.
    
###### Conclusions:
1. Naive Bayes is much faster algorithm than KNN
2. The performance of bernoulli naive bayes is way much more better than multinomial naive bayes.
3. Best F1 score is acheived by BOW featurization which is 0.9342
<hr>

### Logistic Regression
1. Applied Logistic Regression on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec 
2. Used both Grid Search & Randomized Search Cross Validation
3. Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne
4. Showed How Sparsity increases as we increase lambda or decrease C when L1 Regularizer is used for each featurization<br>
5. Did pertubation test to check whether the features are multi-collinear or not

**Logistic regression with different text classifications:**
**Below values are for test data**    

|          |sample size|Optimal_lambda(1/C)|              |              |              |            |                            
|----------|-----------|-------------------|--------------|--------------|--------------|------------| 
|          |           |                   |Accuracy      |Precision     |Recall        |F1-score    |
|BOW       |  100k     |1/5=0.2            |   88.24%     |    0.900     |    0.960     |  0.929     |
|TF-IDF    |  100k     |1/1=1.0            |   87.77%     |    0.888     |    0.969     |  0.927     |
|Avg-W2V   |  100k     |1/0.00005=20,000   |   82.88%     |    0.839     |    0.971     |  0.900     |
|TF-IDF W2V|  100k     |1/0.1=10           |   83.33%     |    0.845     |    0.968     |  0.903     |
    


###### Conclusions:
1. Sparsity increases as we decrease C (increase lambda) when we use L1 Regularizer for regularization.
2. TF_IDF Featurization performs best with F1_score of 0.967 and Accuracy of 91.39.
3. Features are multi-collinear with different featurization.
4. Logistic Regression is faster algorithm.
<hr>

###  SVM
1. Applied SVM with rbf(radial basis function) kernel on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec 
2. Used both Grid Search & Randomized Search Cross Validation 
3. Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne
4. Evaluated SGDClassifier on the best resulting featurization

**Linear Kernel:**    

|          |sample size|Optimal_lambda(1/C)|              |              |              |            |       |                            
|----------|-----------|-------------------|--------------|--------------|--------------|------------|-------| 
|          |           |                   |Accuracy      |Precision     |Recall        |F1-score    |AUC    |
|BOW       |  100k     |1/5=0.2            |   88.24%     |    0.900     |    0.960     |  0.929     |75.88  |
|TF-IDF    |  100k     |1/1=1.0            |   87.77%     |    0.888     |    0.969     |  0.927     |75.98  |
|Avg-W2V   |  100k     |1/0.00005=20,000   |   82.88%     |    0.839     |    0.971     |  0.900     |64.90  |
|TF-IDF W2V|  100k     |1/0.1=10           |   83.33%     |    0.845     |    0.968     |  0.903     |60.98  |
    
 **RBF Kernel:**    

|          |sample size|Optimal_lambda(1/C)|Gamma|              |              |              |            |        |                            
|----------|-----------|-------------------|-----|--------------|--------------|--------------|------------|--------|
|          |           |                   |     |Accuracy      |Precision     |Recall        |F1-score    |AUC     |
|BOW       |  100k     |1                  |1    |   85.90%     |    0.867     |    0.975     |  0.918     |67.67   |
|TF-IDF    |  100k     |1                  |1    |   86.08%     |    0.868     |    0.975     |  0.919     |68.11   |       
|Avg-W2V   |  100k     |1                  |1    |   80.58%     |    0.806     |    1.000     |  0.892     |50.05   |
|TF-IDF W2V|  100k     |1                  |1    |   80.56%     |    0.806     |    1.000     |  0.892     |50.01   |

###### Conclusions:
1. BOW Featurization with linear kernel with grid search gave the best results with F1-score of 0.9201.
2. Using SGDClasiifier takes very less time to train.
<hr>

###  Decision Trees
1. Applied Decision Trees on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec 
2. Used both Grid Search with random 30 points for getting the best max_depth 
3. Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne
4. Plotted feature importance recieved from the decision tree classifier

**Decision trees with different text classifications:**
**Below values are for test data**    

|          |sample size|Maximum depth      |              |              |              |            |                            
|----------|-----------|-------------------|--------------|--------------|--------------|------------| 
|          |           |                   |Accuracy      |Precision     |Recall        |F1-score    |
|BOW       |  100k     |15                 |   81.50%     |    0.830     |    0.973     |  0.896     |
|TF-IDF    |  100k     |10                 |   81.82%     |    0.827     |    0.978     |  0.896     |
|Avg-W2V   |  100k     |5                  |   80.96%     |    0.817     |    0.983     |  0.892     |
|TF-IDF W2V|  100k     |5                  |   80.63%     |    0.809     |    0.992     |  0.892     |
    

###### Conclusions:
1. BOW Featurization(max_depth=8) gave the best results with accuracy of 85.8% and F1-score of 0.858.
2. Decision Trees on BOW and tfidf would have taken forever if had taken all the dimensions as it had huge dimension and hence tried with max 8 as max_depth
<hr>

### Ensembles(RF&GBDT)
1. Applied Random Forest on Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec 
2. Used both Grid Search with random 30 points for getting the best max_depth, learning rate and n_estimators. 
3. Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne
4. Plotted world cloud of feature importance recieved from the RF and GBDT classifier

**Random Forest Classifier:**    

|          |sample size|no. of estimators|max-depth|        |              |              |            |       |                            
|----------|-----------|-----------------|---------|--------|--------------|--------------|------------|-------| 
|          |           |                 |         |Accuracy|Precision     |Recall        |F1-score    |AUC    |
|BOW       |  100k     |5                |25       |80.89%  |    0.809     |    0.998     |  0.994     |51.22  |
|TF-IDF    |  100k     |3                |25       |80.90%  |    0.811     |    0.995     |  0.893     |51.73  |
|Avg-W2V   |  100k     |3                |25       |80.01%  |    0.849     |    0.915     |  0.881     |61.48  |
|TF-IDF W2V|  100k     |3                |25       |79.19%  |    0.843     |    0.911     |  0.876     |60.47  |


###### Conclusions:
1. TFIDF Featurization in Random Forest (BASE-LEARNERS=10) with random search gave the best results with F1-score of 0.857.
2. TFIDF Featurization in GBDT (BASE-LEARNERS=275, DEPTH=10) gave the best results with F1-score of 0.8708.
<hr>

