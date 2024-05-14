Anomaly Detection
In this exercise, you will implement the anomaly detection algorithm and apply it to detect failing servers on a network.



2 - Anomaly detection¶

2.1 Problem Statement
In this exercise, you will implement an anomaly detection algorithm to detect anomalous behavior in server computers.

The dataset contains two features -

throughput (mb/s) and
latency (ms) of response of each server.
While your servers were operating, you collected  𝑚=307  examples of how they were behaving, and thus have an unlabeled dataset  {𝑥(1),…,𝑥(𝑚)} .

You suspect that the vast majority of these examples are “normal” (non-anomalous) examples of the servers operating normally, but there might also be some examples of servers acting anomalously within this dataset.
You will use a Gaussian model to detect anomalous examples in your dataset.

You will first start on a 2D dataset that will allow you to visualize what the algorithm is doing.
On that dataset you will fit a Gaussian distribution and then find values that have very low probability and hence can be considered anomalies.
After that, you will apply the anomaly detection algorithm to a larger dataset with many dimensions.

2.2 Dataset
You will start by loading the dataset for this task.

The load_data() function shown below loads the data into the variables X_train, X_val and y_val
You will use X_train to fit a Gaussian distribution
You will use X_val and y_val as a cross validation set to select a threshold and determine anomalous vs normal examples



View the variables
Let's get more familiar with your dataset.

A good place to start is to just print out each variable and see what it contains.
The code below prints the first five elements of each of the variables


Check the dimensions of your variables
Another useful way to get familiar with your data is to view its dimensions.

The code below prints the shape of X_train, X_val and y_val



Visualize your data
Before starting on any task, it is often useful to understand the data by visualizing it.

For this dataset, you can use a scatter plot to visualize the data (X_train), since it has only two properties to plot (throughput and latency)

Your plot should look similar to the one below





2.3 Gaussian distribution
To perform anomaly detection, you will first need to fit a model to the data’s distribution.

Given a training set {𝑥(1),...,𝑥(𝑚)} you want to estimate the Gaussian distribution for each of the features 𝑥𝑖.

Recall that the Gaussian distribution is given by

𝑝(𝑥;𝜇,𝜎2)=1/2𝜋𝜎2⎯⎯⎯⎯⎯⎯⎯⎯√exp−(𝑥−𝜇)2/2𝜎2
where 𝜇 is the mean and 𝜎2 controls the variance.

For each feature 𝑖=1…𝑛, you need to find parameters 𝜇𝑖 and 𝜎2𝑖 that fit the data in the 𝑖-th dimension {𝑥(1)𝑖,...,𝑥(𝑚)𝑖} (the 𝑖-th dimension of each example).

2.2.1 Estimating parameters for a Gaussian
Implementation:

Your task is to complete the code in estimate_gaussian below.


Exercise 1
Please complete the estimate_gaussian function below to calculate mu (mean for each feature in X)and var (variance for each feature in X).

You can estimate the parameters, ( 𝜇𝑖 ,  𝜎2𝑖 ), of the  𝑖 -th feature by using the following equations. To estimate the mean, you will use:

𝜇𝑖=1𝑚∑𝑗=1𝑚𝑥(𝑗)𝑖
 
and for the variance you will use:
𝜎2𝑖=1𝑚∑𝑗=1𝑚(𝑥(𝑗)𝑖−𝜇𝑖)2
 
If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.



Now that you have completed the code in estimate_gaussian, we will visualize the contours of the fitted Gaussian distribution.


From your plot you can see that most of the examples are in the region with the highest probability, while the anomalous examples are in the regions with lower probabilities.


Selecting the threshold  𝜖 
Now that you have estimated the Gaussian parameters, you can investigate which examples have a very high probability given this distribution and which examples have a very low probability.

The low probability examples are more likely to be the anomalies in our dataset.
One way to determine which examples are anomalies is to select a threshold based on a cross validation set.
In this section, you will complete the code in select_threshold to select the threshold  𝜀  using the  𝐹1  score on a cross validation set.

For this, we will use a cross validation set  {(𝑥(1)cv,𝑦(1)cv),…,(𝑥(𝑚cv)cv,𝑦(𝑚cv)cv)} , where the label  𝑦=1  corresponds to an anomalous example, and  𝑦=0  corresponds to a normal example.
For each cross validation example, we will compute  𝑝(𝑥(𝑖)cv) . The vector of all of these probabilities  𝑝(𝑥(1)cv),…,𝑝(𝑥(𝑚cv)cv)  is passed to select_threshold in the vector p_val.
The corresponding labels  𝑦(1)cv,…,𝑦(𝑚cv)cv  is passed to the same function in the vector y_val.

Exercise 2
Please complete the select_threshold function below to find the best threshold to use for selecting outliers based on the results from a validation set (p_val) and the ground truth (y_val).

In the provided code select_threshold, there is already a loop that will try many different values of 𝜀 and select the best 𝜀 based on the 𝐹1 score.

You need implement code to calculate the F1 score from choosing epsilon as the threshold and place the value in F1.

Recall that if an example 𝑥 has a low probability 𝑝(𝑥)<𝜀, then it is classified as an anomaly.

Then, you can compute precision and recall by:
𝑝𝑟𝑒𝑐𝑟𝑒𝑐==𝑡𝑝𝑡𝑝+𝑓𝑝𝑡𝑝𝑡𝑝+𝑓𝑛,
where

𝑡𝑝 is the number of true positives: the ground truth label says it’s an anomaly and our algorithm correctly classified it as an anomaly.
𝑓𝑝 is the number of false positives: the ground truth label says it’s not an anomaly, but our algorithm incorrectly classified it as an anomaly.
𝑓𝑛 is the number of false negatives: the ground truth label says it’s an anomaly, but our algorithm incorrectly classified it as not being anomalous.
The 𝐹1 score is computed using precision (𝑝𝑟𝑒𝑐) and recall (𝑟𝑒𝑐) as follows:
𝐹1=2⋅𝑝𝑟𝑒𝑐⋅𝑟𝑒𝑐𝑝𝑟𝑒𝑐+𝑟𝑒𝑐
Implementation Note: In order to compute 𝑡𝑝, 𝑓𝑝 and 𝑓𝑛, you may be able to use a vectorized implementation rather than loop over all the examples.


High dimensional dataset
Now, we will run the anomaly detection algorithm that you implemented on a more realistic and much harder dataset.

In this dataset, each example is described by 11 features, capturing many more properties of your compute servers.

Let's start by loading the dataset.

The load_data() function shown below loads the data into variables X_train_high, X_val_high and y_val_high
_high is meant to distinguish these variables from the ones used in the previous part
We will use X_train_high to fit Gaussian distribution
We will use X_val_high and y_val_high as a cross validation set to select a threshold and determine anomalous vs normal examples




Anomaly detection
Now, let's run the anomaly detection algorithm on this new dataset.

The code below will use your code to

Estimate the Gaussian parameters (𝜇𝑖 and 𝜎2𝑖)
Evaluate the probabilities for both the training data X_train_high from which you estimated the Gaussian parameters, as well as for the the cross-validation set X_val_high.
Finally, it will use select_threshold to find the best threshold 𝜀.
