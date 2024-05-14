An Anomaly/Outlier is a data point that deviates significantly from normal/regular data. Anomaly detection problems can be classified into 3 types:

Supervised: In these problems, data contains both Anomalous and Clean data along with labels which tell us which examples are anomalous. We use classification algorithms to perform anomaly detection.
Semi-Supervised: Here, we only have access to ‘Clean’ data during the training. The model tries to capture what ‘normal’ data looks like — and labels data that looks ‘abnormal’ as outliers during prediction. Autoencoders are used widely in this category.
Un-Supervised: Here, data contains both clean and anomalous examples — but does not have labels which tell us which examples are anomalous. This is the category encountered most often .
In this article, we will discuss Un-supervised methods of performing Anomaly/Outlier Detection. I will discuss the Semi-Supervised and Supervised methods in a future article.

Unsupervised Anomaly Detection problems can be solved by 3 kinds of methods:

Business/Domain based EDA
Univariate Methods(Tukey’s Method, z-Score, etc)
Multivariate Methods(Mahalanobis Distance(using MCD), One-Class SVM, Isolation Forests, etc)
We will discuss the Mahalanobis Distance method using FastMCD — which is one of the multivariate methods in relatively more detail — as multivariate methods are less known but extremely useful. Let us take a look at each category and understand them from a practical perspective.

We should also remember that an anomalous point requires further attention —it must be analyzed from a domain perspective. So, in most cases when we say that a point is an anomaly, we mean it deserves more analysis.

Business / Domain based EDA
This is the first approach that must be tried — and it should be an ongoing process throughout the entire Anomaly Detection or ML Pipeline. The goal is to identify unusual behavior by performing domain analysis through Data Visualization. Following are some good ways to start:

Make Box Plots and Histograms to identify regimes of scarce data and extreme values. Scarce data, can also exist between 2 modes as seen in the figure.
Visualize scatterplots — specially between dependent variables(dependent or collinear variables). It is easy to spot deviations when a pattern already exists.


Some common potential anomalies that can be detected using Simple EDA.
The Scatterplot shows an interesting scenario — The 2 isolated highlighted points do not look like anomalies if only the marginal histograms are looked at. Only when we plot the 2 variables in a scatterplot, we see that the COMBINATION of values taken by Var1, Var2 is unusual — not the individual values of Var1 or Var2. This is an example of a case where plotting univariate histograms would not work in identifying the anomalies. This kind of an anomaly is a Multivariate Anomaly and is discussed later on in the article.

Univariate Methods
Univariate methods are easy to implement, and fast to execute. Their results are also easy to explain to business stakeholders. The idea is to look at the variables one at a time and identify regions where either:

Scarce data exists (and/or)
Data takes extreme values
We will briefly discuss Tukey’s Method which treats extreme values in data as outliers/anomalies:

In Tukey’s method, we define a lower limit and upper limit. Data within these limits, is considered ‘clean’. The lower and upper limits are determined in a robust way. That means, that the upper and lower limits do not get influenced by the presence of the outliers. This is a distinction from some other methods like the z-score method, where the lower and upper limits are influenced by the outliers. In general, it is better to use robust methods.

The Upper and Lower limits are defined as follows:

Lower Limit = 25th Percentile — k*IQR

Upper Limit = 75th Percentile + k*IQR

Where, k is generally 1.5 but must be adjusted if required. IQR is the Inter-Quartile Range(IQR = 75th Percentile — 25th Percentile of data) of the variable. Values in data below the lower limit or above the upper limit are called outliers.


The outliers found through tukey’s method, can be visualized using a typical boxplot. The plot gives an example. Extreme values are considered anomalous in this method.
The following function replaces the outliers identified by tukey’s method(values beyond the limits defined above) by NaN:

def tukey(x, k = 1.5):
    x = np.array(x).copy().astype(float)
    first_quartile = np.quantile(x, .25)
    third_quartile = np.quantile(x, .75)
    
    # Define IQR
    iqr = third_quartile - first_quartile
    
    ### Define the allowed limits for 'Normal Data'
    lower_allowed_limit = first_quartile - (k * iqr)
    upper_allowed_limit = third_quartile + (k * iqr)
    
    #set values below the lower limit/above the upper limit as nan
    x[(x<lower_allowed_limit) | (x>upper_allowed_limit)] = np.nan
    return x
Some other Univariate Methods are z-score method and Median Absolute Deviation method-each with its own pros and cons.

Multivariate Methods
Let us understand what is meant by multivariate outliers. Consider a car — and imagine 2 features that we measure:

odo: it shows the odometer reading on the car and measures the speed of the car in mph.
rpm: It measures the number of rotations made by the car wheels per minute.
Let us say that the odo takes values in the range of 0–50mph and rpm takes values in the range of 0–650 rpm. We expect the readings of the 2 features to be correlated i.e. a large rpm will cause the odometer to record higher speeds.

Now, imagine we record a value of 0 on the rpm sensor. We conclude that the car is not moving. Similarly, say while driving, if the odo reads 25mph, we conclude that the car is moving. However, none of these values is an outlier — because they represent perfectly normal modes of operation.

However, let us imagine we note that the odo reads 25 but at the same time, the rpm reads 0. This looks unreasonable. The odo value of 25 in itself is not unreasonable; and rpm of 0 is also not unreasonable(as discussed above)but for them to take those values at the same time is unreasonable. This is an example of a multivariate outlier. Multivariate outliers are observations for which the combination of values taken by the features is improbable. The important thing here is to consider the values of all the features at the same time — as opposed to taking one feature at a time which we did when we discussed univariate methods. Multivariate outliers need specialized methods — and cannot be, in general detected by univariate methods unless the features take extreme values individually. They are also very difficult to detect — the above example had 2 variables(odo and rpm) and that is why we could spot the outlier easily — however, the problem becomes impossible to do manually when we have hundreds of variables. Using Multivariate methods can make the process easy for us even when dealing with hundreds of variables.

Let us now look at an algorithm for detecting Multivariate Anomalies/Outliers. As discussed in the beginning, we will discuss the unsupervised case — where the data is known to be contaminated by outliers but the exact outlying observations are not known.

Mahalanobis Distance Method using FastMCD Algorithm:
We will implement this method using sklearn. Let us first discuss the mechanics of the method.
The Mahalanobis distance can be effectively thought of a way to measure the distance between a point and a distribution. When using it to detect anomalies, we consider the ‘Clean’ data to be the distribution. If the Mahalanobis distance of a point from the ‘Clean’ Data is high, we consider it to be an anomaly. This method assumes the clean data to be Multivariate Normal but in practice, it can be used even for a variety of other cases.

The Mahalanobis distance is closely related to the Multivariate Normal Distribution. Here are some of its characteristics:

If the data follows a multivariate normal distribution, all points having the same probability have the same mahalanobis distance from the mean value of the distribution.
Higher the Mahalanobis distance of a point from the mean of the normal distribution, lower is the probability of that point. Now, it makes sense from a statistical perspective — as to why points having large Mahalanobis distance are potential anomalies — because they correspond to low probabilities.
A normal distribution is uniquely determined by its mean and covariance matrix which needs to be estimated from data. However, we use the FastMCD Algorithm to determine the mean and covariance matrix of the normal distribution. We use the FastMCD Algorithm because we want a robust estimate of the mean and covariance. If we were to use the direct formulae of mean and covariance, then the outliers would also contribute to calculating the mean and covariance-which is not what we want. Following is a good resource to learn about FastMCD: https://onlinelibrary.wiley.com/doi/epdf/10.1002/wics.1421

Fortunately, Scikit-learn has a very convenient way of using this method. Following are the steps:
1. Fit sklearn.covariance.EllipticEnvelope() to data: This calculates the robust mean and covariance of the data using the FastMCD Algorithm. We need to pass it the value of contamination which is an estimate of what fraction of data we expect to be anomalous.
2. Predict: Predict the outlier/anomaly status of each data point. Points labelled -1 by the algorithm are anomalies and +1 are not anomalies.

Here is the example from before:


We want to detect the 2 points on the top left as outliers using Mahalanobis distance method.
### Create the Data
d1 = np.random.multivariate_normal(mean = np.array([-.5, 0]),
                               cov = np.array([[1, 0], [0, 1]]), size = 100)
d2 = np.random.multivariate_normal(mean = np.array([15, 10]),
                               cov = np.array([[1, 0.3], [.3, 1]]), size = 100)
outliers = np.array([[0, 10],[0, 9.5]])
d = pd.DataFrame(np.concatenate([d1, d2, outliers], axis = 0), columns = ['Var 1', 'Var 2'])
### The outliers added above are what we want to detect ####
############# Use Mahalanobis distance method to detect them ####
# Define the Elliptic Envelope
el = covariance.EllipticEnvelope(store_precision=True, assume_centered=False, support_fraction=None, 
                                    contamination=0.0075, random_state=0)
# Fit the data - this is where FastMCD is used by sklearn
el.fit(d)
# Create column that shows anomaly status
d['Anomaly or Not'] = el.predict(d)
# Create scatterplot and color the anomalies differently
plt.figure(figsize = (9, 4))
ax = plt.scatter(d['Var 1'], d['Var 2'], c = d['Anomaly or Not'], cmap = 'coolwarm')
plt.xlabel('Var 1')
plt.ylabel('Var 2')
plt.colorbar(label = '-1: Anomaly; +1: Not Anomaly')
plt.grid()
Output for contamination = 0.075:


As we can see, the Algorithm labels the 2 examples on the top left as anomalies. It is important to experiment with the contamination value to find the correct one.
As we can see, the method works — it detects multivariate anomalies. It can be used for data having hundreds of dimensions. Setting the contamination is very important. To see that, let us check the results of the algorithm as we set different values for the contamination.





For small values of Contamination, the algorithm is conservative and detects few anomalies. With increasing contamination, it labels more “outer” points as anomalies.
As we can see, setting the contamination right is very important. Contamination should be set to our best estimate of the fraction of data points that are anomalous.

What if we dont have an estimate of Contamination?
Then, we directly calculate the Mahalanobis distance of each point from the robust mean and set a cutoff for it based on the distribution of Mahalanobis distances in the data. We do the following:

Fit the sklearn.covariance.EllipticEnvelope() to data.
Calculate the Mahalanobis distance of each data point from the robust mean by using the mahalanobis() method.
Visualize the distribution of Mahalanobis distances present in data. Identify a threshold above which a point will be called an outlier — by visualizing the distribution of the Distances OR Use a univariate Anomaly detection algorithm on the distances to find out which distances are anomalous.
Also, please note that the value of contamination does not matter in this method — so we set to any arbitrary value.

# Create Data - with Anomaly - as before.
d1 = np.random.multivariate_normal(mean = np.array([-.5, 0]),
                               cov = np.array([[1, 0], [0, 1]]), size = 100)
d2 = np.random.multivariate_normal(mean = np.array([15, 10]),
                               cov = np.array([[1, 0.3], [.3, 1]]), size = 100)
outliers = np.array([[0, 10],[0, 9.5]])
d = pd.DataFrame(np.concatenate([d1, d2, outliers], axis = 0), columns = ['Var 1', 'Var 2'])
###### Fit Elliptic Envelope ##############
contamination = .4 # We can set any value here as we will now use our own threshold
el = covariance.EllipticEnvelope(store_precision=True, assume_centered=False, support_fraction=None, 
                                    contamination=contamination, random_state=0)
# Fit the data
el.fit(d)
############# New Part ################
# Create column that measures Mahalanobis distance
d['Mahalanobis Distance'] = el.mahalanobis(d)
# Create scatterplot and color the anomalies differently
plt.figure(figsize = (12, 6))
ax = plt.scatter(d['Var 1'], d['Var 2'], c = d['Mahalanobis Distance'], cmap = 'coolwarm')
#plt.title('Contamination = Does not matter for this method', weight = 'bold')
#ax = sns.scatterplot(d['Var 1'], d['Var 2'], c = d['Anomaly or Not'])
plt.xlabel('Var 1')
plt.ylabel('Var 2')
plt.colorbar(label = 'Mahalanobis Distance')
plt.grid()


As we can see, Mahalanobis distances for the 2 anomalies are more than for the remaining data.
Let us now identify a threshold for the Mahalanobis distance. One way to do this is to apply a univariate anomaly detection algorithm on the calculated Mahalanobis distance — it makes sense, because we converted our 2D data to 1D data by calculating the Mahalanobis distance. Now, this distance represents our data in 1D and we can use a Univariate anomaly detection method on it. Let us make a boxplot of the Mahalanobis distances in the data and find out the extreme distances by tukey’s method.


We can see 2 clear outliers near 100 — indeed those are the 2 anomalous points we have been finding so far.
We clearly see the 2 points near 100 as strong anomalies now. There is one more point near 20 that is being labelled as an anomaly — which needs to be analyzed further.

What we just did, is a standard technique — We converted a Multivariate Outlier detection problem into a univariate outlier detection problem by calculating the Mahalanobis distance of each point from the robust mean. Then, we applied a Univariate method on this distance.

Alternately, we can simply make a histogram and visually identify a good threshold.

NOTE: The Mahalanobis distance method works even for hundreds of features. For ease of Visualization, I have used 2 variables for the discussion.

Summary
We discussed the 3 major families of problems in Anomaly detection and the 3 major families of techniques used to solve them. We discussed how multivariate methods are important and can often give insights that cannot be made by EDA when we have high dimensional data. We discussed Robust methods of performing anomaly detection for Univariate and Multivariate cases. For the Multivariate techniques, we discussed the Robust variation of the Mahalanobis Distance method and discussed the general method of converting a multivariate Anomaly detection problem into a univariate problem. In the next articles we will discuss Autoencoders, Isolation Forests, OC-SVM among other methods.

Please feel free to let me know if you have any feedba