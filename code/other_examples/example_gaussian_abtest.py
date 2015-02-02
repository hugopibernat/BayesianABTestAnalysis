#################################################
####### Author: Hugo Pibernat             #######
####### Contact: hugopibernat@gmail.com   #######
####### Date: April 2014                  #######
#################################################


from bayesianABTest import sampleMeanAndVarianceForGaussian, probabilityOfABetterThanB
from numpy.random import normal
from numpy import mean

# Generate random data with Normal Distribution
A_data = normal(loc=4.1, scale=0.9, size=1000)
B_data = normal(loc=4.0, scale=1.0, size=1000) 

# Get samples from the posterior
A_mean,A_variance = sampleMeanAndVarianceForGaussian(A_data)
B_mean,B_variance = sampleMeanAndVarianceForGaussian(B_data)

# Result 1:
# Probability that the mean of A is greater than the mean of B
print probabilityOfABetterThanB(A_mean,B_mean)

# Result 2:
# probability that the variance of A is greater than the variance of B
print probabilityOfABetterThanB(A_variance,B_variance)