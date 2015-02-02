#################################################
####### Author: Hugo Pibernat             #######
####### Contact: hugopibernat@gmail.com   #######
####### Date: April 2014                  #######
#################################################


from bayesianABTest import sampleMeanForLogNormal, probabilityOfABetterThanB
from numpy.random import lognormal
from numpy import mean


# Generate random data with Log Normal Distribution
data_A = lognormal(mean=4.05, sigma=1.0, size=500)
data_B = lognormal(mean=4.00,  sigma=1.0, size=500) 

# Get samples from the posterior
A_mean = sampleMeanForLogNormal(A_data)
B_mean = sampleMeanForLogNormal(B_data)

# Result:
# Probability that the mean of A is greater than the mean of B
print probabilityOfABetterThanB(A_mean,B_mean)
