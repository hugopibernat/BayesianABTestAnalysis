#################################################
####### Author: Hugo Pibernat             #######
####### Contact: hugopibernat@gmail.com   #######
####### Date: April 2014                  #######
#################################################


from bayesianABTest import compareBinomialDistributions, compareLogNormalDistributions, compareLogNormalDistributions, compareGaussianAndLogNormalCombinedDistribution
from numpy.random import normal, lognormal
from numpy import mean, concatenate, zeros


############# Example: Binomial Distribution #############

# Actual data for case A
A_sessions = 100000
A_purchases = 800
# Actual data for case B
B_sessions = 50000
B_purchases = 300

# Get samples from the posterior
result = compareBinomialDistributions(A_sessions,A_purchases,B_sessions,B_purchases)
print "The probability of A outperforming B is {}".format(result[0])

############# Example: Gaussian Distribution #############

# Generate random data with Normal Distribution
A_data = normal(loc=4.1, scale=0.9, size=1000)
B_data = normal(loc=4.0, scale=1.0, size=1000) 

# Get samples from the posterior
result = compareLogNormalDistributions(A_data,B_data)
print "The probability of A outperforming B is {}".format(result[0])


############# Example: Log-Normal Distribution #############

# Generate random data with Log Normal Distribution
data_A = lognormal(mean=4.05, sigma=1.0, size=500)
data_B = lognormal(mean=4.00,  sigma=1.0, size=500) 

# Get samples from the posterior
result = compareLogNormalDistributions(data_A,data_B)
print "The probability of A outperforming B is {}".format(result[0])

############# Example: Gaussian And Log-Normal Combined Distribution #############

# Generate Log-Normal data
A_actuals = lognormal(mean=4.10, sigma=1.0, size=100)
B_actuals = lognormal(mean=4.00, sigma=1.0, size=100)
# Plus some zeros
A_data = concatenate([A_actuals,zeros(10000)])
B_data = concatenate([B_actuals,zeros(10000)])

# Modeling conversions with a binomial variable
A_purchases = sum(A_data > 0)
A_sessions = len(A_data)
B_purchases = sum(B_data > 0)
B_sessions = len(B_data)

result = compareGaussianAndLogNormalCombinedDistribution(A_sessions,A_purchases,B_sessions,B_purchases,A_actuals,B_actuals)
print "The probability of A outperforming B is {}".format(result[0])

