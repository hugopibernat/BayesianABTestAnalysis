#################################################
####### Author: Hugo Pibernat             #######
####### Contact: hugopibernat@gmail.com   #######
####### Date: April 2014                  #######
#################################################


from bayesianABTest import sampleSuccessRateForBinomial, sampleMeanForLogNormal, probabilityOfABetterThanB
from numpy.random import lognormal
from numpy import mean, concatenate, zeros

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

A_CR = sampleSuccessRateForBinomial(A_sessions,A_purchases)
B_CR = sampleSuccessRateForBinomial(B_sessions,B_purchases)

# Modeling the spend with a log-normal
A_non_zero_data = A_data[A_data > 0]
B_non_zero_data = B_data[B_data > 0]

A_spend = sampleMeanForLogNormal(A_non_zero_data)
B_spend = sampleMeanForLogNormal(B_non_zero_data)

# Combining the two
A_rps = A_CR*A_spend
B_rps = B_CR*B_spend

# Result:
print probabilityOfABetterThanB(A_rps,B_rps)