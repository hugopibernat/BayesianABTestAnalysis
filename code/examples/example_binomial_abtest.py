#################################################
####### Author: Hugo Pibernat             #######
####### Contact: hugopibernat@gmail.com   #######
####### Date: April 2014                  #######
#################################################


from bayesianABTest import sampleSuccessRateForBinomial, probabilityOfABetterThanB, probabilityOfUpliftOfAOverB
from numpy import mean

# Actual data for case A
A_sessions = 100000
A_purchases = 800
# Actual data for case B
B_sessions = 50000
B_purchases = 300

# Sampling the posterior distribution
A_CR = sampleSuccessRateForBinomial(A_sessions,A_purchases)
B_CR = sampleSuccessRateForBinomial(B_sessions,B_purchases)

# Result 1:
# Which is the probability that CR higher than CR?
print probabilityOfABetterThanB(A_CR,B_CR)
# Result 2:
# Which is the probability that CR is 3% higher than CR
print probabilityOfUpliftOfAOverB(A_CR,B_CR,0.03)
