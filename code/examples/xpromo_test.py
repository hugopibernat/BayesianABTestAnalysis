#################################################
####### Author: Hugo Pibernat             #######
####### Contact: hugopibernat@gmail.com   #######
####### Date: April 2014                  #######
#################################################


from bayesianABTest import sampleSuccessRateForBinomial, sampleMeanForLogNormal, probabilityOfABetterThanB, probabilityOfABetterThanBAndC, upliftOfAOverBWithProbability
from numpy import mean, concatenate, loadtxt

# Number of samples to generate when performing numeric integration
accuracy = 10000

# Reading data
print "-- reading data"
dataset_dtype = [('casenum',int),('spend_usd',float)]
dataset = loadtxt('input/bayesian-test-input-case3-onlypositive.csv', dtype = dataset_dtype)
A_actuals_spend = [y for (x,y) in dataset if x==1]
B_actuals_spend = [y for (x,y) in dataset if x==2]
C_actuals_spend = [y for (x,y) in dataset if x==3]

numzeros = loadtxt('input/bayesian-test-input-case3-onlyzeros.csv', dtype=[('casenum',int),('zeros',int)])
A_not_spend = [y for (x,y) in numzeros if x==1][0]
B_not_spend = [y for (x,y) in numzeros if x==2][0]
C_not_spend = [y for (x,y) in numzeros if x==3][0]

# Modeling conversions with a binomial variable
print "-- modeling conversion with a binomial variable"
A_k = len(A_actuals_spend)
B_k = len(B_actuals_spend)
C_k = len(C_actuals_spend)
A_n = A_k + A_not_spend
B_n = B_k + B_not_spend
C_n = C_k + C_not_spend

A_conversions = sampleSuccessRateForBinomial(A_n,A_k,samples=accuracy)
B_conversions = sampleSuccessRateForBinomial(B_n,B_k,samples=accuracy)
C_conversions = sampleSuccessRateForBinomial(C_n,C_k,samples=accuracy)

# Modeling the spend with a log-normal
print "-- modeling spend with a log-normal variable"

A_spend = sampleMeanForLogNormal(A_actuals_spend,samples=accuracy)
B_spend = sampleMeanForLogNormal(B_actuals_spend,samples=accuracy)
C_spend = sampleMeanForLogNormal(C_actuals_spend,samples=accuracy)

# Combining samples
print "-- combining samples from both distributions"
A_rps = A_conversions*A_spend
B_rps = B_conversions*B_spend
C_rps = C_conversions*C_spend

# Result
print "-- result"
print "P(A>B and A>C) = {}".format(probabilityOfABetterThanBAndC(A_rps,B_rps,C_rps))
print "P(B>A and B>C) = {}".format(probabilityOfABetterThanBAndC(B_rps,A_rps,C_rps))
print "P(C>A and C>B) = {}".format(probabilityOfABetterThanBAndC(C_rps,A_rps,B_rps))

print "Summary:"
print "mean_A: {}  --  mean_B: {}  --  mean_C: {}".format(mean(A_rps),mean(B_rps),mean(C_rps))
print "A_k: {}  -- A_n: {}".format(A_k,A_n)
print "B_k: {}  -- B_n: {}".format(B_k,B_n)
print "C_k: {}  -- C_n: {}".format(C_k,C_n)

print ""
print "Find uplift with probability 0.95:"
print "upliftOfAOverBWithProbability(B_rps,A_rps,0.95) = {}".format(upliftOfAOverBWithProbability(B_rps,A_rps,0.95))

print "-- ready for more commands (if executed in the interpreter)"



