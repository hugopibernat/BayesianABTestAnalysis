#################################################
####### Author: Hugo Pibernat             #######
####### Contact: hugopibernat@gmail.com   #######
####### Date: April 2014                  #######
#################################################
# Note: Parts of this code are inspired/based on Sergey Feldman's work on RichRelevance Engineering Blog
################################################


from numpy import sum, mean, size, sqrt, log, exp
from scipy.stats import norm, invgamma
from numpy.random import beta as beta_dist
from scipy.optimize import brentq as find_zeros

############# Posteriori Sampling #############

# Method: sampleSuccessRateForBinomialDataAndBetaPriori(...)
# Definition:
#   This method returns samples of the posterior distribution when:
#    - Data seems to be a Binomial random variable
#    - We use a Beta Distribution as the (conjugate) prior distribution
# Parameters
#   <data_n,data_k>:actual data_n trials and data_k successes observed in group A
#   <alpha,beta>: parameters of the Beta Distribution of the prior
#     Note: Uniform Distribution (alpha=beta=1)
#   <samples>: is the number of requestes samples from the posterior distribution
def sampleSuccessRateForBinomialDataAndBetaPriori(data_n,data_k,alpha=1,beta=1,samples=10000):
    return beta_dist(data_k+alpha, data_n-data_k+beta, samples)

def sampleSuccessRateForBinomial(data_n,data_k,alpha=1,beta=1,samples=10000):
    return sampleSuccessRateForBinomialDataAndBetaPriori(data_n,data_k,alpha,beta,samples)

# Method: sampleMeanAndVarianceForNormalDataAndGelmansPriori(...)
# Definition:
#   This method returns samples of the posterior (joint) distribution when:
#    - Data seems to be a Gaussian random variable
#    - We use Gelman et al. (p.79) joint prior distribution P(\mu,\sigma^2)=P(\sigma^2)P(\mu|\sigma^2):
#       * P(\sigma^2) is an Inverse Gamma Distribution
#       * P(\mu|\sigma^2) is a Gaussian Distribution
# Parameters
#   <data>: array of actual observations
#   <m0>: mean of the priori distribution
#   <k0>: certainty that m0 will be the mean of the posteriori distribution
#     Note: compare with number of observations
#   <s_sq0>: degrees of freedom of variance
#   <v0>: scale of the sigma_squared parameter
#     Note: compare with number of observations
#   <samples>: is the number of requestes samples from the posterior distribution
def sampleMeanAndVarianceForGaussianDataAndGelmansPriori(data,m0=0,k0=1,s_sq0=1,v0=1,samples=10000):
    # actual observations
    N = size(data)
    data_mean = mean(data) 
    sum_sd = sum( (data - data_mean)**2 ) 

    # combining the prior with the data - page 79 of Gelman et al.
    # to make sense of this note that 
    # inv-chi-sq(v,s^2) = inv-gamma(v/2,(v*s^2)/2)
    kN = float(k0 + N)
    mN = (k0/kN)*m0 + (N/kN)*data_mean
    vN = v0 + N
    vN_times_s_sqN = v0*s_sq0 + sum_sd + (N*k0*(m0-data_mean)**2)/kN

    # Sample \sigma^2 from the Inverse Gamma
    # (params: alpha, beta)
    # Note: if X ~ inv-gamma(a,1) then b*X ~ inv-gamma(a,b)
    alpha = vN/2
    beta = vN_times_s_sqN/2
    variance_samples = beta*invgamma.rvs(alpha,size=samples)

    # Sample \mu from a normal conditioned on the sampled \sigma^2
    # (params: mean_norm, var_norm)
    mean_norm = mN
    var_norm = sqrt(variance_samples/kN)
    mean_samples = norm.rvs(mean_norm,scale=var_norm,size=samples)

    return mean_samples, variance_samples

def sampleMeanAndVarianceForGaussian(data,m0=0,k0=1,s_sq0=1,v0=1,samples=10000):
    return sampleMeanAndVarianceForGaussianDataAndGelmansPriori(data,m0,k0,s_sq0,v0,samples)


# Method: sampleMeanForLogNormalDataAndGelmansNormalPriori(...)
# Definition:
#   This method returns samples of the posterior distribution when:
#    - Data seems to be a Log-Normal random variable (this method only returns samples of the mean)
#    - We use that if X~Log-N(\mu,\sigma^2) then ln(X)~N(\mu,\sigma^2)
#    - We use Gelman et al. (p.79) based method for sampling a Gaussian Distribution
# Parameters
#   <data>: array of actual observations
#   <m0>: mean of the priori distribution
#   <k0>: certainty that m0 will be the mean of the posteriori distribution
#     Note: compare with number of observations
#   <s_sq0>: degrees of freedom of variance
#   <v0>: scale of the sigma_squared parameter
#     Note: compare with number of observations
#   <samples>: is the number of requestes samples from the posterior distribution
def sampleMeanForLogNormalDataAndGelmansNormalPriori(data,m0=4,k0=1,s_sq0=1,v0=1,samples=10000):
    log_data = log(data)
    # get samples from the posterior
    mean_samples, variance_samples = sampleMeanAndVarianceForGaussian(log_data,m0,k0,s_sq0,v0,samples)
    # transform into log-normal means
    log_normal_mean_samples = exp(mean_samples + variance_samples/2)
    return log_normal_mean_samples

def sampleMeanForLogNormal(data,m0=4,k0=1,s_sq0=1,v0=1,samples=10000):
    return sampleMeanForLogNormalDataAndGelmansNormalPriori(data,m0,k0,s_sq0,v0,samples)


############# Bayesian Distribution Comparison Methods #############

def compareBinomialDistributions(A_n,A_k,B_n,B_k,precision=10000):
    A_p = sampleSuccessRateForBinomial(A_n,A_k)
    B_p = sampleSuccessRateForBinomial(B_n,B_k)

    return probabilityOfABetterThanB(A_p,B_p), A_p, B_p

def compareGaussianDistributions(A_data,B_data,precision=10000):
    A_mean,A_variance = sampleMeanAndVarianceForGaussian(A_data)
    B_mean,B_variance = sampleMeanAndVarianceForGaussian(B_data)

    return probabilityOfABetterThanB(A_mean,B_mean), A_mean, B_mean, probabilityOfABetterThanB(A_mean,B_mean), A_variance, B_variance

def compareLogNormalDistributions(A_data,B_data,precision=10000):
    A_spend = sampleMeanForLogNormal(A_data)
    B_spend = sampleMeanForLogNormal(B_data)

    return probabilityOfABetterThanB(A_spend,B_spend), A_spend, B_spend

def compareGaussianAndLogNormalCombinedDistribution(A_n,A_k,B_n,B_k,A_data,B_data,precision=10000):
    A_p = sampleSuccessRateForBinomial(A_n,A_k)
    B_p = sampleSuccessRateForBinomial(B_n,B_k)

    # Modeling the spend with a log-normal
    A_non_zero_data = A_data[A_data > 0]
    B_non_zero_data = B_data[B_data > 0]

    A_spend = sampleMeanForLogNormal(A_non_zero_data)
    B_spend = sampleMeanForLogNormal(B_non_zero_data)

    # Combining the two
    A_rps = A_p*A_spend
    B_rps = B_p*B_spend

    # Result:
    return probabilityOfABetterThanB(A_rps,B_rps), A_rps, B_rps, A_p, B_p, A_spend, B_spend

############# BayesianABTest Shortcuts Methods #############

def proportionsABTest(A_n,A_k,B_n,B_k,precision=10000):
    return compareBinomialDistributions(A_n,A_k,B_n,B_k,precision)

def attemptsPerLevelABTest(A_data,B_data,precision=10000):
    return compareGaussianVariable(A_data,B_data,precision)

def spendABTest(A_data,B_data,precision=10000):
    return compareLogNormalDistributions(A_data,B_data,precision)

def spendPerSessionABTest(A_n,A_k,B_n,B_k,A_data,B_data,precision=10000):
    return compareGaussianAndLogNormalCombinedDistribution(A_n,A_k,B_n,B_k,A_data,B_data,precision)

############# Bayesian Evaluation Methods #############

def probabilityOfABetterThanB(A,B):
    return mean( A > B )

def probabilityOfAWorseThanB(A,B):
    return probabilityOfABetterThanB(B,A)

def probabilityOfABetterThanBAndC(A,B,C):
    return mean( (A > B) & (A > C) )

def probabilityOfUpliftOfAOverB(A,B,uplift = 0.05):
    return mean( (A-B)/B > uplift )

def winnerOfAAndB(A,B):
    if probabilityOfABetterThanB(A,B):
        return A
    return B

def upliftOfAOverBWithProbability(A,B,probability=0.95):
    def F(x):
        return mean( (A-B)/B > x ) - probability
    return find_zeros(F,-1,1)



