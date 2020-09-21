import time
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/Pratt/Desktop/MBA/DataScienceProject')
cwd = os.getcwd()

rfrate = 0.025/250
rfrate20 = .0010/250

def findtangency(inverse_cov, excess_ret):
    ones = np.ones(excess_ret.shape[0]).reshape(1,-1)
    numerator = inverse_cov@excess_ret  
    denominator = np.sum(ones@inverse_cov@excess_ret)
    tp = numerator / denominator
    
    return tp
    
def Extract(lst): 
    return [item[0] for item in lst] 

def ConverttoDF(dfname):
    df_list = dfname.tolist()
    tickers_df = dict(zip(ticker_list, df_list))
    wtp_df = pd.DataFrame.from_dict(tickers_df)
    wtp_df2 = wtp_df.T
    wtp_df2.rename(columns={ wtp_df2.columns[0]: "Weight_PreCOVID" }, inplace = True)
    
    return wtp_df2
    
def calcPortfolioPerf(weights, meanReturns, covMatrix):
    '''
    Calculates the expected mean of returns and volatility for a portolio of
    assets, each carrying the weight specified by weights

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    tuple containing the portfolio return and volatility
    '''    
    #Calculate return and variance

    portReturn = np.sum( meanReturns*weights )
    portStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))

    return portReturn, portStdDev

#Test calcPWeights
#testweights = np.array([.25, .25, .25, .25])
#testrets = np.array([1, 1, 0, 0])
#testcov = np.identity(4)

#print(calcPortfolioPerf(testweights, testrets, testcov))
#testportreturn, testportstd = calcPortfolioPerf(testweights, testrets, testcov)
#print("Portfolio Return: ", testportreturn)
#print("Portfolio STD DEV: ", testportstd)
#OK - Return and STD DEV = 0.5, as expected

def negSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate):
    '''
    Returns the negated Sharpe Ratio for the speicified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    p_ret, p_stddev = calcPortfolioPerf(weights, meanReturns, covMatrix)

    return -(p_ret - riskFreeRate) / p_stddev

#testrf = 0.025
#testnegshrpe = negSharpeRatio(testweights, testrets, testcov, testrf)
#print("Portfolio Sharpe Ratio: ", testnegshrpe)
#OK - Neg Sharpe Ratio = -0.95, as expected

def getPortfolioVol(weights, meanReturns, covMatrix):
    '''
    Returns the volatility of the specified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    The portfolio's volatility
    '''
    return calcPortfolioPerf(weights, meanReturns, covMatrix)[1]

def findMaxSharpeRatioPortfolio(meanReturns, covMatrix, riskFreeRate):
    '''
    Finds the weights of assets providing the maximum Sharpe Ratio

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    numAssets = len(meanReturns)
    print(numAssets)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple( (0,1) for asset in range(numAssets))

    opts = sco.minimize(negSharpeRatio, numAssets*[1. / numAssets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return opts

#testopts = findMaxSharpeRatioPortfolio(testrets, testcov, testrf)
#print(testopts)

#OK - the weights are coming out as intended (0.5 and 0.5 for asset 1 and 2, 0 and 0 for asset 3 and 4, respectively)!

snp_url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'

tickers = pd.read_csv(snp_url)
tickers

#tickers.shape

symbols = tickers.Symbol.to_list()
symbols
len(symbols)
symbols2 = sorted(symbols)

#hist

year19 = yf.download(symbols2, start="2019-01-01", end="2019-12-31")

#4 Failed downloads:
#- BF.B: No data found for this date range, symbol may be delisted
#- BRK.B: No data found, symbol may be delisted
#- CARR: Data doesn't exist for startDate = 1546322400, endDate = 1577772000
#- OTIS: Data doesn't exist for startDate = 1546322400, endDate = 1577772000

symbols2.remove('BF.B')
symbols2.remove('BRK.B')
symbols2.remove('CARR')
symbols2.remove('OTIS')

print(len(symbols2))
#Length is 501 - results as expected

#CBOE volatility index spiked in mid March.  We would like to remove egregious volatility from analysis.
#However, from April to September, equities skyrocketed.  
#For now, the first day in our sample will be March 16, the first business day after the national emergency was declared.

year20 = yf.download(symbols2, start = "2020-03-15", end="2020-09-15")

year19.head()
colnames = year19.columns
colnames
category = colnames.to_list()
category

mylist = Extract(category)
myset = set(mylist)
print(myset)
#Categories are as follows:
#{'Open', 'Low', 'Volume', 'Close', 'Adj Close', 'High'}
#For analysis, we will use Adj Clsoe

year19adjclose = year19.drop('Open', axis = 1, level= 0).drop('Low', axis = 1, level = 0).drop('Volume', axis = 1, level = 0).drop('Close', axis = 1, level = 0).drop('High', axis = 1, level = 0)
year19adjclose.head()

year20adjclose = year20.drop('Open', axis = 1, level= 0).drop('Low', axis = 1, level = 0).drop('Volume', axis = 1, level = 0).drop('Close', axis = 1, level = 0).drop('High', axis = 1, level = 0)
year20adjclose.head()
#OK - our dataframe now only has the adjusted close prices, as expected


returns = (year19adjclose - year19adjclose.shift(1)) / year19adjclose.shift(1)
returns.head()

returns20 = (year20adjclose - year20adjclose.shift(1)) / year20adjclose.shift(1)
returns20.head()
#OK - need to drop the first data point

returns1 = returns.iloc[1:]
returns1.head()
#OK - data now begin on 1/3/2019

returns20_1 = returns20.iloc[1:]
returns20_1.head()
#OK - data now begin on 4/16/2020

meanrets = returns1.mean(axis=0).dropna()
meanrets_w_na = returns1.mean(axis=0)

print(meanrets.shape)
print(meanrets_w_na.shape)

meanrets20 = returns20_1.mean(axis=0).dropna()
meanrets20_w_na = returns20_1.mean(axis=0)

#Review summary stats
print("Summary Stats for 2019")
print(meanrets.describe().round(4))
print("Skewness for 2019:", meanrets.skew().round(4))
print("Kurtosis for 2019:", meanrets.kurt().round(4))
print("-------------------------")
print("Summary Stats for relevant 2020 Period")
print(meanrets20.describe().round(4))
print("Skewness for 2020:", meanrets20.skew().round(4))
print("Kurtosis for 2020:", meanrets20.kurt().round(4))
#2020 is more positively skewed, as expected

CoVMatrixID = np.identity(len(meanrets))
CovMatrixID20 = np.identity(len(meanrets20))

#Only done for initial exploratory analysis
#optimalweights = findMaxSharpeRatioPortfolio(meanrets, CoVMatrixID, rfrate)

#np.sum(optimalweights.x)
#OK - the weights indeed sum to 1

#print("Min Weight: ", np.min(optimalweights.x))
#print("Mean Weight: ", np.mean(optimalweights.x))
#print("Median Weight: ", np.median(optimalweights.x))
#print("Max Weight: ", np.max(optimalweights.x))
#np.savetxt("OptimalWeights.csv",optimalweights.x,delimiter=",")

#print(symbolsarray.shape)

#optwght = optimalweights.x.reshape(len(optimalweights.x),1)
#print(optwght.shape)


#symbol_w_wght = np.concatenate([symbolsarray, optwght], axis=1)
#print(symbol_w_wght)

#symbols_w_wght = pd.DataFrame(symbol_w_wght, columns = ["Ticker", "Weight"])
#symbols_w_wght["Weight"] = pd.to_numeric(symbols_w_wght["Weight"])
#print(symbols_w_wght.sort_values(by = "Weight", ascending = False))
#All of the portfolio is being allocated to HWM.  
#This optimization is as expected, given we are assuming the correlation matrix = identity matrix

#print(symbols_w_wght['Weight'].sum()) 

symbolsarray = np.array(symbols2).reshape(len(symbols2),1)
############Now, let's find the tangency portfolio returns############
excess_ret = meanrets - rfrate
excess_ret.shape
excess_ret_mat = excess_ret.values.reshape(len(excess_ret),1)
eye = np.zeros(len(excess_ret_mat)).reshape(1,len(excess_ret_mat)) + 1
inv_id_matrix = np.linalg.inv(CoVMatrixID)
wtp_id = (inv_id_matrix@excess_ret_mat) / (eye@inv_id_matrix@excess_ret_mat)

excess_ret20 = meanrets20 - rfrate20
excess_ret20.shape

excess_ret20_mat = excess_ret20.values.reshape(len(excess_ret20),1)
inv_id_matrix20 = np.linalg.inv(CovMatrixID20)

#print(excess_ret_mat)
#print(inv_id_matrix)
#print(eye.shape)
print('Sum of Weights: ', np.sum(wtp_id))
#OK - sum of weights = 1, as expected

print("Min Weight: ", np.min(wtp_id))
print("Mean Weight: ", np.mean(wtp_id))
print("Median Weight:", np.median(wtp_id))
print("Max Weight: ", np.max(wtp_id))

plt.hist(wtp_id, bins = 50)

tngtwght_id = wtp_id.reshape(len(wtp_id),1)
print(tngtwght_id.shape)

symbol_w_tngtwght_id = np.concatenate([symbolsarray, tngtwght_id], axis=1)
print(symbol_w_tngtwght_id)

symbol_w_tngtwght_id = pd.DataFrame(symbol_w_tngtwght_id, columns = ["Ticker", "Weight"])
symbol_w_tngtwght_id["Weight"] = pd.to_numeric(symbol_w_tngtwght_id["Weight"])
sorted_tngtwght = symbol_w_tngtwght_id.sort_values(by = "Weight", ascending = False)
#Top 20 long holdings
print(sorted_tngtwght.head(20))

#Top 20 short holdings
print(sorted_tngtwght.tail(20))
#All of the portfolio is being allocated to HWM.  
#This optimization is as expected, given we are assuming the correlation matrix = identity matrix

#print(returns1.columns)
#OK - Need to remove the first index from the multiindex function

returns2 = returns1
returns2.columns = returns2.columns.droplevel()
#np.savetxt("returns2.csv",returns2,delimiter=",")
print(returns2.head())
ret_for_covariance = returns2.drop(['BF.B', 'BRK.B', 'CARR', 'OTIS'], axis = 1)
print(ret_for_covariance.head())
#OK - we have 501 columns, as expected

returns20_2 = returns20_1
returns20_2.columns = returns20_2.columns.droplevel()
print(returns20_2.head())
ret_for_covariance20 = returns20_2
print(ret_for_covariance20.head())
#OK - we ahve 501 columns, as expected

standard_covariance_matrix = np.cov(ret_for_covariance, bias = True)
print(standard_covariance_matrix)

#Covariance matrix is returning NAN - there must be an issue with the returns
nanval = ret_for_covariance.isna().sum()
print(nanval.sort_values(ascending = False).head(20))
#Output:
#VIAC    235
#HWM     193
#CTVA    100
#DOW      54
#FOX      49
#FOXA     48
#NLOK     46
#To get a better estimate for our covariance matrix, we will drop these tickers

nanval20 = ret_for_covariance20.isna().sum()
print(nanval20.sort_values(ascending = False).head())
#Output:
#VIAC    14
#HWM      9
#To get a better estimate for our covariance matrix, we will drop these tickers

ret_for_covariance2 = ret_for_covariance.drop(['VIAC','HWM','CTVA','DOW','FOX','FOXA','NLOK'], axis = 1)
print(ret_for_covariance2.head())
#OK - we have 494 columns, as expected

ret_for_covariance20_2 = ret_for_covariance20.drop(['VIAC','HWM'], axis = 1)
print(ret_for_covariance20_2.head())
#OK - we have 499 columns, as expected

#To get a list of tickers, we'll convert the columns of the dataframe to an array
ticker_list = list(ret_for_covariance2.columns)
print(len(ticker_list))

ticker_list20 = list(ret_for_covariance20_2.columns)
print(len(ticker_list20))

covariance_matrix_standard = ret_for_covariance2.cov()
print(covariance_matrix_standard.shape)
print(covariance_matrix_standard[:1])
dummy = np.std(ret_for_covariance2['A'])*np.std(ret_for_covariance2['A']) 
print(round(dummy, 6))
#OK - the Covariance Matrix is being calculated as we expected.

covariance_matrix_standard20 = ret_for_covariance20_2.cov()
print(covariance_matrix_standard20.shape)
print(covariance_matrix_standard20[:1])
dummy = np.std(ret_for_covariance20_2['A'])*np.std(ret_for_covariance20_2['A'])
print(round(dummy, 6))
#OK - the Covariance Matrix is being calculated as we expected.

nanvalcov = covariance_matrix_standard.isna().sum()
print(nanvalcov.sort_values(ascending = False).head(20))
#Excellent - there are no NAN instances in the covariance matrix

nanvalcov20 = covariance_matrix_standard20.isna().sum()
print(nanvalcov20.sort_values(ascending = False).head(20))
#Excellent - there are no NAN instances in the covariance matrix

#Calculate tangency portfolio weights using Markowitz framework:
#First, need to delete the tickers with material missing data
returns1.columns = returns1.columns.droplevel()
returns2 = returns1.drop(columns = ['VIAC','HWM','CTVA','DOW','FOX','FOXA','NLOK', 'BF.B', 'BRK.B', 'CARR', 'OTIS'])
returns20_2 = returns20_1.drop(columns = ['VIAC','HWM'])

#print(returns2.shape)
#print(returns20_2.shape)
#OK - sizes are as expected.

#XX - CP to resume here - XX

meanrets_standard = returns2.mean(axis=0).dropna()
#print(meanrets_standard.shape)
#Excellent - 494 tickers, as expected
excess_standard_ret = meanrets_standard - rfrate
excess_standard_ret.shape
excess_standard_ret_mat = excess_standard_ret.values.reshape(len(excess_standard_ret),1)
eye_std = np.zeros(len(excess_standard_ret)).reshape(1,len(excess_standard_ret)) + 1
inv_cov_matrix = np.linalg.inv(covariance_matrix_standard)

#print(inv_cov_matrix)
#Ok - the covariance matrix blew up, as expected.  No economic meaning to this inverse matrix, but we'll inspect for pedagogical purposes

#############Hard-Coding of the tangency portfolio#############
#wtp_standard = (inv_cov_matrix@excess_standard_ret_mat) / (eye_std@inv_cov_matrix@excess_standard_ret_mat)

#print("Min Weight: ", np.min(wtp_standard))
#print("Mean Weight: ", np.mean(wtp_standard))
#print("Median Weight: ", np.median(wtp_standard))
#print("Max Weight: ", np.max(wtp_standard))
#############Hard-Coding of the tangency portfolio#############

#Supporting information: https://www.machinelearningplus.com/plots/matplotlib-histogram-python-examples/
# Information on pseudo inverse
# Wiki article: https://en.wikipedia.org/wiki/Tikhonov_regularization
# https://books.google.com/books?id=Jv_ZBwAAQBAJ&pg=PA86#v=onepage&q&f=false
# Business Data Science: Combining Machine Learning and Economics to Optimize, Automate, and Accelerate Business Decisions
# James Stein Regulator

#print(covariance_matrix_standard)
#print(inv_cov_matrix)

wtp_standard_function = findtangency(inv_cov_matrix, excess_standard_ret_mat)
wtp_id_function = findtangency(np.identity(excess_standard_ret_mat.shape[0]), excess_standard_ret_mat)
pseudo_inv = np.linalg.inv(covariance_matrix_standard.T@covariance_matrix_standard + 0.01*np.identity(494))@covariance_matrix_standard.T
wtp_pseudo = findtangency(pseudo_inv, excess_standard_ret_mat)
ones = np.ones(excess_standard_ret_mat.shape[0]).reshape(1,-1)

#3 sets of weights to examine:
#wtp_standard_function: tangency portfolio using the standard Markowitz framework
#wtp_id_function: tangency portfolio using the standard Markowitz framework, but using the identity matrix for the covariance matrix
#wtp_pseudo.values: tangency portfolio using a regularization algorithm to invert a matrix

plt.figure()
plt.hist(wtp_standard_function, bins = 50)
plt.gca().set(title = 'Tangency Portfolio (with Standard Markowitz Framework)', ylabel = 'Frequency')

plt.figure()
plt.hist(wtp_id_function, bins = 50)
plt.gca().set(title = 'Tangency Portfolio (Covariance Matrix = Identity Matrix)', ylabel = 'Frequency')

plt.figure()
plt.hist(wtp_pseudo.values, bins = 50)
plt.gca().set(title = 'Tangency Portfolio (with Pseudo-Inverse)', ylabel = 'Frequency')


#Begin Hard Code of ConverttoDF Function
#wtp_standard_function_list = wtp_standard_function.tolist()
#print(wtp_standard_function_list)

#tickers_wtp_standard = dict(zip(ticker_list, wtp_standard_function_list))

#wtp_standard_df = pd.DataFrame.from_dict(tickers_wtp_standard)
#wtp_standard_df2 = wtp_standard_df.T
#wtp_standard_df2.rename(columns={ wtp_standard_df2.columns[0]: "Weight" }, inplace = True)
#print(wtp_standard_df2)
#End Hard Code of ConverttoDF Function

#wtp_standard_df2_function = ConverttoDF(wtp_standard_function)
#print(wtp_standard_df2.equals(wtp_standard_df2_function))
#Excellent - function works as expected

wtp_standard_df = ConverttoDF(wtp_standard_function)
wtp_id_df = ConverttoDF(wtp_id_function)
wtp_pseudo_df = ConverttoDF(wtp_pseudo.values)

#OK - thsee matrices tell us the weights from the 2019.  Now, we must calculate the weights from a post_COVID

print(wtp_id_function[:5])
print("")
print(wtp_id[:5])

#print(wtp_standard[:5])
#print("")
#print(wtp_standard_function[:5])

#Eigenvalues
print(np.linalg.eig(covariance_matrix_standard)[0])

alpha = 0.01

pseudo_inverse = np.linalg.inv