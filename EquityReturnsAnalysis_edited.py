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

def ConverttoDF20(dfname):
    df_list = dfname.tolist()
    tickers_df = dict(zip(ticker_list, df_list))
    wtp_df = pd.DataFrame.from_dict(tickers_df)
    wtp_df2 = wtp_df.T
    wtp_df2.rename(columns={ wtp_df2.columns[0]: "Weight_PostCOVID" }, inplace = True)
    
    return wtp_df2

def CalcWeightChange(dfname):
    dfname['weight_change'] = dfname['Weight_PostCOVID'] - dfname['Weight_PreCOVID']
    
    return dfname

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
returns20_1 = returns20.iloc[1:]

meanrets = returns1.mean(axis=0).dropna()
meanrets_w_na = returns1.mean(axis=0)

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

############Now, let's find the tangency portfolio returns############
symbolsarray = np.array(symbols2).reshape(len(symbols2),1)
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

print('Sum of Weights: ', np.sum(wtp_id))
#OK - sum of weights = 1, as expected

print("Min Weight: ", np.min(wtp_id))
print("Mean Weight: ", np.mean(wtp_id))
print("Median Weight:", np.median(wtp_id))
print("Max Weight: ", np.max(wtp_id))

tngtwght_id = wtp_id.reshape(len(wtp_id),1)
symbol_w_tngtwght_id = np.concatenate([symbolsarray, tngtwght_id], axis=1)

symbol_w_tngtwght_id = pd.DataFrame(symbol_w_tngtwght_id, columns = ["Ticker", "Weight"])
symbol_w_tngtwght_id["Weight"] = pd.to_numeric(symbol_w_tngtwght_id["Weight"])
sorted_tngtwght = symbol_w_tngtwght_id.sort_values(by = "Weight", ascending = False)
#Top 20 long holdings
print(sorted_tngtwght.head(20))

#Top 20 short holdings
print(sorted_tngtwght.tail(20))
#Majority of portfolio is being allocated to HWM - Howmet Aerospace Inc.  They had a terrific 2019.

returns2 = returns1
returns2.columns = returns2.columns.droplevel()
ret_for_covariance = returns2.drop(['BF.B', 'BRK.B', 'CARR', 'OTIS'], axis = 1)

returns20_2 = returns20_1
returns20_2.columns = returns20_2.columns.droplevel()
ret_for_covariance20 = returns20_2

standard_covariance_matrix = np.cov(ret_for_covariance, bias = True)

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
#print(covariance_matrix_standard[:1])
#dummy = np.std(ret_for_covariance2['A'])*np.std(ret_for_covariance2['A']) 
#print(round(dummy, 6))
#OK - the Covariance Matrix is being calculated as we expected.

covariance_matrix_standard20 = ret_for_covariance20_2.cov()
#print(covariance_matrix_standard20.shape)
#print(covariance_matrix_standard20[:1])
#dummy = np.std(ret_for_covariance20_2['A'])*np.std(ret_for_covariance20_2['A'])
#print(round(dummy, 6))
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


meanrets_standard = returns2.mean(axis=0).dropna()
#print(meanrets_standard.shape)
#Excellent - 494 tickers, as expected
excess_standard_ret = meanrets_standard - rfrate
#print(excess_standard_ret.shape)
excess_standard_ret_mat = excess_standard_ret.values.reshape(len(excess_standard_ret),1)
eye_std = np.zeros(len(excess_standard_ret)).reshape(1,len(excess_standard_ret)) + 1
inv_cov_matrix = np.linalg.inv(covariance_matrix_standard)

meanrets20_standard = returns20_2.mean(axis=0).dropna()
#print(meanrets20_standard.shape)
#Excellent - 499 tickers, as expected
excess_standard_ret20 = meanrets20_standard - rfrate20
#print(excess_standard_ret20.shape)
excess_standard_ret20_mat = excess_standard_ret20.values.reshape(len(excess_standard_ret20),1)
eye_std_20 = np.zeros(len(excess_standard_ret20)).reshape(1,len(excess_standard_ret20)) + 1
inv_cov_matrix20 = np.linalg.inv(covariance_matrix_standard20)

#Supporting information: https://www.machinelearningplus.com/plots/matplotlib-histogram-python-examples/
# Information on pseudo inverse
# Wiki article: https://en.wikipedia.org/wiki/Tikhonov_regularization
# https://books.google.com/books?id=Jv_ZBwAAQBAJ&pg=PA86#v=onepage&q&f=false
# Business Data Science: Combining Machine Learning and Economics to Optimize, Automate, and Accelerate Business Decisions
# James Stein Regulator
# In our pseudo-inverse, use a hand wave approach, we'll start witha an alpha value = 0.01

#print(covariance_matrix_standard)
#print(inv_cov_matrix)

wtp_standard = findtangency(inv_cov_matrix, excess_standard_ret_mat)
wtp_id = findtangency(np.identity(excess_standard_ret_mat.shape[0]), excess_standard_ret_mat)
pseudo_inv = np.linalg.inv(covariance_matrix_standard.T@covariance_matrix_standard + 0.01*np.identity(494))@covariance_matrix_standard.T
wtp_pseudo = findtangency(pseudo_inv, excess_standard_ret_mat)
ones = np.ones(excess_standard_ret_mat.shape[0]).reshape(1,-1)

wtp_standard20 = findtangency(inv_cov_matrix20, excess_standard_ret20_mat)
wtp_id20 = findtangency(np.identity(excess_standard_ret20_mat.shape[0]), excess_standard_ret20_mat)
pseudo_inv20 = np.linalg.inv(covariance_matrix_standard20.T@covariance_matrix_standard20 + 0.01*np.identity(499))@covariance_matrix_standard20.T
wtp_pseudo20 = findtangency(pseudo_inv20, excess_standard_ret20_mat)
ones20 = np.ones(excess_standard_ret20_mat.shape[0]).reshape(1,-1)

#3 sets of weights to examine:
#wtp_standard_function: tangency portfolio using the standard Markowitz framework
#wtp_id_function: tangency portfolio using the standard Markowitz framework, but using the identity matrix for the covariance matrix
#wtp_pseudo.values: tangency portfolio using a regularization algorithm to invert a matrix

###Histograms for 2019###
plt.figure()
plt.hist(wtp_standard, bins = 50)
plt.gca().set(title = 'Tangency Portfolio (with Standard Markowitz Framework) - 2019', ylabel = 'Frequency')

plt.figure()
plt.hist(wtp_id, bins = 50)
plt.gca().set(title = 'Tangency Portfolio (Covariance Matrix = Identity Matrix) - 2019', ylabel = 'Frequency')

plt.figure()
plt.hist(wtp_pseudo.values, bins = 50)
plt.gca().set(title = 'Tangency Portfolio (with Pseudo-Inverse) - 2019', ylabel = 'Frequency')
###Histograms for 2019###

###Histograms for 2020###
plt.figure()
plt.hist(wtp_standard20, bins = 50)
plt.gca().set(title = 'Tangency Portfolio (with Standard Markowitz Framework) - 2020', ylabel = 'Frequency')

plt.figure()
plt.hist(wtp_id20, bins = 50)
plt.gca().set(title = 'Tangency Portfolio (Covariance Matrix = Identity Matrix) - 2020', ylabel = 'Frequency')

plt.figure()
plt.hist(wtp_pseudo20.values, bins = 50)
plt.gca().set(title = 'Tangency Portfolio (with Pseudo-Inverse) - 2020', ylabel = 'Frequency')
###Histograms for 2020###

wtp_standard_df = ConverttoDF(wtp_standard)
wtp_id_df = ConverttoDF(wtp_id)
wtp_pseudo_df = ConverttoDF(wtp_pseudo.values)

wtp_standard20_df = ConverttoDF20(wtp_standard20)
wtp_id20_df = ConverttoDF20(wtp_id20)
wtp_pseudo20_df = ConverttoDF20(wtp_pseudo20.values)

wtp_standard_19_20 = wtp_standard_df.join(wtp_standard20_df, how='inner')
wtp_id_19_20 = wtp_id_df.join(wtp_id20_df, how='inner')
wtp_pseudo_19_20 = wtp_pseudo_df.join(wtp_pseudo20_df, how='inner')

CalcWeightChange(wtp_standard_19_20)
CalcWeightChange(wtp_id_19_20)
CalcWeightChange(wtp_pseudo_19_20)

#The weights for the standard tangency portfolio are too erratic.
#Therefore, for simplicity, we will refer to the weights derived from the identity covariance matrix
#and from the pseudo inverse methodology.
sorted_id_wght = wtp_id_19_20.sort_values(by = "weight_change", ascending = False)
sorted_pseudo_wght = wtp_pseudo_19_20.sort_values(by = "weight_change", ascending = False)

print(("Identity Matrix as Covariance Matrix:"))
print(sorted_id_wght.head(50))
print("")
print("Pseudo Inverse Methodology:")
print(sorted_pseudo_wght.head(50))

