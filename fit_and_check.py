import scipy.stats as stats
import seaborn as sns

def fit_and_check( tofit, distribution):
 '''
 fit_and_check( tofit, distribution)
  
 Fits empirical data to a user-defined distribution, then performs a Kolmogorov-Smirnov test of the empirical data 
 against that fitted distribution. If the p-value of the test is low, it indicates that the data may not actually 
 be from the user-defined distribution. Note the KS test is a weak test, so a low p-value is a strong statement.

 inputs:  tofit - a list-like variable of the empirical data. 
          distribution - any of the subclasses of scipy.stats.rv_continuous
 outputs: ksresult - tuple of (ksstat, pvalue) describing the fit between tofit and distribution described by...
          pdfvars - the MLE parameters of distribution

 example usage:
          plt.figure()
          sns.distplot(tofit)
          ksresult, params = fit_and_check(tofit, stats.norm )
          print 'MLE fit to normal distr:', params, 'KS test of data against MLE fit', ksresult
 '''
  params = distribution.fit( tofit ) 
  ksresult = stats.kstest(tofit, distribution.name, args=params)
  return ksresult, params




