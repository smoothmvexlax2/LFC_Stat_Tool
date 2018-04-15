# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:55:55 2018

@author: Ben Heitkotter
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import sklearn as sk
import scipy as sp
import scipy.stats as st
import statsmodels.api as sm
#import statsmodels.formula.api as smfa
import seaborn as sns

#my path r'C:\Users\Owner\Downloads\train.csv
# read the data and store data in DataFrame titled melbourne_data
# print a summary of the data in Melbourne data
class MyInferTool:
    #reads comma seperated value formatted files
    #converts csv file to pandas dataframe
    def __init__(self, file_path, dependent_var=None):
        
        imported_data = self.collectData(file_path)
        self.data = imported_data
        self.dep_var = dependent_var 
        self.quant_vars = []
        self.qual_vars = []
        self.figureindex = 0
    
    def deleteVariable(self, var_name):
        
        if var_name in self.data.columns:
            del self.data[var_name]
            
        else:
            print('There was an issue with your variable name. Please check',
                  ' that you have the correct variable name.')
        
    def setDependentVar(self, var_name):
        '''
        The variable name needs to be a column name in this object's dataframe. 
        '''
        if var_name in self.data.columns:
            self.dep_var = var_name
            
        else:
            print('There was an issue with your variable name. Please check',
                  ' that you have the correct variable name.')
    
    def checkForNulls(self, trim=True, limit=.25):
        ''' The Default for this function is to delete columns with null values
        exceeding 25%. If trim is false a graph will provide details for each 
        column.
        '''
        df = self.data
        total = len(df.index)
        if trim:
            for var in df.columns:
                tot = df[var].isnull().sum()
                if (tot/total) > limit:
                    del df[var]
        else:
            results = {}
            for var in df.columns:
                tot = df[var].isnull().sum()
                if tot>0:
                    results[var]= tot/total
            if results:
                thetitle = "Variable Data Percent of Nulls"
            else:
                thetitle = "You Have No Variables With Null Data"

            plt.figure(self.figureindex)
            resultDf = pd.DataFrame(list(results.items()), list(results.keys()))
            resultDf.plot.bar(title=thetitle)
            plt.xlabel('Variables')
            plt.ylabel('Percent')
            self.figureindex += 1
    
    @property    
    def setQuantandQualData(self):
        if self.quant_vars or self.qual_vars:
            self.quant_vars = []
            self.qual_vars = []
        df = self.data
        for var in df.columns:
            if df[var].dtype.kind in 'bifc':
                #b bool, i int, f float, c complex
                self.quant_vars.append(var)
            else:
                self.qual_vars.append(var)
            
#another option is below, I believe one for loop is better than two inline loops
#despite inline loop being more efficient than appending 
                
#        self.quant_vars = [var for var in df.columns if df.dtypes[var] != 'object']
#        self.qual_vars = [var for var in df.columns if df.dtypes[var] == 'object']
    
    def distributionPlots(self, variable):
        if variable in self.quant_vars:
            y = self.data[variable]
            index = self.figureindex
            plt.figure(index) 
            plt.title('Johnson SU')
            sns.distplot(y, kde=False, fit=st.johnsonsu)
            plt.figure(index+1)
            plt.title('chisquare')
            sns.distplot(y, kde=False, fit=st.chi2)
            plt.figure(index+2)
            plt.title('Log Normal')
            sns.distplot(y, kde=False, fit=st.lognorm)
            plt.figure(index+3)
            plt.title('Normal')
            sns.distplot(y, kde=False, fit=st.norm)
            self.figureindex +=4
        else:
            print("Please check that your variable is quantitative and is",
                  "spelled correctly.")
    
    def scatterPlots(self, exclude=[], include=[]):
        if exclude and include:
            raise ValueError("You can't use both include and exclude.")
        elif exclude and not include:
            variables = [var for var in self.quant_vars if not(var in exclude)]
        elif not exclude and include:
            variables = [var for var in self.quant_vars if var in include]
        else:
            variables = self.quant_vars
        for var in variables:
            plt.figure(self.figureindex)
            self.data.plot.scatter(var, self.dep_var)
            plt.xlabel(var)
            plt.ylabel(self.dep_var)
            plt.title("Scatterplot of "+var+" With Respect to "+
                      self.dep_var)
            plt.show()
            self.figureindex +=1
    
    def residPlots(self, exclude=[], include=[]):
        ''' Using only quantitaive data for each variable in the include arg
        or variables not in the exclude arg plot the residual values with 
        respect to the independent variable. You do not want to see patterns 
        from these plots. 
        '''
        # y^ - y vs x
        
        if exclude and include:
            raise ValueError("You can't use both include and exclude.")
        elif exclude and not include:
            variables = [var for var in self.quant_vars if not(var in exclude)]
        elif not exclude and include:
            variables = [var for var in self.quant_vars if var in include]
        else:
            variables = self.quant_vars
        
        df = self.data
        
        for var in variables:
            plt.figure(self.figureindex)
            sns.residplot(var, self.dep_var, data=df)
            plt.title('Residual plot '+var)
            plt.show()
            self.figureindex +=1

    def residPlotsY2Y(self, exclude=[], include=[]):
        ''' Using only quantitaive data for each variable in the include arg
        or variables not in the exclude arg view two plots. 
            1) The residual values with respect to the dependent variable.
                One does not want to see patterns in these plots. 
            2) The predicted dependent values based on the linear model
               with respect to the actual dependent value.  
               One would hope to see something resembling a line
        '''
        # y^ - y vs y
        if exclude and include:
            raise ValueError("You can't use both include and exclude.")
        elif exclude and not include:
            variables = [var for var in self.quant_vars if not(var in exclude)]
        elif not exclude and include:
            variables = [var for var in self.quant_vars if var in include]
        else:
            variables = self.quant_vars
        
        df = self.data
        
        for var in variables:
            results = self.linreg(df[var], df[self.dep_var])
            
            yhat = results.predict()
            residual = df[self.dep_var].values[1:]-yhat
            
            plt.figure(self.figureindex)
            fig, axs = plt.subplots(nrows=1, ncols=2)
            ax1, ax2 = axs.flatten()
            
            ax1.scatter(residual,df[self.dep_var][1:])
            ax1.set_ylabel("Ys")
            ax1.set_xlabel("Residuals from "+var)
            ax1.set_title("Residuals vs Ys")
            
            ax2.scatter(yhat, df[self.dep_var][1:])
            ax2.set_xlabel("Estimated Ys from "+var)
            ax2.set_ylabel("Ys")
            ax2.set_title("Estimated Ys vs Ys")
            
            plt.xticks(rotation=45)#for ax2 x labels
            labels = ax1.get_xticklabels()
            plt.setp(labels, rotation=45)
            plt.show()
            
            self.figureindex +=1
    
    def oneWayAnova(self, include=[], exclude=[]):
        ''' Using qualitative variables that are assumed to be normally
        distributed for each variable in the include arg or variables not in 
        the exclude arg plot a graph to compare the potential impact each 
        variable has on the dependent variable. This method returns a pandas 
        dataframe for the user to check the pvalue for each variable's 
        difference in mean.
        '''
        if exclude and include:
            raise ValueError("You can't use both include and exclude.")
        elif exclude and not include:
            variables = [var for var in self.data.columns if not(var in exclude)]
        elif not exclude and include:
            variables = [var for var in self.data.columns if var in include]
        else:
            variables = self.data.columns
        
        df = self.data
        anv = {}
        anv['feature'] = variables
        pvals = []
        
        for var in variables:
            samples = []
            for cls in df[var].unique():
                #cls is a unique qualitative characteristic 
                #for example if you have car info, cls could be color
                s = df[df[var] == cls][self.dep_var].values
                samples.append(s)
            
            pval = st.f_oneway(*samples)[1]
            pvals.append(pval)
        anv['pvalue'] = pvals
        anv['disparity'] = np.log(1.0/np.array(anv['pvalue']))
        
        anvdf = pd.DataFrame(anv)
        
        plt.figure(self.figureindex)
        plt.title('Anova')
        sns.barplot(data=anvdf, x='feature', y='disparity')
        plt.xticks(rotation=90)
        plt.show()
        
        self.figureindex +=1
        
        return anvdf    
    
    def corrHeatMap(self, include=[], exclude=[]):
        '''
        This method takes qualitative and orders the values based on 
        their corresponding average dependant variable value. In other words,
        this takes car color and orders the colors based on the average value 
        of the cars with color "X". Then takes quantitative variables and runs
        the correlation amongst all other quantitative variables in one graph,
        and in 3rd graph depicts the correlation amongt all qualitative 
        variables.
        '''
        if exclude and include:
            raise ValueError("You can't use both include and exclude.")
        elif exclude and not include:
            variables = [var for var in self.data.columns if not(var in exclude)]
        elif not exclude and include:
            variables = [var for var in self.data.columns if var in include]
        else:
            variables = self.data.columns
        df = self.data
        qual_var = []
        quant_var = []
        
        for var in variables:
            if var in self.qual_vars:
                qual_var.append(var)
            else:
                quant_var.append(var)
                
        plt.figure(self.figureindex)
        plt.title('Quantitative Vars Corr to '+self.dep_var)
        corr = df[quant_var+[self.dep_var]].corr()
        sns.heatmap(corr)
        self.figureindex +=1
        
        plt.figure(self.figureindex)
        plt.title('Qualitative Vars Corr to '+self.dep_var)
        corr = df[qual_var+[self.dep_var]].corr()
        sns.heatmap(corr)
        self.figureindex +=1
        
        plt.figure(self.figureindex)
        plt.title('Qualitative Corr to Quantitative')
        corr = pd.DataFrame(np.zeros([len(quant_var)+1, len(qual_var)+1]),
                            index=quant_var+[self.dep_var],
                            columns=qual_var+[self.dep_var])
        for q1 in quant_var+[self.dep_var]:
            for q2 in qual_var+[self.dep_var]:
                corr.loc[q1, q2] = df[q1].corr(df[q2])
        sns.heatmap(corr)
        self.figureindex +=1
    
    def spearman(self, include=[], exclude=[]):
        '''
        This method is a non parameteric correlation plot and ought to be used 
        on the ranked qualitative data.  
        '''
        if exclude and include:
            raise ValueError("You can't use both include and exclude.")
        elif exclude and not include:
            variables = [var for var in self.qual_vars if not(var in exclude)]
        elif not exclude and include:
            variables = [var for var in self.qual_vars if var in include]
        else:
            variables = [var for var in self.qual_vars if '_E' in var]

        df = self.data
        spr = pd.DataFrame()
        spr['variable'] = variables
        spr['spearman'] = [df[var].corr(df[self.dep_var],
            'spearman') for var in variables]
        spr = spr.sort_values('spearman')
        
        plt.figure(self.figureindex)
        plt.title('Spearman Correlation')
        sns.barplot(data=spr, y='variable', x='spearman', orient='h')
        plt.show()
        
        self.figureindex +=1
        
    def rankQualData(self, include=[], exclude=[]):
        '''
        This method takes qualitative variables and orders the values based on 
        their corresponding average dependant variable value. In other words,
        this takes car color and orders the colors based on the average value 
        of the cars with color "X". 
        '''
        if exclude and include:
            raise ValueError("You can't use both include and exclude.")
        elif exclude and not include:
            variables = [var for var in self.qual_vars if not(var in exclude)]
        elif not exclude and include:
            variables = [var for var in self.qual_vars if var in include]
        else:
            variables = self.qual_vars
        
        
        index = 0
        for variable in variables:
            index +=1
            qual_var_df = pd.DataFrame()
            qual_var_df['value'] = self.data[variable].unique()
            qual_var_df.index = qual_var_df.value
            qual_var_df['dep_var_mean'] = pd.Series(
                    self.data[[variable, self.dep_var]].groupby(variable).mean()[self.dep_var],
                    )
            qual_var_df = qual_var_df.sort_values('dep_var_mean')
            qual_var_df['ordering'] = range(1, qual_var_df.shape[0]+1)
            
            qual_var_df = qual_var_df['ordering'].to_dict()
            
            for var, place in qual_var_df.items():
                self.data.loc[self.data[variable] == var, variable+'_E'] = place
            self.data.rename(columns={variable: variable+'_E'})   
        self.setQuantandQualData
    
    @staticmethod
    def collectData(path):
        try:
            file_data = pd.read_csv(path)
        except IOError:
            print('Path error, please double check your directory path. '+
                  'If you have not already try using a raw string for your path.')
        return(file_data)

    @staticmethod
    def linreg(x,y):
        '''
        y is variable we are interested in a.k.a. Endogenousis (dependent), 
        x is the benchmark variable (independent).
        '''
        x = sm.add_constant(x)[1:]
        y = y[1:]
        model = sm.OLS(y, x)
        result = model.fit()
        plt.figure()
        plt.plot(x.iloc[:,1], y, 'bo', label='original data')
        plt.plot(x.iloc[:,1], result.params[0] + result.params[1]*x.iloc[:,1], 'r',
                 label='fitted line')
        plt.xlabel(x.columns[1])
        plt.ylabel(y.name)
        plt.title('Regression with '+x.columns[1])
        plt.legend()
        return(result)
    
    
#def regress():
#    df = collectData()
#    quant, qual = splitDf(df)
#    figind = 1
#    for label in quant:
#        x1 = df[label]
#        y1 = df['SalePrice']
#        coef = np.corrcoef(x1 ,y=y1)
#        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x1, y1)
#        plt.figure(figind)
#        plt.plot(x1, y1, 'o', label='original data')
#        plt.plot(x1, intercept + slope*x1, 'r', label='fitted line')
#        plt.legend()
#        plt.title('Regression with '+label)
#        plt.show()
#        figind +=1
