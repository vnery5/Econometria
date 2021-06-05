# Created by: Vinícius de Almeida Nery Ferreira (ECO - UnB)
#Github: https://github.com/vnery5/Econometria

## Importing
import pandas as pd
# import swifter # vectorizes function operations in dataframes
import numpy as np

# Linear Regression and Statistical Tests
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset

# Patsy formulas
from statsmodels.formula.api import logit, probit, poisson, ols

# Panel and IV regressions
from linearmodels import PanelOLS, FirstDifferenceOLS, PooledOLS, RandomEffects
from linearmodels.panel import compare as panel_compare
from linearmodels.iv import IV2SLS
from linearmodels.iv import compare as iv_compare

# Graphs
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Cutie tables
from statsmodels.iolib.summary2 import summary_col
from stargazer.stargazer import Stargazer

# General
import os
import pathlib
import glob
from IPython.display import clear_output
import gc
import subprocess #permite a cópia para o clipboard das equações gerados com as funções equation()

####################################### Functions ###############################################################
####################################### .dta Data Colection ###############################################################
def get_data(name = ""):
    '''
    Reads STATA (.dta) archives; the extension is not necessary
    The file must be in the same folder as the .py ou .ipynb archive
    Name = "" reads the most recent added file to the folder
    '''

    # getting current files path
    path = pathlib.Path().absolute()

    # in my specific case:
    path_vinicius = f"{path}/datasets"

    #checando se o nome foi inserido ou não; caso não, pegar o arquivo .dta mais recente
    if name == "":
        try:
            file = max(glob.glob(f"{str(path)}/*.dta"), key=os.path.getctime)
            df = pd.read_stata(file)
        except:
            file = max(glob.glob(f"{str(path_vinicius)}/*.dta"), key=os.path.getctime)
            df = pd.read_stata(file)
        
        print(f"{file}.dta was read successfully!")
        return df
    else:
        try:
            file = f"{str(path)}/{str(name)}.dta"
            df = pd.read_stata(file)
            print(f"{name}.dta was read successfully!")
            return df
        except:
            try:
                file = f"{str(path_vinicius)}/{str(name)}.dta"
                df = pd.read_stata(file)
                print(f"{name}.dta was read successfully!")
                return df
            except: # file not found
                print('''
                It was not possible to find the requested file :(\n
                Check if the name is correct (without the extension) and if the file is in the same directory as this program!
                ''')

####################################### Variáveis Dependentes Contínuas ###############################################################
def OLS_reg(formula, data, cov = 'unadjusted'):
    """
    Fits a standard OLS model with the corresponding covariance matrix.
    To compute without an intercept, use -1 in the formula.
    Remember to use mod = OLS_reg(...)
    cov : str
        unadjusted: common standard errors
        robust: HC1 standard errors
        cluster or clustered: clustered standard errors (must specify group)
    """

    # creating and fitting the model
    if cov == "robust":
        mod = ols(formula, data).fit(use_t = True, cov_type = 'HC1')
    elif cov == "cluster" or cov == "clustered":
        group = str(input("Which column is the group?"))
        try:
            mod = ols(formula, data).fit(use_t = True, cov_type = 'cluster', cov_kwds = {'groups':data[group]})
        except:
            erro = "It was not possible to find the selected group. Try again!"
            return erro
    else:
        mod = ols(formula, data).fit(use_t = True)

    ## printing the summary and returning the object
    print(mod.summary())
    return mod

def f_test(H0, model, level = 0.05):
    '''
    Calculates a F test based on H0 restrictions. Uses the same type of covariance as the model.
    It is not necessary to assign the function to an object!

    H0 : must be on standard patsy syntax ('(B1 = B2 =...), ...')
    model: fit instance (usually 'mod')
    '''
    ## usually, we use the wald_test method from the fit instance
    # for panel models (from linearmodels), we must specify the parameteer 'formula'ar o parâmetro 'formula', o que não ocorre com cortes transversais
    try:
        test = 'LM'
        est = model.wald_test(formula=H0).stat
        p = model.wald_test(formula=H0).pval
    except:
        test = 'F'
        est = float(str(model.wald_test(H0))[19:29])
        p = float(str(model.wald_test(H0))[36:47])

    if level > p:
        print(f"The value of {test} is {round(est,6)} and its p-value is {round(p,7)}.")
        print(f"Therefore, Ho is rejected at {level*100}% (statistically significant).")
    else:
        print(f"The value of {test} is {round(est,6)} and its p-value is {round(p,7)}.")
        print(f"Therefore, Ho is NOT rejected at {level*100}% (statistically NOT significant).")

def heteroscedascity_test(model, formula, data, level = 0.05):
    '''
    Executes the BP AND White test for heteroskedacity.
    It is not necessary to assign the function to an object!
    
    model : which model to use 
        ols
        PooledOLS
        PanelOLS (FixedEffects)
        RandomEffects
        FirstDifferenceOLS
    formula : model formula
    data : dataframe
    level : significance level (defaults to 5%)
    est : type of statistic to use (LM or F) (defaults to LM)
    '''

    ## executing model choice
    try: # for sm objects
        mod = model(formula, data).fit()
    except Exception: # for linearmodels objects
        if model == "PanelOLS":
            formula += " + EntityEffects"
            mod = model.from_formula(formula, data, drop_absorbed=True).fit()
        else:
            mod = model.from_formula(formula, data).fit()
    
    ## getting the squares of residuals
    try: # for sm objects
        res_sq = mod.resid**2
    except Exception: # for linearmodels objects
        res_sq = mod.resids**2
    
    ## getting the squares of the predicted values (for White)
    predicted = mod.predict()
    predicted_sq = predicted**2

    ## appending to dataframe
    data['res_sq'] = res_sq
    data['predicted'] = predicted
    data['predicted_sq'] = predicted_sq

    ## creating formulas
    bp_formula = 'res_sq ~ ' + formula.split(' ~ ')[1]
    white_formula = 'res_sq ~ predicted + predicted_sq'

    ## executing the tests
    print("H0: Error is homoscedastic.\n")
    print("############### BREUSCH-PAGAN ##############")
    mod_bp = ols(formula = bp_formula, data = data).fit()
    h0_bp = bp_formula.split(' ~ ')[1].replace('+','=') + ' = 0'
    f_test(H0 = h0_bp, model = mod_bp, level = level)

    print("\n############## WHITE ##############")
    mod_white = ols(formula = white_formula, data = data).fit()
    h0_white = white_formula.split(' ~ ')[1].replace('+','=') + ' = 0'
    f_test(H0 = h0_white, model = mod_white, level = level)

def reset_ols(formula, data, cov = 'normal', level = 0.05):
    '''
    Executes a RESET test for a OLS model specification, where H0: model is well specified
    It is not necessary to assign the function to an object!

    formula : patsy formula
    data : dataframe
    cov : str
        normal: common standard errors
        robust: HC1 standard errors
    level : significance level (default 5%)
    '''
    ## getting covariance type
    if cov == 'normal':
        cov_type = 'nonrobust'
    else:
        cov_type = 'HC1'

    ## OLS model 
    mod = ols(formula = formula, data = data).fit(use_t = True, cov_type = cov_type)
    
    ## executing test
    test = linear_reset(mod, power = 3, use_f = False, cov_type = cov_type)
    if test.pvalue < level:
        print(f"""
        The test's p-value is equal to {np.around(test.pvalue,6)} < {level*100}%\n
        Therefore, Ho is rejected (the model is badly specified)."""
        )
    else:
        print(f"""
        The test's p-value is equal to {np.around(test.pvalue,6)} > {level*100}%\n
        Therefore, Ho is NOT rejected (the model is not badly specified)."""
        )

def j_davidson_mackinnon_ols(formula1, formula2, data, cov = 'normal', level = 0.05):
    '''
    Executes a J test to verify which model is more adequate.
    H0 says that model 1 is preferable
    It is not necessary to assign the function to an object!
    
    formula1 : formula for the first model (use -1 for an origin regression)
    formula2: formula for the second model (use -1 for an origin regression)
    data : dataframe
    cov : str
        normal: common standard errors
        robust: HC1 standard errors
    level : significance level (default 5%)
    '''
    ## getting covariance type
    if cov == 'normal':
        cov_type = 'nonrobust'
    else:
        cov_type = 'HC1'

    ## executing a regression of the second model
    mod = ols(formula = formula2, data = data).fit(use_t = True, cov_type = cov_type)

    ## getting predicted values and adding then to dataframe
    predicted = mod.predict()
    data['predicted'] = predicted

    ## adding the predicted values to the first formula
    formula1 += ' + predicted'

    ## executing regression
    mod1 = ols(formula = formula1, data = data).fit(use_t = True, cov_type = cov_type)

    ## getting p-value of the predicted coefficient
    p = mod1.pvalues[-1]

    if p < level:
        print(f"""
        The test's p-value is equal to {np.around(p,6)} < {level*100}%\n
        Therefore, Ho is rejected (model 2 is better specified)."""
        )
    else:
        print(f"""
        The test's p-value is equal to {np.around(p,6)} > {level*100}%\n
        Therefore, Ho is NOT rejected (model 1 is better specified)."""
        )

####################################### Panel Models (linearmodels) ###############################################################
def panel_structure(data, entity_column, time_column):
    """
    Takes a dataframe and creates a panel structure.

    data : dataframe
    entity_column : str, column that represents the individuals (1st level index)
    time_column : str, column that represents the time periods (2nd level index)
    """
    try:
        time = pd.Categorical(data[time_column])
        data = data.set_index([entity_column,time_column])
        data[time_column] = time # creating a column with the time values (makes it easier to access it later)
    except Exception:
        print("One of the columns is not in the dataframe. Please try again!")
        return None
    
def pooled_ols(panel_data, formula, cov = "unadjusted"):
    """
    Fits a standard Pooleed OLS model with the corresponding covariance matrix.
    Remember to include a intercept in the formula!
    Remember to use mod = OLS_reg(...)!

    panel_data : dataframe (which must be in a panel structure)
    formula : patsy formula
    cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        cluster or clustered: clustered standard errors by the entity column
    """

    ## creating model instance
    mod = PooledOLS.from_formula(formula = formula, data = panel_data)

    if cov == 'clustered' or cov == 'cluster':
        mod = mod.fit(cov_type = 'clustered', cluster_entity = True)
    else:
        mod = mod.fit(cov_type = cov)
    
    print(mod.summary)
    return mod

def first_difference(panel_data, formula, cov = "unadjusted"):
    """
    Fits a standard FD model with the corresponding covariance matrix and WITHOUT an intercept.
    Remember to use mod = OLS_reg(...)!
    
    panel_data : dataframe (which must be in a panel structure)
    formula : patsy formula
    cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        cluster or clustered: clustered standard errors by the entity column
    """

    ## creating model instance
    mod = FirstDifferenceOLS.from_formula(formula = formula, data = panel_data)
    if cov == 'clustered' or cov == 'cluster':
        mod = mod.fit(cov_type = 'clustered', cluster_entity = True)
    else:
        mod = mod.fit(cov_type = cov)
    
    print(mod.summary)
    return mod

def fixed_effects(panel_data, formula, time_effects = False, cov = "unadjusted"):
    """
    Fits a standard Fixed Effects model with the corresponding covariance matrix.
    It can be estimated WITH and WITHOUT a constant.
    It is preferred when the unobserved effects are correlated with the error term
    and, therefore, CAN'T estimate constant terms.
    Remember to use mod = OLS_reg(...)!

    panel_data : dataframe (which must be in a panel structure)
    formula : patsy formula
    time_effects : bool, default to False
        Whether to include time effects alongside entity effects
    cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        cluster or clustered: clustered standard errors by the entity column
    """

    ## creating model instance
    if time_effects:
        formula += ' + EntityEffects + TimeEffects'
    else:
        formula += ' + EntityEffects'

    mod = PanelOLS.from_formula(formula = formula, data = panel_data, drop_absorbed=True)

    if cov == 'clustered' or cov == 'cluster':
        mod = mod.fit(cov_type = 'clustered', cluster_entity = True)
    else:
        mod = mod.fit(cov_type = cov)
    
    print(mod.summary)
    return mod

def random_effects(panel_data, formula, cov = "unadjusted"):
    """
    Fits a standard Random Effects model with the corresponding covariance matrix.
    It can be estimated WITH and WITHOUT a constant.
    It is preferred when the unobserved effects aren't correlated with the error term
    and, therefore, CAN estimate constant terms.
    Remember to use mod = OLS_reg(...)!

    panel_data : dataframe (which must be in a panel structure)
    formula : patsy formula
    cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        cluster or clustered: clustered standard errors by the entity column
    """
    ## creating model instance
    mod = RandomEffects.from_formula(formula = formula, data = panel_data)

    if cov == 'clustered' or cov == 'cluster':
        mod = mod.fit(cov_type = 'clustered', cluster_entity = True)
    else:
        mod = mod.fit(cov_type = cov)
    
    print(mod.summary)
    return mod

def hausman_fe_re(panel_data, inef_formula, level = 0.05, cov = "unadjusted"):
    """
    Executes a Hausman test, in which H0: there is not correlation between unobserved effects and the independent variables
    It is not necessary to assign the function to an object!

    panel_data : dataframe (which must be in a panel structure)
    inef_formula : patsy formula for the inefficient model under H0 (fixed effects)
    cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
    """
    ## Random Effects
    random = RandomEffects.from_formula(formula = inef_formula, data = panel_data).fit(cov_type = cov)

    ## Fixed Effects
    formula_fe = inef_formula + ' + EntityEffects'
    fixed = PanelOLS.from_formula(formula = formula_fe, data = panel_data, drop_absorbed=True).fit(cov_type = cov)
    
    ## Computing the Hausman statistic
    # difference between the assyntotic variance
    var_assin = fixed.cov - random.cov
    # difference between parameters
    d = fixed.params - random.params
    # calculating H
    H = d.dot(np.linalg.inv(var_assin)).dot(d)
    # calculating degrees of freedom
    freedom = random.params.size -1

    # calculating p-value using chi2 survival function (1 - cumulative distribution function)
    p = stats.chi2(freedom).sf(H)

    if p < level:
        print(f"""
        The value of H is {round(H,6)} with {freedom} degrees of freedoom in the chi-squared distribution.
        The p-value of the test is {round(p,6)} and, therefore, H0 is REJECTED and fixed effects is preferable.
        """)
    else:
        print(f"""
        The value of H is {round(H,6)} with {freedom} degrees of freedoom in the chi-squared distribution.
        The p-value of the test is {round(p,6)} and, therefore, H0 is NOT REJECTED and random effects is preferable.
        """)

def iv_2sls(data, formula, cov = "unadjusted"):
    """
    Fits a 2SLS model with the corresponding covariance matrix.
    The endogenous terms can be formulated using the following syntax:
        lwage ~ 1 + [educ ~ psem + educ_married] + age + agesq
    Remember to use mod = iv_2sls(...)!

    data : dataframe
    formula : patsy formula
    cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        cluster or clustered: clustered standard errors by the entity column
    """
    # creating and fitting the model
    if cov == "cluster" or cov == "clustered":
        group = str(input("Which column is the group?"))
        try:
            mod = IV2SLS.from_formula(formula = formula, data = data).fit(cov_type = 'clustered', cluster_entity = True)
        except:
            erro = "It was not possible to find the selected group. Try again!"
            return erro
    else:
        mod = IV2SLS.from_formula(formula = formula, data = data).fit(cov_type = cov)

    ## printing the summary
    print(mod.summary)
    # printing helpful information
    print("\nIn order to see the first stage results (and check if the instruments are relevant with the 'Partial P-Value'), call 'mod.first_stage'.")
    print("\nTo check if the instrumentated variable is exogenous, call 'mod.wooldridge_regression' or 'mod.wooldridge_regression'.")
    print("\nTo test for the exogeneity of the instruments (when they are more numerous than the number of endogenous variable - therefore, are overidentified restrictions), call 'mod.wooldridge_overid', where Ho: instruments are exogenous.\n")

    ## returning the object
    return mod

####################################### Variáveis Dependentes Discretas e Seleção Amostral ###############################################################
## MISSING: Tobit and discontinous/censored regressions
## Heckman procedures for sample correction can be imported from the Heckman.py file
# Alternatively, these models can be used in R, as examplified in the file 'Tobit_Heckman.R'

def probit_logit(formula, data, model = probit, cov ='normal'):
    """
    Creates a probit/logit model and returns its summary and average parcial effects (get_margeff).
    Documentation can be found at https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_example.html
    Remember to use mod = OLS_reg(...)!
    
    cov : str
        normal: common standard errors
        robust: HC1 standard errors
        cluster or clustered: clustered standard errors (must specify group)
    """
    # creating and fitting the model
    if cov == "robust":
        mod = model(formula, data).fit(use_t = True, cov_type = 'HC1')
    elif cov == "cluster" or cov == "clustered":
        group = str(input("Qual o rótulo da coluna de grupo?"))
        try:
            mod = model(formula, data).fit(use_t = True, cov_type = 'cluster', cov_kwds = {'groups':data[group]})
        except:
            erro = "Não foi possível encontrar o grupo selecionado. Tente novamente!"
            return erro
    else:
        mod = model(formula, data).fit(use_t = True)
        
    ## capturing the marginal effects
    mfx = mod.get_margeff(at = 'overall')
    clear_output()

    print(mod.summary())
    print("\n##############################################################################\n")
    print(mfx.summary())
    print("\nMarginal effects on certain values can be found using 'mod.get_margeff(atexog = values).summary()', where values must be generated using:\nvalues = dict(zip(range(1,n), values.tolist())).update({0:1})")
    print("\nFor discrete variables, create a list of the values which you want to test and compute 'mod.model.cdf(sum(map(lambda x, y: x * y, list(mod.params), values)))")

    return mod

def poisson_reg(formula, data, cov ='normal'):
    """
    Creates a poisson model and returns its summary and average parcial effects (get_margeff).
    Documentation can be found at https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_example.html
    Remember to use mod = OLS_reg(...)!
    
    cov : str
        normal: common standard errors
        robust: HC1 standard errors
        cluster or clustered: clustered standard errors (must specify group)
    """
    # creating and fitting the model
    if cov == "robust":
        mod = poisson(formula, data).fit(use_t = True, cov_type = 'HC1')
    elif cov == "cluster" or cov == "clustered":
        group = str(input("Qual o rótulo da coluna de grupo?"))
        try:
            mod = poisson(formula, data).fit(use_t = True, cov_type = 'cluster', cov_kwds = {'groups':data[group]})
        except:
            erro = "Não foi possível encontrar o grupo selecionado. Tente novamente!"
            return erro
    else:
        mod = poisson(formula, data).fit(use_t = True)
    
    ## calculating under/overdispersion
    sigma = np.around((sum(mod.resid**2/mod.predict())/mod.df_resid)**(1/2),2)

    ## capturing the marginal effects
    mfx = mod.get_margeff(at = 'overall')
    clear_output()

    print(mod.summary())
    print(f"The coefficient to determine over/underdispersion is σ = {sigma}, which must be close to one for standard errors to be valid. If not, they must be multiplied by {sigma}.")
    print("\n##############################################################################\n")
    print(mfx.summary())
    print("\nMarginal effects on certain values can be found using 'mod.get_margeff(atexog = values).summary()', where values must be generated using:\nvalues = dict(zip(range(1,n), values.tolist())).update({0:1})")
    print("\nUsually, the wanted effect of the poisson coefficients is it's semi-elasticity, which is 100*[exp(ß) - 1].")

    return mod