"""
Functions that will do most of the basic econometric regression models and tests.
Based on the seminal book of Introduction to Econometrics by Jeffrey Wooldridge.
Author: Vinícius de Almeida Nery Ferreira (ECO - UnB)
E-mail: vnery5@gmail.com
Github: https://github.com/vnery5/Econometria
"""

## Importing
import pandas as pd
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

# Cute tables
from statsmodels.iolib.summary2 import summary_col
from stargazer.stargazer import Stargazer

# General
import os
import pathlib
import glob
from IPython.display import clear_output
import gc

####################################### Functions ###############################################################
####################################### .dta Data Collection ###########################################################
def get_data_stata(name=""):
    """
    Reads STATA (.dta) archives; the extension is not necessary.
    The file must be in the same folder as the .py ou .ipynb archive.
    Parameters
    ----------
    :param name: name/path of data to be read.
        Defaults to "", which reads the most recent added file to the folder

    Returns
    -------
    df: dataframe containing the data

    """

    # getting current files path
    path = pathlib.Path().absolute()

    # in my specific case:
    path_vinicius = f"{path}/datasets"

    ## checking to see if the name was the inserted or not;
    # if not, get the latest .dta file
    if name == "":
        try:
            file = max(glob.glob(f"{str(path)}/*.dta"), key=os.path.getctime)
            df = pd.read_stata(file)
        except Exception:
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
        except Exception:
            try:
                file = f"{str(path_vinicius)}/{str(name)}.dta"
                df = pd.read_stata(file)
                print(f"{name}.dta was read successfully!")
                return df
            except Exception:  # file not found
                print("It was not possible to find the requested file :(")
                print("Check the file name (without the extension) and if it is in the same directory as this program!")


####################################### Continuous Dependent Variables ##############################################
def ols_reg(formula, data, cov='unadjusted'):
    """
    Fits a standard OLS model with the corresponding covariance matrix.
    To compute without an intercept, use -1 in the formula.
    Remember to use mod = ols_reg(...).
    For generalized and weighted estimation, see statsmodels documentation or the first version of this file.
    :param formula: patsy formula (R style)
    :param data: dataframe containing the data
    :param cov : str
        unadjusted: common standard errors
        robust: HC1 standard errors
        cluster or clustered: clustered standard errors (must specify group)
    """

    # creating and fitting the model
    if cov == "robust":
        mod = ols(formula, data).fit(use_t=True, cov_type='HC1')
    elif cov == "cluster" or cov == "clustered":
        group = str(input("Which column is the group?"))
        try:
            mod = ols(formula, data).fit(use_t=True, cov_type='cluster', cov_kwds={'groups': data[group]})
        except Exception:
            erro = "It was not possible to find the selected group. Try again!"
            return erro
    else:
        mod = ols(formula, data).fit(use_t=True)

    ## printing the summary and returning the object
    print(mod.summary())
    return mod


def f_test(H0, model, level=0.05):
    """
    Calculates a F test based on H0 restrictions. Uses the same type of covariance as the model.
    It is not necessary to assign the function to an object!

    :param H0 : must be on standard patsy syntax ('(var1 = var2 =...), ...')
    :param model: fit instance (usually 'mod')
    :param level: significance level. Defaults to 5%
    """
    ## usually, we use the wald_test method from the fit instance
    # for panel models (from linearmodels), we must specify the parameter 'formula'
    try:
        test = 'LM'
        est = model.wald_test(formula=H0).stat
        p = model.wald_test(formula=H0).pval
    except Exception:
        test = 'F'
        est = float(str(model.wald_test(H0))[19:29])
        p = float(str(model.wald_test(H0))[36:47])

    if level > p:
        print(f"The value of {test} is {round(est, 6)} and its p-value is {round(p, 7)}.")
        print(f"Therefore, Ho is rejected at {level * 100}% (statistically significant).")
    else:
        print(f"The value of {test} is {round(est, 6)} and its p-value is {round(p, 7)}.")
        print(f"Therefore, Ho is NOT rejected at {level * 100}% (statistically NOT significant).")


def heteroscedascity_test(model, formula, data, level=0.05):
    """
    Executes the BP AND White test for heteroskedacity.
    It is not necessary to assign the function to an object!

    :param model : which model to use
        ols
        PooledOLS
        PanelOLS (FixedEffects)
        RandomEffects
        FirstDifferenceOLS
    :param formula : model formula
    :param data : dataframe
    :param level : significance level (defaults to 5%)
    """

    ## executing model choice
    try:  # for sm objects
        mod = model(formula, data).fit()
    except Exception:  # for linearmodels objects
        if model == "PanelOLS":
            formula += " + EntityEffects"
            mod = model.from_formula(formula, data, drop_absorbed=True).fit()
        else:
            mod = model.from_formula(formula, data).fit()

    ## getting the squares of residuals
    try:  # for sm objects
        res_sq = mod.resid ** 2
    except Exception:  # for linearmodels objects
        res_sq = mod.resids ** 2

    ## getting the squares of the predicted values (for White)
    predicted = mod.predict()
    predicted_sq = predicted ** 2

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
    mod_bp = ols(formula=bp_formula, data=data).fit()
    h0_bp = bp_formula.split(' ~ ')[1].replace('+', '=') + ' = 0'
    f_test(H0=h0_bp, model=mod_bp, level=level)

    print("\n############## WHITE ##############")
    mod_white = ols(formula=white_formula, data=data).fit()
    h0_white = white_formula.split(' ~ ')[1].replace('+', '=') + ' = 0'
    f_test(H0=h0_white, model=mod_white, level=level)


def reset_ols(formula, data, cov='normal', level=0.05):
    """
    Executes a RESET test for a OLS model specification, where H0: model is well specified
    It is not necessary to assign the function to an object!

    :param formula : patsy formula
    :param data : dataframe
    :param cov : str
        normal: common standard errors
        robust: HC1 standard errors
    :param level : significance level (default 5%)
    """
    ## getting covariance type
    if cov == 'normal':
        cov_type = 'nonrobust'
    else:
        cov_type = 'HC1'

    ## OLS model 
    mod = ols(formula=formula, data=data).fit(use_t=True, cov_type=cov_type)

    ## executing test
    test = linear_reset(mod, power=3, use_f=False, cov_type=cov_type)
    if test.pvalue < level:
        print(f"The test's p-value is equal to {np.around(test.pvalue, 6)} < {level * 100}%")
        print("Therefore, Ho is rejected (the model is badly specified).")
    else:
        print(f"The test's p-value is equal to {np.around(test.pvalue, 6)} > {level * 100}%")
        print("Therefore, Ho is NOT rejected (the model is not badly specified).")


def j_davidson_mackinnon_ols(formula1, formula2, data, cov='normal', level=0.05):
    """
    Executes a J test to verify which model is more adequate.
    H0 says that model 1 is preferable
    It is not necessary to assign the function to an object!

    :param formula1 : formula for the first model (use -1 for an origin regression)
    :param formula2: formula for the second model (use -1 for an origin regression)
    :param data : dataframe
    :param cov : str
        normal: common standard errors
        robust: HC1 standard errors
    :param level : significance level. defaults to 5%
    """
    ## getting covariance type
    if cov == 'normal':
        cov_type = 'nonrobust'
    else:
        cov_type = 'HC1'

    ## executing a regression of the second model
    mod = ols(formula=formula2, data=data).fit(use_t=True, cov_type=cov_type)

    ## getting predicted values and adding then to dataframe
    predicted = mod.predict()
    data['predicted'] = predicted

    ## adding the predicted values to the first formula
    formula1 += ' + predicted'

    ## executing regression
    mod1 = ols(formula=formula1, data=data).fit(use_t=True, cov_type=cov_type)

    ## getting p-value of the predicted coefficient
    p = mod1.pvalues[-1]

    if p < level:
        print(f"The test's p-value is equal to {np.around(p, 6)} < {level * 100}%")
        print("Therefore, Ho is rejected (model 2 is better specified).")
    else:
        print(f"The test's p-value is equal to {np.around(p, 6)} > {level * 100}%")
        print("Therefore, Ho is rejected (model 1 is better specified).")


####################################### Panel Models (linearmodels) ##########################################
def panel_structure(data, entity_column, time_column):
    """
    Takes a dataframe and creates a panel structure.

    :param data : dataframe
    :param entity_column : str, column that represents the individuals (1st level index)
    :param time_column : str, column that represents the time periods (2nd level index)
    """

    ## Creating MultiIndex and maintains columns in the DataFrame
    try:
        time = pd.Categorical(data[time_column])
        data = data.set_index([entity_column, time_column])
        data[time_column] = time  # creating a column with the time values (makes it easier to access it later)
        return data
    except Exception:
        print("One of the columns is not in the dataframe. Please try again!")
        return None


def pooled_ols(panel_data, formula, weights=None, cov="unadjusted"):
    """
    Fits a standard Pooled OLS model with the corresponding covariance matrix.
    Remember to include a intercept in the formula and to assign it to an object!

    :param panel_data : dataframe (which must be in a panel structure)
    :param formula : patsy formula
    :param weights : N x 1 Series or vector containing weights to be used in estimation; defaults to None
        Use is recommended when analyzing survey data, passing on the weight available in the survey
    :param cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        clustered: clustered standard errors by the entity column
    """

    ## Creating model instance
    if weights is None:
        mod = PooledOLS.from_formula(formula=formula, data=panel_data)
    else:
        mod = PooledOLS.from_formula(formula=formula, data=panel_data, weights=weights)
    
    ## Fitting with desired covariance matrix
    mod = mod.fit(cov_type='clustered', cluster_entity=True) if cov == 'clustered' else mod.fit(cov_type=cov)

    # Prints summary and returning
    print(mod.summary)
    return mod


def first_difference(panel_data, formula, weights=None, cov="unadjusted"):
    """
    Fits a standard FD model with the corresponding covariance matrix and WITHOUT an intercept.
    Remember to assign it to an object!
    
    :param panel_data : dataframe (which must be in a panel structure)
    :param formula : patsy formula
    :param weights : N x 1 Series or vector containing weights to be used in estimation; defaults to None
        Use is recommended when analyzing survey data, passing on the weight available in the survey
    :param cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        clustered: clustered standard errors by the entity column
    """
    ## Creating model instance
    if weights is None:
        mod = FirstDifferenceOLS.from_formula(formula=formula, data=panel_data)
    else:
        mod = FirstDifferenceOLS.from_formula(formula=formula, data=panel_data, weights=weights)
    
    ## Fitting with desired covariance matrix
    mod = mod.fit(cov_type='clustered', cluster_entity=True) if cov == 'clustered' else mod.fit(cov_type=cov)

    print(mod.summary)
    return mod


def fixed_effects(panel_data, formula, weights=None, time_effects=False, cov="unadjusted"):
    """
    Fits a standard Fixed Effects model with the corresponding covariance matrix.
    It can be estimated WITH and WITHOUT a constant.
    It is preferred when the unobserved effects are correlated with the error term
    and, therefore, CAN'T estimate constant terms.
    Remember to assign it to an object!

    :param panel_data : dataframe (which must be in a panel structure)
    :param formula : patsy formula
    :param weights : N x 1 Series or vector containing weights to be used in estimation; defaults to None
        Use is recommended when analyzing survey data, passing on the weight available in the survey
    :param time_effects : bool, defaults to False
        Whether to include time effects alongside entity effects (and estimate a 2WFE)
    :param cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        clustered: clustered standard errors by the entity column
    """

    ## Creating model instance
    # Defining which effects to control for
    formula += ' + EntityEffects + TimeEffects' if time_effects else ' + EntityEffects'

    ## Creating model instance
    if weights is None:
        mod = PanelOLS.from_formula(formula=formula, data=panel_data, drop_absorbed=True)
    else:
        mod = PanelOLS.from_formula(formula=formula, data=panel_data, drop_absorbed=True, weights=weights)

    ## Fitting with desired covariance matrix
    mod = mod.fit(cov_type='clustered', cluster_entity=True) if cov == 'clustered' else mod.fit(cov_type=cov)

    print(mod.summary)
    return mod


def random_effects(panel_data, formula, weights=None, cov="unadjusted"):
    """
    Fits a standard Random Effects model with the corresponding covariance matrix.
    It can be estimated WITH and WITHOUT a constant.
    It is preferred when the unobserved effects aren't correlated with the error term
    and, therefore, CAN estimate constant terms.
    Remember to assign it to an object!

    :param panel_data : dataframe (which must be in a panel structure)
    :param formula : patsy formula
    :param weights : N x 1 Series or vector containing weights to be used in estimation; defaults to None
        Use is recommended when analyzing survey data, passing on the weight available in the survey
    :param cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        clustered: clustered standard errors by the entity column
    """

    ## Creating model instance
    if weights is None:
        mod = RandomEffects.from_formula(formula=formula, data=panel_data)
    else:
        mod = RandomEffects.from_formula(formula=formula, data=panel_data, weights=weights)

    ## Fitting with desired covariance matrix
    mod = mod.fit(cov_type='clustered', cluster_entity=True) if cov == 'clustered' else mod.fit(cov_type=cov)

    print(mod.summary)
    return mod


def hausman_fe_re(panel_data, inef_formula, weights=None, cov="unadjusted", level=0.05):
    """
    Executes a Hausman test, which H0: there is not correlation between unobserved effects and the independent variables
    It is not necessary to assign the function to an object!

    :param panel_data : dataframe (which must be in a panel structure)
    :param inef_formula : patsy formula for the inefficient model under H0 (fixed effects)
    :param weights : N x 1 Series or vector containing weights to be used in estimation; defaults to None
        Use is recommended when analyzing survey data, passing on the weight available in the survey
    :param cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
    :param level : significance level for the test. Defaults to 5%.
    """

    ## Random Effects
    if weights is None:
        random = RandomEffects.from_formula(formula=inef_formula, data=panel_data).fit(cov_type=cov)
    else:
        random = RandomEffects.from_formula(formula=inef_formula, data=panel_data, weights=weights).fit(cov_type=cov)

    ## Fixed Effects
    formula_fe = inef_formula + ' + EntityEffects'
    if weights is None:
        fixed = PanelOLS.from_formula(formula=formula_fe, data=panel_data, drop_absorbed=True).fit(cov_type=cov)
    else:
        fixed = PanelOLS.from_formula(formula=formula_fe, data=panel_data,
                                      drop_absorbed=True, weights=weights).fit(cov_type=cov)

    ## Computing the Hausman statistic
    # difference between the asymptotic variance
    var_assin = fixed.cov - random.cov
    # difference between parameters
    d = fixed.params - random.params
    # calculating H
    H = d.dot(np.linalg.inv(var_assin)).dot(d)
    # calculating degrees of freedom
    freedom = random.params.size - 1

    # calculating p-value using chi2 survival function (1 - cumulative distribution function)
    p = stats.chi2(freedom).sf(H)

    if p < level:
        print(f"The value of H is {round(H, 6)} with {freedom} degrees of freedom in the chi-squared distribution.")
        print(f"The p-value of the test is {round(p, 6)} and, therefore, H0 is REJECTED and fixed effects is preferred")
    else:
        print(f"The value of H is {round(H, 6)} with {freedom} degrees of freedom in the chi-squared distribution.")
        print(f"The p-value of the test is {round(p, 6)} and H0 is NOT REJECTED and random effects is preferred.")


def iv_2sls(data, formula, weights=None, cov="unadjusted"):
    """
    Fits a 2SLS model with the corresponding covariance matrix.
    The endogenous terms can be formulated using the following syntax:
        lwage ~ 1 + [educ ~ psem + educ_married] + age + agesq
    Remember to use mod = iv_2sls(...)!

    :param data : dataframe
    :param formula : patsy formula
        The endogenous terms can be formulated using the following syntax:
        lwage ~ 1 + [educ ~ psem + educ_married] + age + agesq
    :param weights : N x 1 Series or vector containing weights to be used in estimation; defaults to None
        Use is recommended when analyzing survey data, passing on the weight available in the survey
    :param cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        clustered: clustered standard errors by the entity column
    """

    ## Creating model instance
    if weights is None:
        mod = IV2SLS.from_formula(formula=formula, data=data)
    else:
        mod = IV2SLS.from_formula(formula=formula, data=data, weights=weights)
    
    ## Fitting with desired covariance matrix
    mod = mod.fit(cov_type='clustered', cluster_entity=True) if cov == 'clustered' else mod.fit(cov_type=cov)

    ## Summary
    print(mod.summary)

    # Helpful information
    print("To see 1st stage results (and if the instruments are relevant with Partial P-Value), call 'mod.first_stage")
    print("To check if the instrumentated variable is exogenous, call 'mod.wooldridge_regression'.")
    print("To test for the instruments exogeneity (when they are more numerous than the number of endogenous variables")
    print("- therefore, are overidentified restrictions), call 'mod.wooldridge_overid' (Ho: instruments are exogenous)")

    ## Returning the object
    return mod


####################################### Discrete Dependent Variables and Selection Bias #############################
## MISSING: Heckit, Tobit and discontinuous/censored regressions
## Heckman procedures for sample correction can be imported from the Heckman.py file
# Alternatively, these models can be used in R, as exemplified in the file 'Tobit_Heckman.R'

def probit_logit(formula, data, model=probit, cov='normal'):
    """
    Creates a probit/logit model and returns its summary and average parcial effects (get_margeff).
    Documentation: https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_example.html
    Remember to use mod = probit_logit(...)!
    
    :param formula: patsy formula
    :param data: dataframe
    :param model: probit or logit. Defaults to probit.
    :param cov : str
        normal: common standard errors
        robust: HC1 standard errors
        cluster or clustered: clustered standard errors (must specify group)
    """

    # creating and fitting the model
    if cov == "robust":
        mod = model(formula, data).fit(use_t=True, cov_type='HC1')
    elif cov == "cluster" or cov == "clustered":
        group = str(input("What is the group column?"))
        try:
            mod = model(formula, data).fit(use_t=True, cov_type='cluster', cov_kwds={'groups': data[group]})
        except Exception:
            erro = "It was not possible to find the desired group. Check the spelling and the data and try again!"
            return erro
    else:
        mod = model(formula, data).fit(use_t=True)

    ## capturing the marginal effects
    mfx = mod.get_margeff(at='overall')
    clear_output()

    print(mod.summary())
    print("\n##############################################################################\n")
    print(mfx.summary())
    print(
        "\nMarginal effects on certain values can be found using 'mod.get_margeff(atexog = values).summary()', " +
        "where values must be generated using:\nvalues = dict(zip(range(1,n), values.tolist())).update({0:1})")
    print(
        "\nFor discrete variables, create a list of the values which you want to test and compute " +
        "'mod.model.cdf(sum(map(lambda x, y: x * y, list(mod.params), values)))")
    print("To predict values using the CDF, use mod.predict(X). X can be blank (use values from the dataset")
    print("or a K x N Dimensional array, where K = number of variables and N = number of observations.")

    return mod

def poisson_reg(formula, data, cov='normal'):
    """
    Creates a poisson model and returns its summary and average parcial effects (get_margeff).
    Documentation: https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_example.html
    Remember to use mod = poisson_reeg(...)!

    :param formula: patsy formula
    :param data: dataframe
    :param cov: str
        normal: common standard errors
        robust: HC1 standard errors
        cluster or clustered: clustered standard errors (must specify group)
    """
    # creating and fitting the model
    if cov == "robust":
        mod = poisson(formula, data).fit(use_t=True, cov_type='HC1')
    elif cov == "cluster" or cov == "clustered":
        group = str(input("What is the group column?"))
        try:
            mod = poisson(formula, data).fit(use_t=True, cov_type='cluster', cov_kwds={'groups': data[group]})
        except Exception:
            erro = "It was not possible to find the desired group. Check the spelling and the data and try again!"
            return erro
    else:
        mod = poisson(formula, data).fit(use_t=True)

    ## calculating under/overdispersion
    sigma = np.around((sum(mod.resid ** 2 / mod.predict()) / mod.df_resid) ** (1 / 2), 2)

    ## capturing the marginal effects
    mfx = mod.get_margeff(at='overall')
    clear_output()

    print(mod.summary())
    print(
        f"The coefficient to determine over/underdispersion is σ = {sigma}, " +
        f"which must be close to one for standard errors to be valid. " +
        f"If not, they must be multiplied by {sigma}.")

    print("##############################################################################")
    
    print(mfx.summary())
    print(
        "\nMarginal effects on certain values can be found using 'mod.get_margeff(atexog = values).summary()', " +
        "where values must be generated using:\nvalues = dict(zip(range(1,n), values.tolist())).update({0:1})")
    print(
        "\nUsually, the wanted effect of the poisson coefficients is it's semi-elasticity, which is 100*[exp(ß) - 1].")

    print("To predict values using the CDF, use mod.predict(X). X can be blank (use values from the dataset")
    print("or a K x N Dimensional array, where K = number of variables and N = number of observations.")

    return mod
