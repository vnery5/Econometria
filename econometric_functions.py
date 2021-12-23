"""
Functions that will do most of the basic econometric regressions, tests, panel models, IVs and Time Series applications.

Author: Vinícius de Almeida Nery Ferreira (FACE/ECO - University of Brasília (UnB)).

E-mail: vnery5@gmail.com

GitHub: https://github.com/vnery5/Econometria
"""

####################################### Imports #################################################################
import pandas as pd
import numpy as np

# Linear Regression and Statistical Tests
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Patsy formulas
from statsmodels.formula.api import logit, probit, poisson, ols
from patsy import dmatrices

# Panel and IV regressions
from linearmodels import PanelOLS, FirstDifferenceOLS, PooledOLS, RandomEffects
from linearmodels.panel import compare as panel_compare
from linearmodels.iv import IV2SLS
from linearmodels.iv import compare as iv_compare

# Time Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

# Graphs
import matplotlib.pyplot as plt
import seaborn as sns

# General
from IPython.display import clear_output
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


####################################### Functions ###############################################################
####################################### Continuous Dependent Variables ##########################################
def ols_reg(formula, data, cov='unadjusted'):
    """
    Fits a standard OLS model with the corresponding covariance matrix using an R-style formula (y ~ x1 + x2...).
    To compute without an intercept, use -1 or 0 in the formula.
    Remember to use mod = ols_reg(...).
    For generalized and weighted estimation, see statsmodels documentation or the first version of this file.
    :param formula: patsy formula (R style)
    :param data: dataframe containing the data
    :param cov : str
        unadjusted: common standard errors
        robust: HC1 standard errors
        cluster or clustered: clustered standard errors (must specify group)
    :return : statsmodels model instance
    """

    # Creating and fitting the model
    if cov == "robust":
        mod = ols(formula, data).fit(use_t=True, cov_type='HC1')
    elif cov == "cluster" or cov == "clustered":
        group = str(input("Which column is the group?"))
        try:
            mod = ols(formula, data).fit(use_t=True, cov_type='cluster', cov_kwds={'groups': data[group]})
        except KeyError:
            erro = "It was not possible to find the selected group. Check the spelling and try again!"
            return erro
    else:
        mod = ols(formula, data).fit(use_t=True)

    ## Printing the summary and returning the object
    print(mod.summary())
    return mod


def f_test(H0, model, level=0.05):
    """
    Calculates an F test based on H0 restrictions. Uses the same type of covariance as the model.
    It is not necessary to assign the function to an object!

    :param H0 : must be on standard patsy/R syntax ('(var1 = var2 =...), ...').
        For significance tests, the syntax is 'var1 = var2 = ... = 0'
    :param model: fit instance (usually 'mod')
    :param level: significance level. Defaults to 5%
    """
    ## Usually, we use the wald_test method from the fit instance
    # For panel models (from linearmodels), we must specify the parameter 'formula'
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
    :param formula : patsy/R formula of the model to be tested
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


def ols_diagnostics(formula, model, data, y_string):
    """
    Given the OLS model supplied, calculates statistics and draws graphs that check 4 of the 6 multiple linear
    regressions' hypothesis. Tests done: Harvey-Collier, Variance Influence Factor, RESET, Breusch-Pagan, Jarque-Bera.
    References:
        https://www.statsmodels.org/dev/examples/notebooks/generated/regression_diagnostics.html
        https://medium.com/@vince.shields913/regression-diagnostics-fa476b2f64db
    :param formula : patsy formula of the model;
    :param model : fitted model object;
    :param data : DataFrame containing the data;
    :param y_string : string (name) of the dependent variable
    """

    ## Harvey-Collier: linearity (MLR 1)
    try:
        print(f"Harvey-Collier P-value for linearity (MLR 1): {round(sms.linear_harvey_collier(model)[1], 4)}")
        print("H0: Model is linear.")
        print("For more information, see the 'Residuals vs Fitted Values' plot.\n")
    except ValueError:
        print("For information on linearity (MLR 1),  see the 'Residuals vs Fitted Values' plot.\n")

    ## Reset: specification of the functional form of the model
    reset = linear_reset(model, use_f=True, cov_type='HC1')
    print(f"Linear Reset (MLR 1) P-value: {reset.pvalue}")
    print("H0: model is well specified and linear.")
    print("For more information, see the Residuals vs Fitted Values plot.\n")

    ### Condition number: multicollinearity (MLR 3)
    print(f"Condition Number for Multicollinearity (MLR 3): {round(np.linalg.cond(model.model.exog), 2)}")
    print("The larger the number, the bigger the multicollinearity. For more information, see the 'VIF' plot.\n")

    ## Calculating Variance Influence Factors (VIF)
    # Matrices
    y, X = dmatrices(formula, data, return_type='dataframe')

    ## Calculating VIFs and storing in a DataFrame
    dfVIF = pd.DataFrame()
    dfVIF["Variables"] = X.columns
    dfVIF["VIF_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    ## Breusch-Pagan (MLR 5):
    breusch_pagan_pvalue = np.around(sms.het_breuschpagan(model.resid, model.model.exog)[3], 4)
    print(f"Breusch-Pagan P-value for heteroskedasticity (MLR 5): {breusch_pagan_pvalue}")
    print("H0: Variance is homoskedasticity.")
    print("For White's test and use in panel models, call the 'heteroscedascity_test' function.")
    print("For more information, see the 'Scale-Location' plot.\n")

    ## Durbin-Watson: correlation between the residuals
    print(f"Durbin-Watson statistic for residual correlation is: {np.around(durbin_watson(model.resid), 2)}")
    print("If the value is close to 0, there is positive serial correlation.")
    print("If the value is close to 4, there is negative serial correlation.")
    print("Rule of thumb: 1.5 < DW < 2.5 indicates no serial correlation.\n")

    ## Jarque-Bera: normality of the residuals (MLR 6, used for statistic inference)
    print(f"Jarque-Bera P-value (MLR 6): {np.around(sms.jarque_bera(model.resid)[1], 4)}")
    print("H0: Data has a normal distribution.")
    print("For more information, see the 'Normal Q-Q' plot.\n")

    print("To test for exogeneity (MLR 4), an IV2SLS must be constructed.")
    print("Test for random sampling (Heckit) are not yet available in this module.")

    ## Creating graphic object
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    plt.style.use('seaborn-white')

    ### Plots
    ## Linearity: residuals x predicted values. The less inclined the lowess, the more linear the model.
    ax00 = sns.residplot(x=model.fittedvalues, y=y_string, data=data, lowess=True,
                         scatter_kws={'facecolors': 'none', 'edgecolors': 'black'},
                         line_kws={'color': 'blue', 'lw': 1, 'alpha': 0.8}, ax=ax[0, 0])

    # Titles
    ax00.set_title('Linearity: Residuals vs Fitted', fontsize=12)
    ax00.set_xlabel('Fitted Values', fontsize=10)
    ax00.set_ylabel('Residuals (horizontal lowess: linearity)', fontsize=10)

    ## Multicollinearity: VIF
    ax01 = dfVIF["VIF_Factor"].plot(kind='bar', stacked=False, ax=ax[0, 1])

    # X tick labels
    ax01.set_xticklabels(labels=dfVIF["Variables"], rotation=0, color='k')

    # Annotations
    for p in ax01.patches:
        ax01.annotate(round(p.get_height(), 2), (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    ## Titles
    ax01.set_title("Multicollinearity Test - VIF", color='k', fontsize=12)
    ax01.set_ylabel("Variance Influence Factor (> 5: multicollinearity)", color='k', fontsize=10)
    ax01.set_xlabel("Variable", color='k', fontsize=10)

    ## Heteroskedasticity: the more disperse and horizontal the points,
    # the more likely it is that homoskedasticity is present
    ax10 = sns.regplot(x=model.fittedvalues, y=np.sqrt(np.abs(model.get_influence().resid_studentized_internal)),
                       ci=False, lowess=True, line_kws={'color': 'blue', 'lw': 1, 'alpha': 0.8},
                       scatter_kws={'facecolors': 'none', 'edgecolors': 'black'}, ax=ax[1, 0])

    # Titles
    ax10.set_title('Heteroskedasticity: Scale-Location', fontsize=12)
    ax10.set_xlabel('Fitted Values', fontsize=10)
    ax10.set_ylabel('$\sqrt{|Standardized Residuals|}$ (disperse and horizontal: homoskedasticity)', fontsize=10)

    ## Normality of the residuals: Q-Q Plot
    probplot = sm.ProbPlot(model.get_influence().resid_studentized_internal, fit=True)
    ax11 = probplot.qqplot(line='45', marker='o', color='black', ax=ax[1, 1])


def j_davidson_mackinnon_ols(formula1, formula2, data, cov='normal', level=0.05):
    """
    Executes a J test to verify which model is more adequate. H0 says that model 1 is preferable.
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


def cooks_distance_outlier_influence(model):
    """
    Calculates and plots the Cooks Distance metric, which shows the influence of individual points
    (dependent and independent variables) in the regression results.
    If a point is above D = 0.5, then it affects the regression results.
    High leverage: extreme X value; outlier: extreme y value.
    References:
        https://medium.com/@vince.shields913/regression-diagnostics-fa476b2f64db
        https://www.sthda.com/english/articles/39-regression-model-diagnostics/161-linear-regression-assumptions-and-diagnostics-in-r-essentials/
    :param model: fitted OLS model object.
    """

    ## Creating functions that define D = 0.5 and D = 1.0
    def one_line(x):
        return np.sqrt((1 * len(model.params) * (1 - x)) / x)

    def point_five_line(x):
        return np.sqrt((0.5 * len(model.params) * (1 - x)) / x)

    def show_cooks_distance_lines(tx, inc, color, label):
        plt.plot(inc, tx(inc), label=label, color=color, ls='--')

    ## Plotting
    sns.regplot(x=model.get_influence().hat_matrix_diag, y=model.get_influence().resid_studentized_internal,
                ci=False, lowess=True, line_kws={'color': 'blue', 'lw': 1, 'alpha': 0.8},
                scatter_kws={'facecolors': 'none', 'edgecolors': 'black'})

    show_cooks_distance_lines(one_line, np.linspace(.01, .14, 100), 'red', 'Cooks Distance (D=1)')

    show_cooks_distance_lines(point_five_line, np.linspace(.01, .14, 100), 'black', 'Cooks Distance (D=0.5)')

    plt.title('Residuals vs Leverage', fontsize=12)
    plt.xlabel('Leverage', fontsize=10)
    plt.ylabel('Standardized Residuals', fontsize=10)
    plt.legend()


####################################### Panel Models (linearmodels) #############################################
def xtdescribe_panel(data, entity_column):
    """
    Calculates the total appearances for each individual and checks how balanced the panel dataset is.

    :param data : dataframe
    :param entity_column : str, column that represents the individuals (what would be the 1st level index)
        Important: the fuction must be called BEFORE panel_structure

    :return : modified dataset with number of appearances column and prints how balanced the panel is
    """

    ## Number of appearances of each individual and adding as a column to the dataset
    data["number_appearances"] = data.groupby(entity_column)[entity_column].transform('count')

    ## Printing xtdescribe
    print(data.drop_duplicates(subset=[entity_column], keep='first')["number_appearances"].value_counts(normalize=True))

    return data


def panel_structure(data, entity_column, time_column):
    """
    Takes a dataframe and creates a panel structure.

    :param data : dataframe
    :param entity_column : str, column that represents the individuals (1st level index)
    :param time_column : str, column that represents the time periods (2nd level index)

    :return : modified DataFrame with the panel structure
    """

    ## Creating MultiIndex and mantaining columns in the DataFrame
    try:
        time = pd.Categorical(data[time_column])
        data = data.set_index([entity_column, time_column])
        data[time_column] = time  # creating a column with the time values (makes it easier to access it later)
        return data
    except KeyError:
        print("One of the columns is not in the dataframe. Please try again!")
        return None


def pooled_ols(panel_data, formula, weights=None, cov="unadjusted"):
    """
    Fits a standard Pooled OLS model with the corresponding covariance matrix.
    Remember to include an intercept in the formula ('y ~ 1 + x1 + ...') and to assign it to an object!

    :param panel_data : dataframe (which must be in a panel structure)
    :param formula : patsy formula
    :param weights : N x 1 Series or vector containing weights to be used in estimation; defaults to None
        Use is recommended when analyzing survey data, passing on the weight available in the survey
    :param cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        clustered: clustered standard errors by the entity column
    :return : linearmodels model instance
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
    :return : linearmodels model instance
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
    Remember to include an intercept in the formula ('y ~ 1 + x1 + ...') and to assign it to an object!

    :param panel_data : dataframe (which must be in a panel structure)
    :param formula : patsy/R formula (without EntityEffects, will be added inside the function)
    :param weights : N x 1 Series or vector containing weights to be used in estimation; defaults to None
        Use is recommended when analyzing survey data, passing on the weight available in the survey
    :param time_effects : bool, defaults to False
        Whether to include time effects alongside entity effects (and estimate a 2WFE)
    :param cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        clustered: clustered standard errors by the entity column
    :return : linearmodels model instance
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
    Remember to include an intercept in the formula ('y ~ 1 + x1 + ...') and to assign it to an object!

    :param panel_data : dataframe (which must be in a panel structure)
    :param formula : patsy formula
    :param weights : N x 1 Series or vector containing weights to be used in estimation; defaults to None
        Use is recommended when analyzing survey data, passing on the weight available in the survey
    :param cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        clustered: clustered standard errors by the entity column
    :return : linearmodels model instance
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
    Executes a Hausman test, which H0: there is no correlation between unobserved effects and the independent variables
    It is not necessary to assign the function to an object! But remember to include an intercept in the formulas.

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
    # Difference between asymptotic variances
    var_assin = fixed.cov - random.cov
    # Difference between parameters
    d = fixed.params - random.params
    # Calculating H (statistic)
    H = d.dot(np.linalg.inv(var_assin)).dot(d)
    # Degrees of freedom
    freedom = random.params.size - 1

    # Calculating p-value using chi2 survival function (sf, 1 - cumulative distribution function)
    p = stats.chi2(freedom).sf(H)

    if p < level:
        print(f"The value of H is {round(H, 6)} with {freedom} degrees of freedom in the chi-squared distribution.")
        print(f"The p-value of the test is {round(p, 6)} and, therefore, H0 is REJECTED and fixed effects is preferred")
    else:
        print(f"The value of H is {round(H, 6)} with {freedom} degrees of freedom in the chi-squared distribution.")
        print(f"The p-value of the test is {round(p, 6)} and H0 is NOT REJECTED and random effects is preferred.")


def iv_2sls(data, formula, weights=None, cov="robust", clusters=None):
    """
    Fits a 2SLS model with the corresponding covariance matrix.
    The endogenous terms can be formulated using the following syntax: lwage ~ 1 + [educ ~ psem + educ_married] + age...
    Remember to include an intercept in the formula ('y ~ 1 + x1 + ...') and to assign it to an object!

    :param data : dataframe
    :param formula : patsy formula ('lwage ~ 1 + [educ ~ psem + educ_married] + age + agesq...')
    :param weights : N x 1 Series or vector containing weights to be used in estimation; defaults to None
        Use is recommended when analyzing survey data, passing on the weight available in the survey
    :param cov : str
        unadjusted: common standard errors
        robust: robust standard errors
        kernel: robust to heteroskedacity AND serial autocorrelation
        clustered: clustered standard errors by the entity column
    :param clusters : str or list containing names of the DataFrame variables to cluster by
        Only should be used when cov="clustered"
    :return : linearmodels model instance
    """

    ## Creating model instance
    if weights is None:
        mod = IV2SLS.from_formula(formula=formula, data=data)
    else:
        mod = IV2SLS.from_formula(formula=formula, data=data, weights=weights)

    ## Fitting with desired covariance matrix
    mod = mod.fit(cov_type='clustered', clusters=data[clusters]) if cov == 'clustered' else mod.fit(cov_type=cov)

    ## Summary
    print(mod.summary)

    # Helpful information
    print("To see 1st stage results (and if the instruments are relevant with Partial P-Value), call 'mod.first_stage")
    print("To check if the instrumentated variable is exogenous, call 'mod.wooldridge_regression'.")
    print("To test for the instruments exogeneity (when they are more numerous than the number of endogenous variables")
    print("- therefore, are overidentified restrictions), call 'mod.wooldridge_overid' (Ho: instruments are exogenous)")

    ## Returning the object
    return mod


####################################### Discrete Dependent Variables and Selection Bias #########################
## MISSING: Heckit, Tobit and discontinuous/censored regressions
## Heckman procedures for sample correction can be imported from the Heckman.py file (unreleased version of statsmodels)

def probit_logit(formula, data, model=probit, cov='normal', marg_effects='overall'):
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
    :param marg_effects : str, either 'overall' (APE), 'mean' (PEA) or 'zero'. Defaults to 'overall' (APE).
    :return : statsmodels model instance
    """

    # Creating and fitting the model
    if cov == "robust":
        mod = model(formula, data).fit(use_t=True, cov_type='HC1')
    elif cov == "cluster" or cov == "clustered":
        group = str(input("What is the group column?"))
        try:
            mod = model(formula, data).fit(use_t=True, cov_type='cluster', cov_kwds={'groups': data[group]})
        except KeyError:
            erro = "It was not possible to find the desired group. Check the spelling and the data and try again!"
            return erro
    else:
        mod = model(formula, data).fit(use_t=True)

    ## Capturing the marginal effects
    mfx = mod.get_margeff(at=marg_effects)
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
    Creates a poisson model (counting y variable) and returns its summary and average parcial effects (get_margeff).
    Documentation: https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_example.html
    Remember to use mod = poisson_reg(...)!

    :param formula: patsy formula
    :param data: dataframe
    :param cov: str
        normal: common standard errors
        robust: HC1 standard errors
        cluster or clustered: clustered standard errors (must specify group)
    :return : statsmodels model instance
    """

    # Creating and fitting the model
    if cov == "robust":
        mod = poisson(formula, data).fit(use_t=True, cov_type='HC1')
    elif cov == "cluster" or cov == "clustered":
        group = str(input("What is the group column?"))
        try:
            mod = poisson(formula, data).fit(use_t=True, cov_type='cluster', cov_kwds={'groups': data[group]})
        except KeyError:
            erro = "It was not possible to find the desired group. Check the spelling and the data and try again!"
            return erro
    else:
        mod = poisson(formula, data).fit(use_t=True)

    ## Calculating under/overdispersion
    sigma = np.around((sum(mod.resid ** 2 / mod.predict()) / mod.df_resid) ** (1 / 2), 2)

    ## Capturing marginal effects
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


####################################### Time Series #########################
### Box-Jenkings: Identification #########################
def stationarity_test(vColumn, nLevel=0.05, sConstandTrend="c", bGraph=True, nBinsGraph=20):
    """
    Performs a stationarity test using the Augmented Dickey Fuller framework.
    By definition, a stationary series should have constant mean and variance/standard error.

    :param vColumn: DataFrame column/numpy vector in which to perform the test;
    :param nLevel: significance level; defaults to 0.05.
    :param sConstandTrend: which type of regression to perform in the test;
        'c' - constant
        'ct' - constant and trend
        'ctt' - constant, linear and quadratic trend
    :param bGraph: draw a graph displaying general and binned mean/standard error? Defaults to True.
    :param nBinsGraph: if a graph is drawn, in how many bins the data is to be split?
    """

    ## Executing test
    adf_results = adfuller(vColumn, regression=sConstandTrend)

    ## Getting p-value
    p_value = float(np.around(adf_results[1], 5))

    ## Printing result
    if p_value < nLevel:
        print(f"ADF p-value: {p_value}. The H0 of non-stationarity and unit root is rejected (series is stationary).")
        print("Consider some descriptive analysis using graphs and checking for NaNs! Also, create a DateTime index!")
    else:
        print(f"ADF p-value: {p_value}. The H0 of non-stationarity and unit root cannot be rejected.")
        print("Try differentiating the series using .diff() in order to make it stationary and ready for model use.")
        print("When testing the differenciated series, remember to remove the initial NaN.")
        print("Consider some descriptive analysis using graphs and checking for NaNs! Also, create a DateTime index!")

    ## Drawing graph (if requested)
    if bGraph:
        ## Binning data into nBinsGraph groups of (almost) equal size
        chunks = np.array_split(vColumn.values, nBinsGraph)

        ## List to store mean and SEs of bins
        vMeans, vSEs = [], []

        ## Statistic for each group
        for chunk in chunks:
            vMeans.append(np.mean(chunk))
            vSEs.append(np.std(chunk))

        ## Plotting
        plt.title('Means and Standard Deviations', size=20)
        plt.plot(np.arange(len(vMeans)), [np.mean(vColumn.values)] * len(vMeans), label='Global Mean', lw=1.5)
        plt.scatter(x=np.arange(len(vMeans)), y=vMeans, label='Chunk Means', s=100)
        plt.plot(np.arange(len(vSEs)), [np.std(vColumn.values)] * len(vSEs), label='Global SE', lw=1.5, color='orange')
        plt.scatter(x=np.arange(len(vSEs)), y=vSEs, label='Chunk SEs', color='orange', s=100)
        plt.legend()


def plot_autocorrelation(vColumn, nLags=12):
    """
    Plots the autocorrelation and partial autocorrelation function for vColumn.

    :param vColumn: DataFrame column/numpy vector;
    :param nLags: number of lags.
    """

    ## Plotting autocorrelation
    print("Plotting autocorrelation (determines 'Q' in ARIMA)...")
    print("Shaded area: zone of significance.")
    plot_acf(vColumn.tolist(), lags=nLags)

    ## Plotting partial autocorrelation
    print("Plotting partial autocorrelation (determines 'P' in ARIMA)...")
    plot_pacf(vColumn.tolist(), lags=nLags)


def cointegration(vColumn1, vColumn2, nLevel=0.05, sTrend='c'):
    """
    Performs the Engle-Granger test for cointegration. Both series must be differentiated ONE time.

    :param vColumn1: DataFrame column/numpy vector;
    :param vColumn2: DataFrame column/numpy vector;
    :param nLevel: level of significance
    :param sTrend: string containing which trend to consider:
        'c' - constant
        'ct' - constant and trend
        'ctt' - constant, linear and quadratic trend
    """

    ## Plotting
    plt.plot(vColumn1, color="red", label="Series 1")
    plt.plot(vColumn2, color="blue", label="Series 2")
    plt.legend()

    ## Performing test
    print("Reminder: both series must be I(1) (differentiated/integrated of order 1).")
    coint_test = coint(vColumn1, vColumn2, trend=sTrend)

    ## Getting p-value
    p_value = float(np.around(coint_test[1], 5))

    ## Printing result
    if p_value < nLevel:
        print(f"Coint p-value: {p_value}. The H0 of no cointegration is rejected (series are cointegrated).")
    else:
        print(f"Coint p-value: {p_value}. The H0 of no cointegration cannot be rejected (series are not cointegrated).")


### Box-Jenkings: Estimation #########################
def arima_model(vEndog, mExog=None, tPDQ=None):
    """
    Fits an ARIMA model. Order can be specified or determined by auto_arima.
    Differently from other models, it does not work on patsy/R formula syntax.

    :param vEndog: DataFrame column/numpy vector containing endogenous data (which will be regressed upon itself)
    :param mExog: vector/matrix containing exogenous data. Defaults to None
    :param tPDQ: tuple (p, d, q) containing order of the model;
        p: number of autorregressions (AR)
        q: number of differentiations (I)
        q: number of past prevision errors/moving averages (MA)
        If None (default), performs an auto_arima()

    :return mod: fitted model instance
    """

    ## Creating model
    # If order is specified
    if tPDQ is not None:
        # Conditional on whether there are exogenous variables
        if mExog is None:
            mod_arima = ARIMA(endog=vEndog, order=tPDQ).fit(cov_type='robust')
        else:
            mod_arima = ARIMA(endog=vEndog, exog=mExog, order=tPDQ).fit(cov_type='robust')
    # If order isn't specified, use auto_arima()
    else:
        mod_arima = auto_arima(y=vEndog, X=mExog)
        mod_arima = mod_arima.fit(y=vEndog, cov_type='robust')

    ## Printing summary and diagnostics
    print(mod_arima.summary())

    print("For heteroskdasticity, check Prob(H), where H0: homoskedasticity, and the standardized residual graph.")
    print("If there is hetero., the model error can't be a white noise (which is the desired thing).")
    print("Estimaed Density and Jarque-Bera have information on normality.")
    print("In the correlogram, all lollipops must be inside of the shaded area.")

    # Plots
    mod_arima.plot_diagnostics(figsize=(10, 10))
    plt.show()

    # Residual means
    tMean0 = stats.ttest_1samp(mod_arima.resid(), 0, nan_policy='omit')
    print(f"P-value for the test that residual mean is equal to 0: {np.around(tMean0[1], 5)}.")
    print("If p < 0.05, H0 is rejected and the residual mean is different from 0 (not ideal).")

    ## Returning
    return mod_arima


### Box-Jenkings: Diagnostics and Prediction #########################
def arima_fit_prediction(modARIMA, dfData, sColumn, nPeriods, sFreq,
                         sFitColumn="Fit", sFitPercentageErrorColumn="ErroFit"):
    """
    Investigates the fit and predicts nPeriods ahead. In order to work, dfData must have a date-like index.

    :param modARIMA: ARIMA model instance (from auto_arima())
    :param dfData: DataFrame containing the data; must contain a DateTime index
    :param sColumn: string of the column that contains the endogenous variable in modARIMA
    :param nPeriods: number of periods ahead to forecast
    :param sFreq: "months", "days", "years...". See pd.offsets.DateOffset for options.
    :param sFitColumn: string of the column that will be created containing fitted values
    :param sFitPercentageErrorColumn: string of the column that will be created containing fitted values % errors
    :return
        dfData: modified DataFrame containing fitted values and errors
        seriesPrediction: series containing the prediction of nPeriods ahead
    """

    ## Getting fitted values
    vFit = modARIMA.predict_in_sample((0, dfData[sColumn].shape[0] - 1))

    ## Adding to DataFrame
    dfData[sFitColumn] = vFit

    ## Calculating percentage error
    dfData[sFitPercentageErrorColumn] = 100 * np.abs((dfData[sColumn] - dfData[sFitColumn]) / dfData[sColumn])

    ## Describing errors
    print(f"Pseudo-R2: {np.around(stats.pearsonr(dfData[sColumn], dfData[sFitColumn])[0] ** 2, 4)}.")
    print("Describing percentage errors...")
    print(dfData[sFitPercentageErrorColumn].describe())

    ## Predicting nPeriods ahead
    vPrediction, mConfInt = modARIMA.predict(n_periods=nPeriods, return_conf_int=True)

    ## Creating index of prediction
    # Dictionaries to pass sFreq as argument using **
    dateStart = dfData.index[-1]
    dictPandasOffsetStart = {sFreq: 1}
    dictPandasOffsetEnd = {sFreq: 1 + nPeriods}
    # Index
    indexPrediction = pd.date_range(start=dateStart + pd.offsets.DateOffset(**dictPandasOffsetStart),
                                    end=dateStart + pd.offsets.DateOffset(**dictPandasOffsetEnd),
                                    periods=nPeriods, normalize=True).normalize()

    # Creating series to plot
    seriesPrediction = pd.Series(vPrediction, index=indexPrediction)
    lower_series = pd.Series(mConfInt[:, 0], index=indexPrediction)
    upper_series = pd.Series(mConfInt[:, 1], index=indexPrediction)

    ## Creating figure object
    fig, ax = plt.subplots(nrows=2, figsize=(10, 10))

    ## First plot: fit
    # Lines and legend
    ax[0].plot(dfData[sColumn], label="Valores Reais")
    ax[0].plot(dfData[sFitColumn], label="Modelo")
    ax[0].legend(frameon=False)

    # Titles
    ax[0].set(xlabel="Data", title="Valores Reais x Previstos")

    ## Second plot: prediction and confidence interval
    # Actual data
    ax[1].plot(dfData[sColumn])

    # Predictions and confidenc intervals
    ax[1].plot(seriesPrediction, color='darkgreen')
    ax[1].fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)

    # Titles
    ax[1].set(xlabel="Data", title=f"Previsão para {nPeriods} períodos a frente")

    ## Printing prediction values
    print("Prediction values:")
    print(seriesPrediction)

    ## Returning
    return dfData, seriesPrediction


def arima_train_test_prediction(dfData, sColumn, nPeriods, sFreq, mExog=None):
    """
    Investigates the fit and sees how the model competed with the last nPeriods of real data.
    In order to work, dfData must have a date-like index.

    :param dfData: DataFrame containing the data; must contain a DateTime index
    :param sColumn: string of the column that contains the endogenous variable in modARIMA
    :param nPeriods: number of periods ahead to forecast
    :param sFreq: "months", "days", "years...". See pd.offsets.DateOffset for options.
    :param mExog: vector/matrix containing exogenous data. Defaults to None

    :return dfTestPrediction: DataFrame containing train and test data
    """

    ## Splitting data into train and test
    vTrain = dfData[sColumn].values[:-nPeriods]
    vTest = dfData[sColumn].values[-nPeriods:]

    ## Fitting the model
    mod_arima = auto_arima(y=vTrain, X=mExog)
    mod_arima = mod_arima.fit(y=vTrain, cov_type='robust')

    ## Predicting the last nPeriods
    vPrediction, mConfInt = mod_arima.predict(n_periods=nPeriods, return_conf_int=True)

    ## Creating index of prediction
    # Dictionaries to pass sFreq as argument using **
    dateStart = dfData.index[-1 - nPeriods]
    dictPandasOffsetStart = {sFreq: 1}
    dictPandasOffsetEnd = {sFreq: nPeriods}
    # Index
    indexPrediction = pd.date_range(start=dateStart + pd.offsets.DateOffset(**dictPandasOffsetStart),
                                    end=dateStart + pd.offsets.DateOffset(**dictPandasOffsetEnd),
                                    periods=nPeriods, normalize=True).normalize()

    # Indexing train and test data
    seriesTrain = pd.Series(vTrain, index=dfData.index.values[:-nPeriods])
    seriesTest = pd.Series(vTest, index=dfData.index.values[-nPeriods:])

    # Creating series to plot
    seriesPrediction = pd.Series(vPrediction, index=indexPrediction)
    lower_series = pd.Series(mConfInt[:, 0], index=indexPrediction)
    upper_series = pd.Series(mConfInt[:, 1], index=indexPrediction)

    ## Plotting
    plt.figure(figsize=(12, 5), dpi=100)
    # Train
    plt.plot(seriesTrain, label="Treino")
    # Test
    plt.plot(seriesTest, label="Teste")

    # Prediction and Confidence Intervals
    plt.plot(seriesPrediction, color='darkgreen', label="Previsão ARIMA")
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)

    # Legend
    plt.legend()

    ## Calculating accuracy metrics
    print("Accuracy Metrics:")
    prediction_accuracy(vPrediction, vTest)

    ## Creating DataFrame, printing and returning
    dfTestPrediction = pd.DataFrame([seriesPrediction, seriesTest], index=["Previsão", "Real"]).T

    print("\nTest x Prediction:")
    print(dfTestPrediction)

    return dfTestPrediction


def prediction_accuracy(vPrediction, vTest):
    """
    Calculates metrics that show the (lack of) quality of the prediction made by an ARIMA model
    vPrediction and vTest must be the same size!

    :param vPrediction: unidimensional array containing predicted values
    :param vTest: unidimensional array containing real values
    """

    ## MAPE: mean absolute percentage error (erro absoluto percentual médio)
    mape = np.mean(np.abs(vPrediction - vTest) / np.abs(vTest))
    ## MAE: mean absolute error (erro absoluto medio)
    mae = np.mean(np.abs(vPrediction - vTest))
    ## RMSE: root mean squared error (raiz do erro quadrático médio)
    rmse = np.mean((vPrediction - vTest) ** 2) ** (1 / 2)
    ## Erro máximo absoluto
    erro_maximo = max(vTest - vPrediction)

    print(f"MAPE: {np.around(mape * 100, 4)}%")
    print(f"MAE: {np.around(mae, 4)}")
    print(f"RMSE: {np.around(rmse, 4)}")
    print(f"Erro Máximo: {np.around(erro_maximo, 4)}")


####################################### Policy Evaluation (TO DO) #########################
"""
Functions that will contain tools to evaluate policies. All methods have been implemented in their respective notebooks,
but have not yet been generalized to functions. Go check the notebooks in Notebooks/Avaliação de Políticas!
"""


def t_test_variables(dfDataPreTreatment, sColumnTreated, lVariables=None):
    """
    Loops through the columns in lVariables and performs t-test of means in order to see if
    treatment and control group are similar PRE-treatment. Ideally, they are only different in the outcome variable.
    If all/most of the test are insignificant, randomization was done properlly and
    OLS can be used to assess the program's results.

    :param dfDataPreTreatment: DataFrame containing all observations pre-treatment;
    :param sColumnTreated: string that identifies the 1/0 column that determines if an individual is treated or not;
    :param lVariables: list of variables to test. If None, does all column in dfDataPreTreatment

    """
    ## Counting ratio of treated and controls in respect to total
    print(dfDataPreTreatment[sColumnTreated].value_counts(normalize=True))

    ## Creating DataFrames
    dfTreatment = dfDataPreTreatment.query(f'{sColumnTreated} == 1')
    dfControl = dfDataPreTreatment.query(f'{sColumnTreated} == 0')

    ## Looping
    lVariables = lVariables if lVariables is not None else list(dfDataPreTreatment.columns)
    for sVariable in lVariables:
        ## Test
        tuplaTeste = stats.ttest_ind(dfTreatment[sVariable], dfControl[sVariable], nan_policy='omit')

        ## Checking to see if the difference is significant
        sAsterisco = "**" if tuplaTeste[1] < 0.05 else ""

        ## Getting means
        nMeanTreated = dfTreatment[sVariable].mean()
        nMeanControl = dfControl[sVariable].mean()

        ## Printing
        print(f"\n========================= {sVariable}{sAsterisco} =========================")
        print(f"Média Tratamento: {np.around(nMeanTreated, 2)}")
        print(f"Média Comparação: {np.around(nMeanControl, 2)}")
        print(f"Diferença = {np.around(nMeanTreated - nMeanControl, 2)}")
        print(f"Estatística = {np.around(tuplaTeste[0], 4)} \t P-valor = {np.around(tuplaTeste[1], 4)}")


def normalize(dfData, sColumnNormalization, sColumnToBeNormalized):
    """
    Normalizes a column (sColumnToBeNormalized) based on sColumnNormalization mean and std.

    :param dfData: DataFrame;
    :param sColumnNormalization: string that identifies the column which contains the characteristic used for norm.
    :param sColumnToBeNormalized: string that identifies the column whose values will be normalized.

    :return dfData: DataFrame with normalized column
    """

    ## Mean and Standard Deviation
    nMean = dfData[sColumnNormalization].mean()
    nSTD = dfData[sColumnNormalization].std()

    ## Z-score
    vZ = (dfData[sColumnNormalization] - nMean) / nSTD

    ## Normalized
    dfData[f"{sColumnToBeNormalized}_Norm"] = dfData[sColumnToBeNormalized] + (dfData[sColumnToBeNormalized].std() * vZ)

    ## Printing and returning
    print(dfData.head())
    return dfData
