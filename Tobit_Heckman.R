## importando os pacotes necessários
library(tidyverse)
library(survival)
library(haven)
library(AER)
library(plm)
library(stargazer)
library(sampleSelection)
library(glm2)

# rm(list=ls())
## lendo o df mroz <- read_dta("/Users/vinicius/Google Drive/UnB/Econometria/datasets/MROZ.dta")
df <- read_dta("/Users/vinicius/Downloads/PNAD/pnad_co_filtrada3.dta")

## removendo os novos e os velhinhos
df <- df[!(df$idade < 18 | df$idade >= 65),]

## vendo a estrutura do painel
#str(df)

## substituindo os valores de grau_educ, ppi, lsalh e nf05
names(df)[names(df) == 'fund_incompleto'] <- 'fund_inc'
names(df)[names(df) == 'fund_completo'] <- 'fund_comp'
names(df)[names(df) == 'medio_incompleto'] <- 'med_inc'
names(df)[names(df) == 'medio_completo'] <- 'med_comp'
names(df)[names(df) == 'superior_incompleto'] <- 'sup_inc'
names(df)[names(df) == 'superior_completo'] <- 'sup_comp'
names(df)[names(df) == 'preta_parda_ind'] <- 'ppi'
names(df)[names(df) == 'lsalariohora'] <- 'lsalh'
names(df)[names(df) == 'num_filhos_05'] <- 'nf05'

## criando idadesq e nf06m e renda_res_dom
df$idadesq <- df$idade^2
df$nf06m <- df$num_filhos_06_10 + df$num_filhos_11_18 + df$num_filhos_18m
df$renda_res_dom <- df$renda_hab_domiciliar - df$renda_hab_prin
df$renda_res_dom1000 <- df$renda_res_dom/1000

## criando variavel de emp
df$emp <- with(df, ifelse(fora_ft == 1 | desocupado == 1, 0, 1))

summary(df$emp)
## testando pra ver se tá tudo ok
formula = lsalh ~ 1 +educ + idade + idadesq + feminino + ppi

panel <- plm(formula, data = df, model = "pooling", index = c("idind",'data'))
summary(panel)

## rodando o modelo de mqo completo
formula = lsalh ~ educ + educ:ppi + educ:feminino + fund_inc + fund_inc:ppi + fund_comp + fund_comp:ppi + med_inc + med_inc:ppi + med_comp + med_comp:ppi + sup_inc + sup_inc:ppi + sup_comp + sup_comp:ppi + idade + idadesq + ppi + feminino + feminino:ppi + chefe_dom + casado + nf05 + nf05:ppi + DF + DF:ppi + rural + rural:ppi + cont_prev

mqo_na <- plm(formula, data = df, model = "pooling", index = c("idind",'data'))
summary(mqo_na)

df1 <- df

## substituindo valores nulos por 0
df1$renda_hab_hora[is.na(df1$renda_hab_hora)] <- 0
df1$lsalh <- log(df1$renda_hab_hora)

## substituindo valores nulos e infiinitos
df1$lsalh[is.na(df1$lsalh)] <- 0
df1$lsalh[which(!is.finite(df1$lsalh))] <- 0
df1$cont_prev[is.na(df1$cont_prev)] <- 0

## rodando mqo com todo mundo
mqo_0 <- plm(formula, data = df1, model = "pooling", index = c("idind",'data'))
summary(mqo_0)

## rodando um tobit
fmtobit <- AER::tobit(formula, data = df1, left = 0, x = TRUE)
summary(fmtobit)

## vendo o APE
fmtobit$scale
pnorm(sum(apply(fmtobit$x,2,FUN=mean) * fmtobit$coef)/fmtobit$scale) * fmtobit$coef

## HECKIT
pro = emp ~ 1 + educ + educ:ppi + fund_inc + fund_inc:ppi + fund_comp + fund_comp:ppi + med_inc + med_inc:ppi + med_comp + med_comp:ppi + sup_inc + sup_inc:ppi + sup_comp + sup_comp:ppi + idade + idadesq + ppi + chefe_dom + casado + nf05 + nf05:ppi + DF + DF:ppi + rural + rural:ppi + renda_res_dom
form = lsalh ~ 1 + educ + educ:ppi + fund_inc + fund_inc:ppi + fund_comp + fund_comp:ppi + med_inc + med_inc:ppi + med_comp + med_comp:ppi + sup_inc + sup_inc:ppi + sup_comp + sup_comp:ppi + idade + idadesq + ppi + chefe_dom + casado + nf05 + nf05:ppi + DF + DF:ppi + rural + rural:ppi + cont_prev

## contando os valores que aparecem em emp
# dplyr::count(df, emp, sort = TRUE)

## criando uma base só com mulheres
dffem <- df[!(df$feminino < 1),]
dffem$renda_res_dom[is.na(dffem$renda_res_dom)] <- 0
dffem$cont_prev[is.na(dffem$cont_prev)] <- 0

hfem <- heckit(pro, form, dffem)
summary(hfem)

# há uma seleção amostral (rho tem um p-valor de 0,003 e uma estimativa de -0,016), mas que não parece afetar os coeficientes
# como o coeficiente negativo se aplica a obs foras da amostra, parece haver viés positivo?

# estimando o modelo por MQO
mqo_fem <- plm(form, data = dffem, model = "pooling", index = c("idind",'data'))
summary(mqo_fem)

## tabela no latex
#stargazer(mqo_fem, hfem, title="MQOA vs Heckit", align=TRUE)