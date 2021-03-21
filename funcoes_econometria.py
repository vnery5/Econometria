#Criado por: Vinícius de Almeida Nery Ferreira (ECO - UnB)
#Github: https://github.com/vnery5/Econometria

#######################################################################################################################
###COMO USAR AS FUNÇÕES EM UM NOTEBOOK
##Antes, copie e cole todos os imports e definições daqui na primeira célula do notebook e pressione Shift + Enter
##Para coletar os dados do arquivo "carros.dta" (só funciona com arquivos .dta):
#coletar_dados("carros")

## Exportar resultados como imagem ou texto: https://stackoverflow.com/questions/46664082/python-how-to-save-statsmodels-results-as-image-file
#######################################################################################################################

##Importando os pacotes e módulos necessários
import pandas as pd
import numpy as np
import math

#Para Regressão Linear Simples e Teste F
from scipy import stats

#Para Regressão Linear Múltipla (OLS, GLS e WLS) e Testes Estatísticos
import statsmodels.api as sm
import econtools
import econtools.metrics as mt

#Para Regressão em Painel
from linearmodels import PanelOLS, FirstDifferenceOLS, PooledOLS, RandomEffects
from linearmodels.panel import compare

#Pacotes para gráficos (caso precise)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

#Pacotes para fazer a coleta dos dados armazenados no mesmo diretório e outros pacotes gerais
import os
import pathlib
import glob
from IPython.display import clear_output
import gc
import subprocess #permite a cópia para o clipboard das equações gerados com as funções equation()

####################################### Criando as Funções ###############################################################

def coletar_dados(nome = ""):
    '''
    Função que le os arquivos do Stata (.dta) - NÃO COLOQUE A EXTENSÃO NA HORA DE NOMEAR O "NOME"!
    O arquivo deve estar na mesma pasta do arquivo de Python ou do notebook do jupyter.
    Deixe em branco para ler o arquivo mais recentemente adicionado à pasta.
    '''

    global df

    #Pegando qual a pasta do arquivo que está sendo usado pra programar
    caminho = pathlib.Path().absolute()

    #No meu caso específico:
    caminho_vinicius = f"{caminho}/datasets"

    #checando se o nome foi inserido ou não; caso não, pegar o arquivo .dta mais recente
    if nome == "":
        try:
            arquivo = max(glob.glob(f"{str(caminho)}/*.dta"), key=os.path.getctime)
            df = pd.read_stata(arquivo)
            print(f"{arquivo}.dta foi lido com sucesso!")
            return df
        except:
            arquivo = max(glob.glob(f"{str(caminho_vinicius)}/*.dta"), key=os.path.getctime)
            df = pd.read_stata(arquivo)
            print(f"{arquivo}.dta foi lido com sucesso!")
            return df
    else:
        try:
            arquivo = f"{str(caminho)}/{str(nome)}.dta"
            df = pd.read_stata(arquivo)
            print(f"{nome}.dta foi lido com sucesso!")
            return df
        except:
            try:
                arquivo = f"{str(caminho_vinicius)}/{str(nome)}.dta"
                df = pd.read_stata(arquivo)
                print(f"{nome}.dta foi lido com sucesso!")
                return df
            except: #caso não tenha sido encontrado o arquivo com o nome inserido
                print('''
                Não foi possível achar o arquivo :(\n
                Verifique se seu nome está correto (sem a extensão) e se ele está no mesmo diretório do programa!
                ''')

def Regressao_Multipla(x, y, constante = "S", cov = "normal"):
    '''
    Função que calcula uma regressão múltipla, sendo, por default, computada com um intercepto e com erros padrões não robustos.

    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    constante: "S" para regressão com intercepto e qualquer outro valor para sem intercepto. Caso em branco, a regressão é computada com intercepto;
    cov: "normal" para regressão com erros-padrão tradicionais (caso padrão);
        "robust" para erros-padrões robustos.
        "cluster" ou "clustered" para erros-padrões clusterizados
    '''

    global Resultado, Lista_ychapeu, Resíduos, SQR, EPR

    #adicionando uma constante ao modelo de Ordinary Least Squares (OLS)
    if constante == "S":
        X = sm.add_constant(x)
    else:
        X = x

    #Criando o Modelo levando em conta a opção de erros padrão
    Modelo = sm.OLS(y,X)

    if cov == "robust":
        Resultado = Modelo.fit(cov_type = 'HC1', use_t = True)
    elif cov == "cluster" or cov == "clustered":
        group = str(input("Qual o rótulo da coluna de grupo?"))
        try:
            Resultado = Modelo.fit(cov_type = 'cluster',cov_kwds  ={'groups':df[group]}, use_t = True)
        except:
            erro = "Não foi possível encontrar o grupo selecionado. Tente novamente!"
            return erro
    else:
        Resultado = Modelo.fit()
    
    Lista_ychapeu = Resultado.predict()
    Resíduos = y - Lista_ychapeu

    #Calculando o Erro Padrão da Regressão (EPR)
    SQR =sum([i**2 for i in Resíduos])
    Número_de_Observações = len(y)
    GL = Número_de_Observações - len(Resultado.params)
    VarianciaReg = SQR/GL
    EPR = math.sqrt(VarianciaReg)
    
    ##Printando o Resultado
    print(Resultado.summary())

    print(f"O erro padrão da regressão é {round(EPR,5)} e a SQR é {round(SQR,5)}")
    print("\nPara ver os valores previstos ou os resídudos, basta chamar 'Lista_ychapeu' e 'Resíduos'.")
    print("Os resultados do modelo podem ser obtidos através de métodos usando a variável 'Resultado'.")
    print("""
    Valores de condição maiores que 20 indicam problemas de multicolinearidade.
    Para ver como achar esse número, entre em https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html"""
    )

def Regressao_MQP(x, y, pesos, constante = "S", cov = "normal"):
    '''
    Função que calcula uma regressão múltipla usando mínimos quadrados ponderados, ou seja,
    recomendada quando o erro é heteroscedástico E se sabe a função da constante. Ela é, por default, computada com um intercepto e com erros padrões não robustos.

    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    pesos: 1/h, sendo h a constante multiplicativa da variância do erro (ou seja, sem a raiz);
    constante: "S" para regressão com intercepto e qualquer outro valor para sem intercepto. Caso em branco, a regressão é computada com intercepto;
    cov: "normal" para regressão com erros-padrão tradicionais (caso padrão);
        "robust" para erros-padrões robustos.
        "cluster" para erros-padrões clusterizados
    '''

    global Resultado, Lista_ychapeu, Resíduos, SQR, EPR

    #adicionando uma constante ao modelo de Ordinary Least Squares(OLS)
    if constante == "S":
        X = sm.add_constant(x)
    else:
        X = x

    #Criando o Modelo levando em conta a opção de erros padrão
    Modelo = sm.WLS(y,X, weights = pesos)

    if cov == "robust":
        Resultado = Modelo.fit(cov_type = 'HC1', use_t = True)
    elif cov == "cluster" or cov == "clustered":
        group = str(input("Qual o rótulo da coluna de grupo?"))
        try:
            Resultado = Modelo.fit(cov_type = 'cluster',cov_kwds  ={'groups':df[group]}, use_t = True)
        except:
            erro = "Não foi possível encontrar o grupo selecionado. Tente novamente!"
            return erro
    else:
        Resultado = Modelo.fit()

    Lista_ychapeu = Resultado.predict()
    Resíduos = y - Lista_ychapeu

    #Calculando o Erro Padrão da Regressão (EPR)
    SQR =sum([i**2 for i in Resíduos])
    Número_de_Observações = len(y)
    GL = Número_de_Observações - len(Resultado.params)
    VarianciaReg = SQR/GL
    EPR = math.sqrt(VarianciaReg)
    
    ##Printando o Resultado
    print(f"O erro padrão da regressão é {round(EPR,5)} e a SQR é {round(SQR,5)}\n")
    print(Resultado.summary())

    print("\nPara ver os valores previstos ou os resídudos, basta chamar 'Lista_ychapeu' e 'Resíduos'.")
    print("Os resultados do modelo podem ser obtidos através de métodos usando a variável 'Resultado'.")
    print("""
    Valores de condição maiores que 20 indicam problemas de multicolinearidade
    Para ver como achar esse número, entre em https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html"""
    )
    
def Regressao_MQGF(x, y, constante = "S", cov = "normal"):
    '''
    Função que calcula uma regressão múltipla usando mínimos quadrados generalizados factíveis, ou seja,
    recomendada quando o erro é heteroscedástico E NÃO se sabe a função da constante multiplicativa da variância do erro, sendo os pesos estimados
    regridindo o log dos quadrados dos resíduos sobre as variáveis explicativas. Os estimadores MQP são gerados com o peso estimado.
    Ela é, por default, computada com um intercepto e com erros padrões não robustos.

    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    constante: "S" para regressão com intercepto e qualquer outro valor para sem intercepto. Caso em branco, a regressão é computada com intercepto;
    cov: "normal" para regressão com erros-padrão tradicionais (caso padrão);
        "robust" para erros-padrões robustos.
        "cluster" ou "clustered" para erros-padrões clusterizados
    '''

    global Resultado, Lista_ychapeu, Resíduos, SQR, EPR

    #Regredindo os valores normalmente a fim de pegar os resíduos
    Regressao_Multipla(x,y, constante, cov)
    clear_output()

    #Coletando o log dos quadrados dos resíduos
    Log_Res_Quad = np.log(Resíduos**2)

    #Regredindo Log_Res_Quad sobre as variáveis explicativas
    Regressao_Multipla(x,Log_Res_Quad, constante, cov)
    clear_output()

    #Estimando os pesos
    Pesos = np.exp(Lista_ychapeu)

    #Fazendo uma Regressão MQP
    Regressao_MQP(x,y, 1/Pesos, constante, cov)

def Teste_LM(x, y, Restrições, Nivel_de_Significância = 0.05):
    '''
    Função que calcula um teste LM e dá o resultado teste de hipótese para o caso de todas as restrições serem conjuntamente estatisticamente não-significantes.

    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    Restrições: lista ou array com os valores a serem tirados do modelo restrito;
    Nivel_de_Significância: nível de significância do teste. Caso branco, o nível de significancia padrão é de 5%.
    '''

    ##Definindo as variáveis de cada modelo
    ModeloIrrestrito = list(x)
    ModeloRestrito = []
    Restrições = list(Restrições)

    Numero_de_Observações = len(y)
    GL_r = len(Restrições)

    for i in ModeloIrrestrito:
        if i not in Restrições:
            ModeloRestrito.append(i)
    
    #Fazendo a regressão do modelo restrito e armazenando os resíduos
    Regressao_Multipla(df[ModeloRestrito], y)
    Resíduos_r = Resíduos

    #Fazendo a regressão dos resíduos sobre as variáveis independentes e armazenando o R2
    Regressao_Multipla(x, Resíduos_r)
    Ru = Resultado.rsquared

    #Calculando a estatística LM
    LM = Numero_de_Observações*Ru

    #Calculando o p-valor
    ##Calculando o P-valor de F
    P_valor = stats.chi2.sf(LM,GL_r)

    #Limpando a tela
    clear_output()

    #Printando o resultado
    if Nivel_de_Significância > P_valor:
        print(f"O valor de LM é {round(LM,3)} e seu p-valor é {round(P_valor,7)}. Portanto, rejeita-se Ho a um nível de significância de {Nivel_de_Significância*100}%.")
    else:
        print(f"O valor de LM é {round(LM,3)} e seu p-valor é {round(P_valor,7)}. Portanto, não se rejeita Ho a um nível de significância de {Nivel_de_Significância*100}%.")


def Teste_F(x, y, Restrições, Nivel_de_Significância = 0.05):
    '''
    Função que calcula um teste F e dá o resultado teste de hipótese para o caso de todas as restrições serem conjuntamente estatisticamente não-significantes.

    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    Restrições: lista ou array com os valores a serem tirados do modelo restrito;
    Nivel_de_Significância: nível de significância do teste. Caso branco, o nível de significancia padrão é de 5%.
    '''

    ##Definindo as variáveis de cada modelo
    #para testar igualdade dos coeficientes, F2, p_valueF2 = results.Ftest(['ACT', 'skipped'], equal=True)
    ModeloIrrestrito = list(x)
    ModeloRestrito = []
    Restrições = list(Restrições)

    Numero_de_Observações = len(y)
    GL_ir = Numero_de_Observações - (len(ModeloIrrestrito) + 1)
    GL_r = len(Restrições)

    for i in ModeloIrrestrito:
        if i not in Restrições:
            ModeloRestrito.append(i)

    ##Fazendo as regressões de cada modelo
    Regressao_Multipla(x, y)
    SQR_ir = SQR
    VarianciaReg_ir = EPR**2

    Regressao_Multipla(df[ModeloRestrito], y)
    SQR_r = SQR

    #Limpando a tela
    clear_output()
    
    ##Calculando F
    F = (SQR_r - SQR_ir)/(len(Restrições)*VarianciaReg_ir)

    ##Calculando o P-valor de F
    P_valor = stats.f.sf(F,GL_r,GL_ir)

    if Nivel_de_Significância > P_valor:
        print(f"O valor de F é {round(F,3)} e seu p-valor é {round(P_valor,7)}. Portanto, rejeita-se Ho à significância de {Nivel_de_Significância*100}%.")
    else:
        print(f"O valor de F é {round(F,3)} e seu p-valor é {round(P_valor,7)}. Portanto, não se rejeita Ho à significância de {Nivel_de_Significância*100}%.")

def Teste_F_Rapido_Robusto(H0, Nivel_de_Significância = 0.05):
    '''
    Função que calcula um teste F de forma mais rápida com base nas restrições de H0, podendo ser robusto se o Resultado for fruto de uma regressão robusta.
    H0 deve estar na forma B1 = B2 =...= Valor que deseja ser testado (0 na maioria das vezes)
    '''
    global Resultado
    ## A função utiliza o método wald_test dos resultados das regressões
    # Para modelos de painel - cujo método usa a estatística LM -, devemos especificar o parâmetro 'formula', o que não ocorre com cortes transversais
    try:
        teste = 'LM'
        est = Resultado.wald_test(formula=H0).stat
        p = Resultado.wald_test(formula=H0).pval
    except:
        teste = 'F'
        est = float(str(Resultado.wald_test(H0))[19:29])
        p = float(str(Resultado.wald_test(H0))[36:47])

    if Nivel_de_Significância > p:
        print(f"O valor de {teste} é {round(est,6)} e seu p-valor é {round(p,7)}.\nPortanto, rejeita-se Ho à significância de {Nivel_de_Significância*100}%, ou seja, as variáveis são conjuntamente significantes.")
    else:
        print(f"O valor de {teste} é {round(est,6)} e seu p-valor é {round(p,7)}.\nPortanto, NÃO se rejeita Ho à significância de {Nivel_de_Significância*100}%, ou seja, as variáveis NÃO são conjuntamente significantes.")

def Teste_t_Dois_Coeficientes_Iguais(x, y, Coeficientes_Testados_para_serem_iguais, Nivel_de_Significância = 0.05):
    '''
    Função que executa um teste t para verificar se dois coeficientes são iguais.

    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    Coeficientes_Testados_para_serem_iguais: array com os valores dos coeficientes que querem ser testados;
    Nivel_de_Significância: nível de significância do teste. Caso branco, o nível de significancia padrão é de 5%.
    '''
    
    ##Fazendo a regressão do modelo irrestrito
    Regressao_Multipla(x, y)
    clear_output()

    #Fazendo o objeto de lista que será usado no teste
    Teste =[0]
    Num_de_Variaveis = 1

    for i in list(x):
        if i not in list(Coeficientes_Testados_para_serem_iguais):
            Teste.append(0)
        elif (Num_de_Variaveis % 2 == 0):
            Teste.append(-1)
        else:
            Teste.append(1)
            Num_de_Variaveis += 1

    Teste_t = Resultado.t_test(Teste)
    print(f"A estatística do teste é {Teste_t.tvalue}, o que resulta em um p-valor bilateral de {Teste_t.pvalue} e em um p-valor unilateral de {Teste_t.pvalue/2}.")

def Teste_Heteroscedasticidade_BP(x, y, constante = "S", Nivel_de_Significância = 0.05, Estatística = "LM"):
    '''
    Função que executa o teste de Breusch-Pagan para a heteroscedasticidade.

    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    constante: "S" para regressão com intercepto e qualquer outro valor para sem intercepto. Caso em branco, a regressão é computada com intercepto;
    Nivel_de_Significância: nível de significância do teste. Caso branco, o nível de significancia padrão é de 5%.
    Estatística = LM ou F
    '''

    #Fazendo a regressão e limpando a tela
    Regressao_Multipla(x,y,constante)
    clear_output()

    #Calculando o quadrado dos resíduos
    Res_Quad = Resíduos**2

    #Realizando o teste F ou LM de Res_Quad sobre as variaveis dependentes para ver se há correlação
    if Estatística == "LM":
        Teste_LM(x, Res_Quad, x, Nivel_de_Significância)
        print("Ho: O erro é homoscedástico")
    else:
        Teste_F(x, Res_Quad, x, Nivel_de_Significância)
        print("Ho: O erro é homoscedástico")

def Teste_Heteroscedasticidade_White(x, y, constante = "S", Nivel_de_Significância = 0.05, Estatística = "LM"):
    '''
    Função que executa o teste de White (modificado por Wooldridge) para a heteroscedasticidade.

    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    constante: "S" para regressão com intercepto e qualquer outro valor para sem intercepto. Caso em branco, a regressão é computada com intercepto;
    Nivel_de_Significância: nível de significância do teste. Caso branco, o nível de significancia padrão é de 5%.
    Estatística: LM ou F
    '''

    #Fazendo a regressão e limpando a tela
    Regressao_Multipla(x,y,constante)
    clear_output()

    #Calculando o quadrado dos resíduos
    Res_Quad = Resultado.resid**2

    #Calculando o quadrado dos valores previstos
    Previstos = Lista_ychapeu
    Previstos2 = Previstos**2

    #Criando um dataframe pra armazenar esses valores
    dfy_y2 = pd.DataFrame({"y":Previstos,"y2":Previstos2})
    y_y2 = dfy_y2[['y','y2']]

    #Realizando o teste F ou LM de Res_Quad sobre y e y^2
    if Estatística == "LM":
        Teste_LM(y_y2, Res_Quad, y_y2, Nivel_de_Significância)
        print("Ho: O erro é homoscedástico")
    else:
        Teste_F(y_y2, Res_Quad, y_y2, Nivel_de_Significância)
        print("Ho: O erro é homoscedástico")

def RESET(x, y, constante = "S", robusta = "N", Nivel_de_Significância = 0.05):
    '''
    Função que executa um teste RESET para verificar a adequação das formas funcionais.
    Ho: o modelo está bem especificado.

    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    constante: "S" para regressão com intercepto e qualquer outro valor para sem intercepto. Caso em branco, a regressão é computada com intercepto;
    robusta: "N" para regressão com erros-padrão tradicionais e qualquer outro valor para erros-padrões robustos. Caso em branco, a regressão é computada com erros-padrão comuns;
    Nivel_de_Significância: nível de significância do teste. Caso branco, o nível de significancia padrão é de 5%.
    '''
    #Fazendo uma regressão múltipla e limpando a tela
    Regressao_Multipla(x, y, constante)
    clear_output()

    #Verificando o tipo da covariância selecionada
    if robusta == "N":
        tipo = 'nonrobust'
    else:
        tipo = 'HC1'

    Teste = sm.stats.diagnostic.linear_reset(Resultado, power = 2, use_f = False, cov_type = tipo)
    
    if Teste.pvalue < Nivel_de_Significância:
        print(f"""
        O p-valor do teste foi de {np.around(Teste.pvalue,6)}, menor que o nível de significância de {Nivel_de_Significância*100}%.\n
        Assim, rejeita-se Ho (o modelo está MAL especificado)."""
        )
    else:
        print(f"""
        O p-valor do teste foi de {np.around(Teste.pvalue,6)}, maior que o nível de significância de {Nivel_de_Significância*100}%.\n
        Assim, não se rejeita Ho (o modelo NÃO está MAL especificado)"""
        )

def Teste_J_Davidson_MacKinnon(x1,x2, y, constante = "S", robusta = "N", Nivel_de_Significância = 0.05):
    '''
    Função que executa um teste J para verificar qual o modelo mais adequado (dentre os dois colocados).
    Ho: o modelo 1 é preferível (ver o p-valor do último coeficiente).

    x1: lista ou array com os valores das variáveis independentes do primeiro modelo;
    x2: lista ou array com os valores das variáveis independentes do segundo modelo;
    y: lista ou array com os valores da variável dependente;
    constante: "S" para regressão com intercepto e qualquer outro valor para sem intercepto. Caso em branco, a regressão é computada com intercepto;
    robusta: "N" para regressão com erros-padrão tradicionais e qualquer outro valor para erros-padrões robustos. Caso em branco, a regressão é computada com erros-padrão comuns;
    Nivel_de_Significância: nível de significância do teste. Caso branco, o nível de significancia padrão é de 5%.
    '''
    
    #Fazendo a regressão do segundo modelo
    Regressao_Multipla(x2, y, constante, robusta)
    clear_output()

    #Criando um novo dataframe e adicionando os valores previstos do modelo 2 à x
    Valores_Previstos_2 = pd.DataFrame({'Previsão M1':Lista_ychapeu})
    x = pd.concat([x1, Valores_Previstos_2], axis=1, sort=False)

    #Fazendo a regressão do primeiro modelo sobre x
    Regressao_Multipla(x, y, constante, robusta)
    clear_output()

    #Pegando o p-valor do teste
    P_valor = Resultado.pvalues[-1]

    if P_valor < Nivel_de_Significância:
        print(f"""
        O p-valor do teste foi de {np.around(P_valor,6)}, menor que o nível de significância de {Nivel_de_Significância*100}%.\n
        Assim, rejeita-se Ho (ou seja, o modelo 2 ({list(x2)}) é mais bem especificado)."""
        )
    else:
        print(f"""
        O p-valor do teste foi de {np.around(P_valor,6)}, menor que o nível de significância de {Nivel_de_Significância*100}%.\n
        Assim, não se rejeita Ho (ou seja, o modelo 1 ({list(x1)}) é mais bem especificado)."""
        )

######### Funções de Dados em Painel #########
def Arrumar_Painel():
    '''
    Função que transforma o painel num formato que o PanelOLS consegue ler (index multinível e coluna do tipo categoria para os anos)
    '''
    global df

    # pedir a coluna com os indivíduos; se o nome for inválido, sair da função.
    coluna_individuos = str(input('Qual o rótulo da coluna de indivíduos/clusters?\n'))
    if coluna_individuos not in df.columns:
        print("Coluna de indivíduos não está no dataframe. Insira uma coluna válida e tente novamente!")
        return None
    
    # pedir a coluna com os períodos de tempo; se o valor for inválido, sair da função.
    coluna_tempo = str(input('Qual o rótulo da coluna de tempo/observações dos clusters?\n'))
    if coluna_tempo not in df.columns:
        print("Coluna de tempo não está no dataframe. Insira uma coluna válida e tente novamente!")
        return None

    ## arrumando o painel
    periodos = pd.Categorical(df[coluna_tempo])
    df = df.set_index([coluna_individuos,coluna_tempo])
    df[coluna_tempo] = periodos
    return df

def Reg_Painel_Primeiras_Diferenças (x,y, cov = "normal"):
    '''
    Função que calcula uma regressão de primeiras diferenças SEM um intercepto, sendo, por default, computada com erros padrões não robustos.
    Para calcular a regressão com um intercepto, ver o notebook "Cap 13 e 14".
    **IMPORTANTE: para o painel estar arrumado, os dados devem estar multi-indexados por indíviduo e por tempo, nesta ordem.
    Caso contrário, transformar o dataframe usando a função 'Arrumar Painel'
    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    cov: "normal" para regressão com erros-padrão tradicionais (caso padrão);
        "robust" para erros-padrões robustos.
        "cluster" para erros-padrões clusterizados
    '''
    global df, Resultado

    Modelo = FirstDifferenceOLS(y, x)

    if cov == "robust":
        Resultado = Modelo.fit(cov_type = 'robust')
    elif cov == 'kernel': ## correlação robusta à heteroscedasticidade e autocorrelação serial
        Resultado = Modelo.fit(cov_type = 'kernel')
    elif cov == 'clustered' or cov == 'cluster':
        Resultado = Modelo.fit(cov_type = 'clustered', cluster_entity = True)
    else:
        Resultado = Modelo.fit()

    print(Resultado)

def Reg_Painel_Efeitos_Fixos(x, y, constante = "S", cov='normal'):
    '''
    Função que calcula uma regressão de efeitos fixos, sendo, por default, computada com um intercepto e com erros padrões não robustos.
    **IMPORTANTE: para o painel estar arrumado, os dados devem estar multi-indexados por indíviduo e por tempo, nesta ordem.
    Caso contrário, transformar o dataframe usando a função 'Arrumar Painel'
    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    constante: "S" para regressão com intercepto e qualquer outro valor para sem intercepto. Caso em branco, a regressão é computada com intercepto;
    cov: "normal" para regressão com erros-padrão tradicionais (caso padrão);
        "robust" para erros-padrões robustos.
        "cluster" ou "clustered" para erros-padrões clusterizados
    '''
    global df, Resultado
    
    # formando o vetor de variáveis independentes
    if constante == "S":
        X = sm.add_constant(x)
    else:
        X = x
    
    #Criando o Modelo levando em conta a opção dos erros padrão
    Modelo = PanelOLS(y,X, entity_effects=True, drop_absorbed=True)

    if cov == "robust":
        Resultado = Modelo.fit(cov_type = 'robust')
    elif cov == 'kernel': ## correlação robusta à heteroscedasticidade e autocorrelação serial
        Resultado = Modelo.fit(cov_type = 'kernel')
    elif cov == 'clustered' or cov == 'cluster':
        Resultado = Modelo.fit(cov_type = 'clustered', cluster_entity = True)
    else:
        Resultado = Modelo.fit()

    print(Resultado)

def Reg_Painel_MQO_Agrupado(x, y, constante = "S", cov = "normal"):
    '''
    Função que calcula uma regressão por MQO agrupado, sendo, por default, computada com um intercepto e com erros padrões  robustos.
    **IMPORTANTE: para o painel estar arrumado, os dados devem estar multi-indexados por indíviduo e por tempo, nesta ordem.
    Caso contrário, transformar o dataframe usando a função 'Arrumar Painel'
    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    constante: "S" para regressão com intercepto e qualquer outro valor para sem intercepto. Caso em branco, a regressão é computada com intercepto;
    cov: "normal" para regressão com erros-padrão tradicionais (caso padrão);
        "robust" para erros-padrões robustos.
        "cluster" ou "clustered" para erros-padrões clusterizados
    '''
    global df, Resultado
    
    # formando o vetor de variáveis independentes
    if constante == "S":
        X = sm.add_constant(x)
    else:
        X = x
    
    #Criando o Modelo levando em conta a opção do erro padrão
    Modelo = PooledOLS(y,X)

    if cov == "robust":
        Resultado = Modelo.fit(cov_type = 'robust')
    elif cov == 'kernel': ## correlação robusta à heteroscedasticidade e autocorrelação serial
        Resultado = Modelo.fit(cov_type = 'kernel')
    elif cov == 'clustered' or cov == 'cluster':
        Resultado = Modelo.fit(cov_type = 'clustered', cluster_entity = True)
    else:
        Resultado = Modelo.fit()
    print(Resultado)

def Reg_Painel_Efeitos_Aleatórios(x, y, constante = "S", cov = "normal"):
    '''
    Função que calcula uma regressão de efeitos fixos, sendo, por default, computada com um intercepto e com erros padrões  robustos.
    **IMPORTANTE: para o painel estar arrumado, os dados devem estar multi-indexados por indíviduo e por tempo, nesta ordem.
    Caso contrário, transformar o dataframe usando a função 'Arrumar Painel'
    x: lista ou array com os valores das variáveis independentes;
    y: lista ou array com os valores da variável dependente;
    constante: "S" para regressão com intercepto e qualquer outro valor para sem intercepto. Caso em branco, a regressão é computada com intercepto;
    robusta: "N" para regressão com erros-padrão tradicionais e qualquer outro valor para erros-padrões robustos. Caso em branco, a regressão é computada com erros-padrão robustos.
    '''
    global df, Resultado
    
    # formando o vetor de variáveis independentes
    if constante == "S":
        X = sm.add_constant(x)
    else:
        X = x
    
    #Criando o Modelo
    Modelo = RandomEffects(y,X)
    if cov == "robust":
        Resultado = Modelo.fit(cov_type = 'robust')
    elif cov == 'kernel': ## correlação robusta à heteroscedasticidade e autocorrelação serial
        Resultado = Modelo.fit(cov_type = 'kernel')
    elif cov == 'clustered' or cov == 'cluster':
        Resultado = Modelo.fit(cov_type = 'clustered', cluster_entity = True)
    else:
        Resultado = Modelo.fit()
    print(Resultado)

def hausman_EF_EA(x_inef, y, Nivel_de_Significância = 0.05):
    '''
    Função que faz um teste de Hausman, em que H0: Não há correlação entre os efeitos não-observados e as variáveis explicativas
    x_inef: variáveis explicativas do modelo ineficiente sob H0 (EF);
    y: variável explicativa
    '''
    ## Fazendo a regressão de efeitos fixos e guardando o resultado
    Reg_Painel_Efeitos_Fixos(x,y)
    clear_output()
    fixed = Resultado

    ## Fazendo a regressão de efeitos aleatórios e guardando o resultado
    Reg_Painel_Efeitos_Aleatórios(x,y)
    clear_output()
    random = Resultado

    ## Calculando a estatística de Hausman
    # calculando a diferença entre os parametros e a variância assíntótica da diferença entre os parametros
    var_assin = fixed.cov - random.cov
    d = fixed.params - random.params
    
    # calculando H
    H = d.dot(np.linalg.inv(var_assin)).dot(d)
    # calculando os graus de liberdade
    gl = random.params.size -1
    # Calculando o P-valor do teste
    p = stats.chi2(gl).sf(H)

    if p < Nivel_de_Significância:
        print(f"O valor de H é {round(H,6)} com {gl} graus de liberdade na distribuição chi2. O p-valor do teste é {round(p,6)} e, portanto, se rejeita H0 e prefere-se o modelo de efeitos fixos.")
    else:
        print(f"O valor de H é {round(H,6)} com {gl} graus de liberdade na distribuição chi2. O p-valor do teste é {round(p,6)} e, portanto, não se rejeita H0 e prefere-se o modelo de efeitos aleatórios.")

def equation(sep_erros= "("):
    '''
    Função que gera uma equação formatada do word
    '''
    
    ## capturando os parametros e os erros
    params = dict(np.around(Resultado.params,3))
    ## linearmodels usa .std_erros para capturar os erros padrão, sm usa .bse
    try:
        std_errors = dict(np.around(Resultado.std_errors,5))
    except:
        std_errors = dict(np.around(Resultado.bse,5))

    ## fazendo o loop para pegar os coeficientes*nome das variáveis e os seus erros-padrão entre parênteses
    parametros = ""
    erros = ""
    for i in params.keys():
        # levando em conta a chave escolhida pelo usuário
        if sep_erros == "("
            erros += f" & ({std_errors[i]})"
        else:
            erros += f" & [{std_errors[i]}]"

        if i != 'const':
            if params[i] > 0:
                parametros += f" & + {params[i]}{i}"
            else:
                parametros += f" & - {-params[i]}{i}"
        else:
            parametros += f"{params[i]}"
    
    ## Fazendo a str que irá pro word (em forma de matriz)
    inicio = "\matrix{"
    fim = "}"

    # linearmodels usa model.dependente; sm usa model.endog_names
    try:
        y = Resultado.model.dependent.dataframe.columns[0]
    except:
        y = Resultado.model.endog_names
    word = f"{inicio}{y} & = & {parametros} \\\ & {erros}{fim}"

    ## Adicionando o numero de obs e os r2
    # linearmodels usa .entity_info['total'] no numero de observações, sm usa nobs
    try:
        word += f"\nn = {int(dict(Resultado.entity_info)['total'])}; R^2 = {np.around(Resultado.rsquared,3)}"
    except:
         word += f"\nn = {int(Resultado.nobs)}; R^2 = {np.around(Resultado.rsquared,3)}; R\\bar^2 = {np.around(Resultado.rsquared_adj,3)}"

    ## substituindo os . por ,
    word = word.replace(".",",")
    
    ## copiando para o clipboard e printando o sucesso
    subprocess.run("pbcopy", universal_newlines=True, input=word)
    print("O código da equação foi copiado para o clipboard!")