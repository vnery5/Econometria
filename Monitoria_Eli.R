print("Para rodar o Script, apertar Ctrl + Enter na Linha")

## para comentar, use #!

## Além disso, o assignment de variáveis se dá por <-;
## rode primeiro a linha da variável!
nome <- "Vinícius"
idade <- 18
print(paste0(
  paste0("Meu nome é ",nome),
  paste0(" e minha idade é ", idade)
  )
)

## Há dois objetos que guardam dados:
# Matrizes e DataFrames (subtipo de matriz)

#podemos criar um vetor para popular uma matriz
vetor1 <- c(1,2,3,4,5,6,7,8,9,10)
  #sequência de 10 a 100, pulando de cinco em cinco
  #em python, similar a range(10,100,5)
vetor2 <- seq(10,100,5) 

# criando uma matriz (lembrar do help!)
matriz1 <- matrix(vetor1,5,2)
# para exibir a matriz, selecioniar 'matriz1' e Cmd + Enter

# para exibir o elemento i,j da matriz
matriz1[1,1]

#para exibir a primeira coluna
matriz1[,1]

#criando um df a partir da matriz1
df <- data.frame(matriz1)

# para editar o df estilo excel
edit(df)

# para alterar o nome das colunas
colnames(df) <- c("1 a 5","6 a 10")

## criando novas colunas
# $ acessa as colunas de um df
df['11 a 15'] <- seq(11,15,1)
df['soma'] <- df$`1 a 5` + df$`6 a 10` + df$`11 a 15`

# para apagar (remover) uma variável
rm(vetor2)

# para listar todas as variáveis
ls()

#para apagar todas as variáveis
rm(list=ls())

################## Questão 2 Lista Econo ##############
## importando as bibliotecas necessárias
library(tidyverse)
library(haven)
wage2 <- read_dta("/Users/vinicius/Google Drive/UnB/Econometria/datasets/WAGE2.DTA")

## alternativamente, load(file.choose())
# permite selecionar bases de dados do R
load(file.choose())
wage2 <- data
wage2.desc <- desc #adicionado a descrição das colunas

# alternativamente, podemos carregar o pacote do wooldridge
rm(list=ls())
library(wooldridge)
df <- wage2

#mostrando apenas os primeiros/últimos resultados
head(df)
tail(df)

#mostrando os nomes das colunas
names(df)

#acessando uma coluna específica
df$KWW

# descrevendo uma coluna
summary(df$educ)

#mostrando o total de aparições de cada valor
table(df$educ) #393 pessoas tem 12 anos de educação

## FAZENDO A REGRESSÃO
reg_1 <- lm(data=df, formula=lwage~educ+exper+tenure+married+south+urban+black+KWW)
summary(reg_1)
reg_1

# Fazendo um Teste F
reg_2 <- lm(data=df, formula=lwage~educ+exper+tenure+married+south+urban+black+KWW+IQ)
reg_3 <- lm(data=df, formula=lwage~educ+exper+tenure+married+south+urban+black)
anova(reg_3,reg_2)
