#### Leitura da PNADC - R

## Pacotes
library(tidylog)
library(magrittr)
library(readstata13)  # lê extensões.dta
# Caso não tenha instalado:
# install.packages("readstata13")

#### Lendo o arquivo dta ####
## mudando o diretório (caso seja necessário)
getwd()
setwd("/Users/vinicius/Google Drive/UnB/4º Semestre/Econometria")

## Lendo a base de dados (demora um bom bucadinho)
pnadc <- read.dta13("PNAD_painel_6_rs.dta")

## Selecionando apenas as observações do CENTRO-OESTE
lCO <- c("Mato Grosso do Sul", "Mato Grosso", "Goiás", "Distrito Federal")
lCO_ids <- c(50, 51, 52, 53)

# De 2,4gb vai para 241mb :)
pnadc <- pnadc[pnadc$UF %in% lCO_ids, ]

## Selecionando apenas algumas colunas
# ATENÇÃO: MERAMENTE ILUSTRATIVO! OLHEM QUAIS COLUNAS VOCÊS VÃO PRECISAR NO DICIONÁRIO!
lColunas <- c("Ano","Trimestre","UF","UPA","Estrato","V1027","V1028","V1029","posest",
              "V1008","V1014","V1022","V2001","V2003","V2005","V2007","V2009","V2010",
              "VD2002","VD2003","VD3004","VD3005","VD3006","VD4001","VD4002","VD4003",
              "VD4005","VD4008","VD4009","VD4010","VD4016","VD4017","VD4019",
              "VD4020","VD4031","VD4035","VD4036","VD4037")
# De 240mb vai para 43,7mb: reduzimos em 98% desde o tamanho da base original :)
pnadc <- pnadc[, lColunas]

## Gerando um novo arquivo dta com a base nova
save.dta13(pnadc, file="pnadc.dta")

# Mais informações: https://cran.r-project.org/web/packages/readstata13/readstata13.pdf

#### Diretamente do IBGE ####
## O IBGE possui um pacote que lê os arquivos .txt disponibilizados em seu FTP
## Há duas opções: baixar o arquivo e ler localmente ou ler direto do site do IBGE,
## sendo que a segunda opção precisa de uma opção bem rápida

## Como vocês vão trabalhar com um painel específico, vocês teriam que baixar
## várias PNADCs trimestrais e filtrar apenas para o seu painel (var. V1014),
## o que dá bem mais trabalho

## Se alguém tiver interesse, seguem links legais:
# https://rpubs.com/BragaDouglas/335574
# https://cran.r-project.org/web/packages/PNADcIBGE/PNADcIBGE.pdf
