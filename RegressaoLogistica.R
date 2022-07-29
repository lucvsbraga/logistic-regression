# Regressao Logistica
# Previsao e Deteccao de Risco de Credito

# Definindo o diretorio de trabalho
getwd()
setwd("/home/lucas/Documentos/Python/Machine Learning/logistic-regression")

# Carregando os pacotes
library(caret)
library(ROCR) 
library(e1071) 

# Carregando o dataset 
credit.df <- read.csv("Data/credit_dataset_final.csv", header = TRUE, sep = ",")
head(credit.df)
summary(credit.df)


## Pre processamento

# Transformando variaveis em fatores

to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

# Normalizacao
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}

# Normalizando as variaveis
numeric.vars <- c("credit.duration.months", "age", "credit.amount")
credit.df <- scale.features(credit.df, numeric.vars)

# Variaveis do tipo fator
categorical.vars <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                      'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                      'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                      'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
                      'dependents', 'telephone', 'foreign.worker')

# Aplicando as conversoes ao dataset
credit.df < to.factors(df = credit.df, variables = categorical.vars)
head(credit.df)
summary(credit.df)

# Preparando os dados de treino e de teste
indexes <- sample(1:nrow(credit.df), size = 0.6 * nrow(credit.df))
train.data <- credit.df[indexes,]
test.data <- credit.df[-indexes,]

# Separando os atributos e as classes
test.feature.vars <- test.data[, -1]
test.class.var <- test.data[,1]

# Construindo o modelo de regressao logistica
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
help(glm)
modelo_v1 <- glm(formula = formula.init, data = train.data, family = "binomial")

# Visualizando os detalhes do modelo
summary(modelo_v1)

# Fazendo previsoes e analisando o resultado
previsoes <- predict(modelo_v1, test.data, type = "response")
previsoes <- round(previsoes)

# Confusion Matrix
confusionMatrix(data = previsoes, reference = test.class.var, positive = '1')

# Feature Selection
formula <- "credit.rating ~ ."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
model <- train(formula, data = train.data, method = "glm", trControl = control)
importance <- varImp(model, scale = FALSE)

# Plot
plot(importance)

# Construindo um novo modelo com as variaveis selecionadas
formula.new <- "credit.rating ~ account.balance + credit.purpose + previous.credit.payment.status + savings + credit.duration.months"
formula.new <- as.formula(formula.new)
modelo_v2 <- glm(formula = formula.new, data = train.data, family = "binomial")

# Visualizando o novo modelo
summary(modelo_v2)

# Prevendo e Avaliando o modelo 
previsoes_new <- predict(modelo_v2, test.data, type = "response") 
previsoes_new <- round(previsoes_new)

# Confusion Matrix
confusionMatrix(data = previsoes_new, reference = test.class.var, positive = '1')


# Avaliando a performance do modelo

# Plot do modelo com melhor acuracia
modelo_final <- modelo_v2
previsoes <- predict(modelo_final, test.feature.vars, type = "response")
previsoes_finais <- prediction(previsoes, test.class.var)

# Função para Plot ROC 
plot.roc.curve <- function(predictions, title.text){
  perf <- performance(predictions, "tpr", "fpr")
  plot(perf,col = "black",lty = 1, lwd = 2,
       main = title.text, cex.main = 0.6, cex.lab = 0.8,xaxs = "i", yaxs = "i")
  abline(0,1, col = "red")
  auc <- performance(predictions,"auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4,legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")
  
}

# Plot
par(mfrow = c(1, 2))
plot.roc.curve(previsoes_finais, title.text = "Curva ROC")
