

 
import pandas as pd 
arquivo = pd.read_csv("wine_dataset.csv")
arquivo.head()
#converter o campo style em valores numéricos 
arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

#separando as variaveis entre preditoras e alvo
y = arquivo ['style'] # variável alvo
x = arquivo.drop ('style', axis =1) # x recebe todo o resto com excepcao da coluna stile
 
from sklearn.model_selection import train_test_split
#criar os conjuntos de dados de treino e teste
x_treino, x_teste, y_treino, y_test = train_test_split(x, y, test_size = 0.3) # 30% dos dados para teste

arquivo.shape # conjunto de dados completo
from sklearn.ensemble import  ExtraTreesClassifier
#Criando o modelo 
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino) # aplicar o algorítmo nos dados 

resultado = modelo.score(x_teste, y_test)
print("Acuracia:", resultado) 

print(y_test[400:403]) 
print(x_teste[400:403])

#saber se o modelo consegue prever a class das 3 amostras 
previsoes = modelo. predict(x_teste[400:403])
print(previsoes)
