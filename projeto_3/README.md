# Cálculo de métricas de avaliação de aprendizado

📌 O resultado das predições do modelo ResNet50 com transfer learning, treinado no conjunto CelebA no projeto_1, foi avaliado a partir das principais métricas de classificação. 

A solução está neste diretório, no notebook do Google Colab [`Métricas de validação para modelos de ML.ipynb`](https://github.com/FlaviaLopes/dio-challenges-coding-the-future-with-baires-dev/blob/main/projeto_3/Métricas de validação para modelos de ML.ipynb).

$Mas$ $antes$ $disso$... Entenda um pouco mais sobre o desafio.

> **Desafio:** 
"Neste projeto, vamos calcular as principais métricas para avaliação de modelos de classificação de dados, como acurácia, sensibilidade (recall), especificidade, precisão e F-score. Para que seja possível implementar estas funções, você deve utilizar os métodos e suas fórmulas correspondentes (Tabela 1). 
>
>| Método           | Fórmula           |                 										                        |
|------------------|-------------------|
| Sensibilidade    | TP / (TP + FN)    |
| Especificidade   | TN / (FP + TN)    | 
| Acurácia	   | (TP + TN) / N     |
| Precisão	   | TP / (TP + FP)    |
| F-Score	   | 2 x (PxS) / (P+S) |
>
>Para a leitura dos valores de VP, VN, FP e FN, será necessário escolher uma matriz de confusão para a base dos cálculos. Essa matriz você pode escolher de forma arbitraria, pois nosso objetivo é entender como funciona cada métrica."

🎯 **Etapas**
- Para atingir os objetivos propostos, as seguintes etapas foram realizadas:  
  1. **Escolha da base e preparação dos dados:** escolhi o dataset CelebA (ver seção 1).  
  2. **Implementação de Transfer Learning com a rede ResNet50:** retreinei a rede com as imagens do dataset selecionado (ver seção 2).  
  3. **Treinamento e validação do modelo com métricas específicas:** utilizei as métricas AUC, recall e precision (ver seção 3).


