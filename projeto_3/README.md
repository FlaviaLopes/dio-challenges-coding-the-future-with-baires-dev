# Cálculo de métricas de avaliação de aprendizado

📌 O resultado das predições do modelo ResNet50 com transfer learning, treinado no conjunto CelebA no projeto_1, foi avaliado a partir das principais métricas de classificação. 

A solução está neste diretório, no notebook do Google Colab [`Métricas de validação para modelos de ML.ipynb`](https://github.com/FlaviaLopes/dio-challenges-coding-the-future-with-baires-dev/blob/main/projeto_3/Métricas de validação para modelos de ML.ipynb).

$Mas$ $antes$ $disso$... Entenda um pouco mais sobre o desafio.

> **Desafio:** 
"Neste projeto, vamos calcular as principais métricas para avaliação de modelos de classificação de dados, como acurácia, sensibilidade (recall), especificidade, precisão e F1-score. Para que seja possível implementar estas funções, você deve utilizar os métodos e suas fórmulas correspondentes (Tabela 1). 
>Para a leitura dos valores de VP, VN, FP e FN, será necessário escolher uma matriz de confusão para a base dos cálculos. Essa matriz você pode escolher de forma arbitraria, pois nosso objetivo é entender como funciona cada métrica."

| Método           | Fórmula                  |                 										                        
|--------------------|---------------------|
| Sensibilidade     | TP / (TP + FN)     |
| Especificidade   | TN / (FP + TN)    | 
| Acurácia	          | (TP + TN) / N      |
| Precisão	          | TP / (TP + FP)     |
| F-Score	              | 2 x (PxS) / (P+S) |


🎯 **Etapas**
- Para atingir os objetivos propostos, as seguintes etapas foram realizadas:  
  1. **Escolha da matriz confusão:** a matriz é produto do projeto 1, onde uma rede de classificação multilabel foi treinada com transfer learning usando o dataset CelebA (ver Projeto 1).  
  2. **Implementação das funções para cálculo das métricas:** o cálculo das métricas foi demonstrado a partir de funções que leêm a matriz confusão multilabel.  
  3. **Uso de funções prontas de bibliotecas para demonstração:** utilizei algumas das funções de cálculo de métricas de bibliotecas como sklearn para comparação.
  4. **Interpretação das métricas**: a partir das métricas o modelo é interpretado e intervenções são sugeridas.

📝 **Contexto**: A seção 1, "Entendendo o cálculo de métricas de modelos de classificação", dá mais detalhes sobre as métricas, como são calculadas e relevância para avaliação dos modelos.

---
🔍 **Interpretação do modelo**

__Qual o foco: diminuir FP ou FN?__ 
Neste cenário, o falso positivo é pior que o falso negativo. Melhorias em todas as métricas importam, mas principalmente em precisão e especificidade, pois quanto menos FP, mais próximo de 1 serão essas métricas.

__Por quais atributos começar?__ 
Podemos usar como referência a pesquisa que publicou este dataset, que já fez uma análise exploratória e trouxe insights importantes sobre os atributos da base: [Deep Learning Face Attributes in the Wild](https://arxiv.org/pdf/1411.7766).

Os autores clusterizaram os pesos da rede para compreender melhor como os atributos estavam se relacionando. E descobriram grupos com relação de co-ocorrrência forte, e outros cujos atributos tinham em comum as cores ou texturas.

>"As shown in Fig.11, Group #1, Group #2 and Group #4 demonstrate co-occurrence relationship between attributes, e.g. ‘Attractive’ and ‘Heavy Makeup’ have high correlation. Attributes in Group #3 share similar color descriptors, while attributes in Group #6 correspond to certain texture and appearance traits."

De acordo com essas informações e análise exploratória realizada no projeto 1, parece mais razoável começar otimizando o modelo a partir dos atributos com forte co-ocorrência. Alguns destes, inclusive, são as classes com maior suporte. Uma melhoria nas métricas dessas classes pode levar o modelo a performar melhor.

Nas iterações seguintes, a questão dos atributos com cores e texturas semelhantes, e que não performaram bem após a primeira otimização pode ser reavaliada.

__Plano para a próxima iteração__ 

- Focar nos grupos 1, 2 e 4, com forte co-ocorrência. Preferencialmente os atributos de maior suporte.
- Para cada atributo, ranqueado pelo suporte:
	- Avaliar a matriz de confusão da classe, e se houver FP significante:
		- Inspecionar algumas imagens com FP
		- Para as imagens com FP nesta classe, verificar quais outras classes ocorrem.
		- algumas perguntas: os atributos associados a esta classe diferem dos que ocorrem no grupo de FP? pode trazer uma direção mais assertiva para o data augmentation, focando nos exemplos que a rede teve dificuldade de aprender.
		
__Grupos semânticos__: atributos dos grupos 1, 2 e 4, ordenados por suporte

- atributos considerados:
	- acima de 50% de suporte, em relação a classe maioritária
	- atributos dos grupos 1, 2 e 4
- atributos selecionados para primeira otimização: No_Beard, Young, Smiling, Mouth_Slightly_Open, High_Cheekbones, Wearing_Lipstick
- em seguida os demais dos grupos 1, 2 e 4 podem ser avaliados.

__Conclusão__

O próximo passo é compreender os falsos positivos. Por exemplo: 
- quais classes ocorrem simultaneamente com as imagens classificadas erroneamente como No_Beard? 
- alguma característica pode estar confundindo o modelo? Esta característica ruidosa tem baixo suporte?
Estas respostas podem indicar a abordagem: oversampling ou undersampling focando em qual grupo de imagens?

---

## Entendendo o cálculo de métricas de modelos de classificação
---
Para um problema com `n` classes, analisamos os arrays `y_true` (rótulos verdadeiros) e `y_pred` (rótulos preditos) para entender o desempenho do modelo em relação às classes {C1, C2, ..., Cn}. Calculamos as seguintes ocorrências:
- Verdadeiros Positivos (TP): Quantas vezes a imagem pertencia à classe 
Ci e foi corretamente classificada como Ci. 
- Falsos Negativos (FN): Quantas vezes a imagem pertencia à classe 
Ci, mas foi erroneamente classificada como outra classe.
- Falsos Positivos (FP): Quantas vezes a imagem não pertencia à classe 
Ci, mas foi incorretamente classificada como Ci.
- Verdadeiros Negativos (TN): Quantas vezes a imagem não pertencia à classe 
Ci e foi corretamente classificada como pertencendo a outra classe.

Essa análise é fundamental para avaliar o desempenho do modelo em classificar corretamente as classes.

### Cálculando indicadores a partir da matriz confusão
---
A partir da matriz confusão de cada classe é possível calcular métricas que resumem o desempenho do modelo por classe. As principais métricas são: i) precisão; ii) recall; iii) especificidade; iv) acurácia.

1. __Precision__: A precision mede a proporção de verdadeiros positivos (TP) em relação ao total de positivos previstos pelo modelo (TP + FP). Avalia a confiabilidade das previsões positivas do modelo, minimizando falsos positivos. _Fórmula: `TP / (TP + FP)`_.

2. __Recall__: O recall mede a proporção de verdadeiros positivos (TP) corretamente identificados em relação ao total de positivos reais. Indica o quão bem o modelo consegue encontrar os exemplos da classe positiva, evitando falsos negativos. _Fórmula: `TP / (TP + FN)`_

3. __Specificity__: A especificidade mede a proporção de verdadeiros negativos (TN) corretamente identificados em relação ao total de negativos reais. Quanto maior é a especificidade, melhor o modelo é em identificar corretamente as instâncias da classe negativa (ou seja, evitar falsos positivos).
_Fórmula: `TN + (TN / FP)`_

4. __Accuracy__: A acurácia mede a proporção de predições corretas em relação ao total de amostras, considerando todas as classes. Para problemas multilabel, a acurácia da classe pode ser calculada como: `(TP + TN) / (TP + FP + FN + TP)`. E a acurácia geral é o somatório das acurácias de cada classe, dividido por n.


__Recall e Specificity (balanceamento entre classes)__
O recall foca nos positivos reais (sensibilidade), enquanto a specificity foca nos negativos reais. Essas duas métricas frequentemente estão em conflito. Por exemplo: se focarmos em aumentar o recall, o modelo pode acabar classificando mais negativos como positivos, reduzindo a specificity. Um modelo balanceado deve maximizar ambos para evitar tanto falsos negativos quanto falsos positivos.

__Precision e Specificity (previsões positivas e negativas)__
- A métrica precision avalia a qualidade das previsões positivas, enquanto a specificity avalia a qualidade das previsões negativas. 

- Um modelo com alta precision, mas baixa specificity, pode prever corretamente os positivos, mas cometer muitos erros ao classificar negativos como positivos (FP).

- A métrica specificity é essencial em contextos onde falsos positivos têm consequências graves:
  - na detecção de fraudes, um FP pode bloquear uma transação legítima.
  - em exames de saúde, um FP pode levar a exames desnecessários

- Por outro lado, em alguns cenários o FN é muito mais grave:
  - detecção de câncer
  - ataques cibernéticos

- Então, se a classe A possui alta precisão, significa que o modelo performa com uma baixa taxa de falsos positivos, e se o recall também é alto, significa que o modelo performa com uma baixa taxa de falsos negativos. Quanto maior a precisão e o recall, melhor o modelo é em reconhecer a presença ou ausência da classe. Especificidade e sensibilidade equilibradas, são resultado de um modelo balanceado. 

- _A maximização de uma métrica em detrimento de outra vai depender do cenário de aplicação._

---
### Combinando métricas
---
__AUC__: especificidade e recall

A curva AUC mede a capacidade do modelo de distinguir entre classes. Para problemas multilabel, a curva AUC é calculada como a média das áreas sob as curvas ROC de cada classe individual:
- A curva ROC (Receiver Operating Characteristic) para cada classe é gerada traçando o Recall (ou Taxa de Verdadeiros Positivos) contra o 1 - Specificity (ou Taxa de Falsos Positivos).
- Um valor de AUC próximo de 1 indica que o modelo é muito bom em separar positivos de negativos, enquanto um valor próximo de 0,5 indica um desempenho semelhante ao acaso.

_Fórmula do AUCi (Área Sob a Curva ROC para uma classe individual)_
$$
\text{AUC}_i = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})
$$

$$
\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
$$

_Aproximação do AUCi pela soma dos trapézios_
$$
\text{AUC}_i \approx \sum_{j=1}^{n-1} \left( \text{FPR}_{j+1} - \text{FPR}_j \right) \cdot \frac{\text{TPR}_j + \text{TPR}_{j+1}}{2}
$$

_AUC médio para problemas multilabel_
$$
\text{AUC}_{\text{multi}} = \frac{1}{n} \sum_{i=1}^{n} \text{AUC}_i
$$


__F1-Score__: precisão e recall

O F1-score combina precision e recall, sendo particularmente útil para lidar com desequilíbrios entre as classes. __Fórmula__: `2 x (Precision + Recall) / (Precision x Recall)`.
- Para problemas multilabel, o F1-score pode ser calculado de duas formas:
  - Macro: Calcula o F1-score para cada classe individualmente e depois tira a média, tratando todas as classes igualmente.
  - Micro: Agrega os valores de TP, FP, e FN de todas as classes antes de calcular o F1-score, dando maior peso às classes mais frequentes.


---

