
# Treinamento de redes neurais com transfer learning

📌 Neste projeto uso a técnica de Transfer Learning para classificação multilabel de atributos faciais com a rede ResNet50.

A solução está neste diretório, no notebook do Google Colab [`Treinamento_de_redes_neurais_com_transfer_learning.ipynb`](https://github.com/FlaviaLopes/dio-challenges-coding-the-future-with-baires-dev/blob/main/projeto_1/Treinamento_de_redes_neurais_com_transfer_learning.ipynb).

$Mas$ $antes$ $disso$... Entenda um pouco mais sobre o desafio.

> **Desafio:** 
"O projeto consiste em aplicar o método de Transfer Learning em uma rede de Deep Learning usando a linguagem Python e no ambiente COLAB. Neste projeto, você pode usar sua própria base de dados (exemplo: fotos suas, dos seus pais, dos seus amigos, dos seus animais domésticos, etc). O exemplo de gatos e cachorros pode ser substituído por duas outras classes do seu interesse. O Dataset criado em nosso projeto anterior pode ser utilizado agora."

🎯 **Etapas**
- Para atingir os objetivos propostos, as seguintes etapas foram realizadas:  
  1. **Escolha da base e preparação dos dados:** escolhi o dataset CelebA (ver seção 1).  
  2. **Implementação de Transfer Learning com a rede ResNet50:** retreinei a rede com as imagens do dataset selecionado (ver seção 2).  
  3. **Treinamento e validação do modelo com métricas específicas:** utilizei as métricas AUC, recall e precision (ver seção 3).

📝 **Contexto**: A seção 1 dá mais detalhes sobre a base utilizada, as seções 2 a 4 trazem mais informações sobre a rede utilizada, métricas de avaliação e o processo de transfer learning.
__Obs.:__ o enunciado propõe o uso de duas classes, e o CelebA é um dataset de faces multilabel com 40 atributos. Ou seja, uma imagem pode ter até 40 labels simultaneamente. Por preferência pessoal resolvi explorar esse desafio, já que o objetivo desta atividade é aplicar as técnicas de transfer learning, não necessariamente apresentar um modelo excelente. Obviamente, devido ao desbalanceamento desse dataset em torno de alguns labels, já era esperado que o modelo não performasse bem para algumas classes (ver notebook `Feature Engineering Celeb A.ipynb`). Entretanto, com as métricas de desempenho deste primeiro protótipo, eu pude explorar melhor o problema no projeto_3, e construir uma estratégia para data augmentation de algumas classes, o que pode melhorar o desempenho do modelo em uma segunda iteração de desenvolvimento.

---
🔍 **Desempenho do modelo**
- Input data:
	- Foram selecionadas 3.800 imagens para o treinamento, e 38 classes das 40 disponíveis (ver `Feature Engineering Celeb A.ipynb`). A imagem a seguir ilustra a distribuição das classes do dataset, a terceira imagem é a distribuição da amostra utilizada para o treino.
![classes distribution](https://raw.githubusercontent.com/FlaviaLopes/dio-challenges-coding-the-future-with-baires-dev/refs/heads/main/projeto_1/output/images/dataset_classes_distribution.png)
- Training phase:
	- A cabeça da rede ResNet50 foi treinada por 8 épocas (Early Stop foi ativado).
- Fine tunning phase:
	- As primeiras 10 camadas continuaram congeladas, e as demais foram descongeladas para que o treinamento continuasse, a uma taxa de aprendizado menor, por mais 20 épocas.
- A imagem a seguir resume a evolução do treinamento através das métricas: AUC, Precision, Recall e loss (binary cross entropy). O traço vertical indica o fim da fase de treinamento e início do fine tunning, onde as métricas de precisão e AUC caem, naturalmente, mas depois começam a evoluir. As métricas são uma média do desempenho das classes, cerca de 3 classes desempenharam bem, mais outras 3 desempenharam razoavelmente. As métricas deste modelo são discutidas no projeto_3.

![Training metrics](https://raw.githubusercontent.com/FlaviaLopes/dio-challenges-coding-the-future-with-baires-dev/refs/heads/main/projeto_1/output/images/ResNet50_CelebA_training_finetune_history.png)

🔍 **Como o modelo classificou...**

- A seguir, temos quatro fotos de pessoas preditas pelo modelo: duas celebridades brasileiras, Lena e eu.
- Blond_Hair parece ser um problema, todas as imagens foram rotuladas assim. Heavy_Makeup também. Blond_Hair está fortemente correlacionada a Heavy_Makeup, Wavy_Hair, Oval_Face, Rosy_Cheeks e Point_Nose.
- Como discutido no notebook `Feature Engineering CelebA`, devido a alta correlação entre Young e Gray_Hair, temos que o modelo tende a classificar mulheres como jovens, provavelmente porque há poucas mulheres grisalhas na base, independentemente da idade delas. 

![Image grid](https://raw.githubusercontent.com/FlaviaLopes/dio-challenges-coding-the-future-with-baires-dev/refs/heads/main/projeto_1/output/images/images_grid_with_predictions.png)

---

## 📸 1. CelebA dataset
---
O **CelebA** é um dataset de imagens de rostos de celebridades, desenvolvido por pesquisadores da **The Chinese University of Hong Kong**. Contém mais de 202 mil imagens rotuladas com 40 atributos faciais binários, como _smiling_, _young_, _black-hair_, _brown-hair_, _blond-hair_, _big-lips_, dentre outros. É uma base bastante útil para treinamento de modelos de reconhecimento de atributos faciais.

- **Dataset obtido em**: [CelebA no Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)  
- **Mais informações em**: [MMLAB](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


## 🧠 2. ResNet50: deep leaning network
---
A **ResNet50** (Residual Neural Network com 50 camadas) é uma arquitetura de rede neural profunda introduzida em 2015 pela [Microsoft Research](https://en.innovatiana.com/post/discover-resnet-50). 

A ResNet50 foi treinada no dataset ImageNet, que contém milhões de imagens distribuídas em 1.000 categorias. Nesse contexto, a ResNet50 alcançou uma acurácia de top-5 de 92,1%, demonstrando sua eficácia em tarefas de classificação de imagens. Embora existam arquiteturas mais recentes que superam a ResNet50 em termos de acurácia, como a ResNet152V2 e as redes da família EfficientNet, a ResNet50 oferece um bom equilíbrio entre desempenho e eficiência computacional [Keras Applications](https://keras.io/api/applications/).  

A ResNet50 é uma boa escolha em tarefas de reconhecimento facial devido à sua capacidade de capturar características faciais complexas.   

## 📊 3. Evaluation metrics
---
A métrica **AUC** (Área sob a Curva ROC) avalia a capacidade do modelo de distinguir entre classes positivas e negativas, medindo a área sob a curva ROC (Receiver Operating Characteristic).  

A métrica AUC considera tanto a taxa de verdadeiros positivos quanto a de falsos positivos, equilibrando a avaliação do modelo em cenários onde há diferenças significativas na distribuição das classes.  

Neste modelo, a AUC multilabel foi utilizada para monitorar a evolução do treino. A função de perda foi Bynary Cross Entropy.


## 🚀 4. Entenda o processo de Transfer Learning
---
Transfer Learning é uma técnica de aprendizado onde um modelo treinado em uma base de dados é adaptado para outra tarefa semelhante, evitando o treinamento do zero e otimizando tempo e recursos.  

**Por que é útil?**  

- **Aproveitamento de conhecimento prévio:** Permite alavancar conhecimento aprendido em bases grandes, como o ImageNet.  
- **Redução de overfitting:** Especialmente importante em bases menores, onde um modelo treinado do zero pode superajustar os dados de treinamento.  
- **Eficiência computacional:** Economiza recursos ao reutilizar pesos previamente treinados.  

**Etapas de Transfer Learning neste projeto:**  

1. **Escolha do modelo base:** a rede ResNet50 pré-treinada no ImageNet foi a escolha para este projeto. 
2. **Congelamento de camadas:** As primeiras camadas da ResNet50 foram congeladas, preservando os pesos que capturam características gerais das imagens, como bordas e texturas. 
3. **Adaptação do modelo:** a última camada foi adaptada para classificar os 38 atributos do CelebA.  
4. **Seleção e preparação dos dados**: 3..800 imagens foram escolhidas para o treinamento, de maneira que diminuisse o desbalanceamento entre as classes. O data augmentation foi feito de maneira geral nesta primeira iteração.
4. **Treinamento personalizado:** as novas camadas adicionadas foram treinadas com o dataset permitindo que o modelo aprendesse características específicas do CelebA por 8 épocas. Depois, durante o fine tunning a rede foi treinada por mais 20 épocas.

## 5. Conclusão

O projeto atingiu o objetivo geral, que foi a escolha de uma rede e dataset para aplicação de transfer learning. Esta solução também teve como adicional a exploração dos desafios de uma classificação multilabel. 

Depois do modelo treinado, a etapa de testes produziu um dataframe com os valores de y e ŷ. Esse dataframe será carregado no Projeto 3 deste repositório, onde será explorado o cálculo de métricas de avaliação de modelos de classificação.

Por agora é isso. Obrigada pela atenção e até a próxima.

---

**Referências:**  
- [Artigo Original ResNet](https://arxiv.org/abs/1512.03385)  
- [ImageNet Benchmark](https://paperswithcode.com/sota/image-classification-on-imagenet)  
- [AUC: Receiver Operating Characteristic](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)  
- [Comparação de Métricas: F1, AUC, Accuracy](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc)  

