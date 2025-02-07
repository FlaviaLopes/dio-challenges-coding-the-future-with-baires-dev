# Criando um sistema de reconhecimento facial do zero

A solução está neste diretório, no notebook do Google Colab [`name`](https://github.com/FlaviaLopes/dio-challenges-coding-the-future-with-baires-dev/blob/main/projeto_5/name.ipynb).

$Mas$ $antes$ $disso$... Entenda um pouco mais sobre o desafio.

> **Desafio:** 
"__Criando um sistema de reconhecimento facial do zero__: o objetivo principal deste projeto é trabalhar com as bibliotecas e frameworks estudados e analisados em nossas aulas. Neste sentido, a proposta padrão envolve um sistema de detecção e reconhecimento de faces, utilizando o framework TensorFlow em conjuntos com as bibliotecas que o projetista julgue necessárias, de forma ilimitada. Por meio da Figura 1 é possível visualizar o resultado esperado para o modelo proposto, devendo detectar e reconhecer mais de uma face ao mesmo tempo. Para isso você deve: i) utilizar uma rede de detecção treinada para detectar faces; ii) utilizar uma rede de classificação para classificar a face detectada. 
>
>Para realizar este projeto, você pode utilizar os seguintes trabalhos de referência: 
>[Detecção Facial](https://colab.research.google.com/drive/1QnC7lV7oVFk5OZCm75fqbLAfD9qBy9bw?usp=sharing)
>[Detecção e classificação de objetos](https://colab.research.google.com/drive/1xdjyBiY75MAVRSjgmiqI7pbRLn58VrbE?usp=sharing )

Neste tipo de tarefa duas redes podem ser empregadas, uma de detecção (FaceNet, Yolo Face, MTCNN, etc) e outra de classificação (ResNet, etc). Entretanto, escolhi utilizar a rede YOLO, que tanto detecta objetos quanto os classifica. Essa escolha resulta na necessidade de um dataset maior para treinamento, o ideal para detecção em tempo real seria entre 150 a 200 bons exemplos.

Neste projeto, eu coletei e anotei cerca de 50-70 imagens de cada personagem. Mesmo com data augmentation essa quantidade é pequena para o propósito final. Ainda assim, o resultado é promissor.
Com os ajustes na base de dados e retreino, a rede por aumentar muito o desempenho.

Então, o objetivo é __detectar as faces da Arya, Sansa e Jonh__ em cenas de Game of Thrones. Para alcançar esse objetivo, eu utilizo a rede YOLO v9 nano para treinar um modelo de detecção e classificação facial específico.


🎯 **Etapas**
- Para atingir os objetivos propostos, as seguintes etapas foram realizadas:  
  1. **1 preparing the data:** são três classes: Arya Stark, Sansa Stark e John Snow
cerca de 50-70 imagens de cada classe são coletadas e rotuladas com o software Roboflow.
link do meu dataset é público em: https://universe.roboflow.com/brothers-onqjy/yolo-face-detector-got/dataset/8
  2. **2 codding:** duas classes são escritas: 
  	- i) DatasetManager, que lê e processa os metadados, divide as imagens em conjuntos equilibrados de treino, teste e validação, e as organiza em diretórios;
	- ii) YOLOFaceDetector, que organiza todo o pipeline de treino, validação e teste do modelo.
  3. **3 trying the model:** um trecho de vídeo com imagens não utilizadas no treino é submetido ao modelo para avaliação.
  
  
---
🔍 **Desempenho do modelo**

Aqui, vemos uma confusão em alguns frames entre as classes Arya, Sansa e Others. No geral, o modelo detecta bem o rosto do John e confunde o rosto de Bran, que deveria ser Others, com o de Árya. Pode ser que os exemplos com Bran/Others tenham sido menos frequentes em treino.

De qualquer forma, isso pode ser bem contornado adicionando mais exemplos de Arya, bem como das demais classes, mantendo o equilíbrio entre os totais de amostras.

![prediction 1](https://raw.githubusercontent.com/FlaviaLopes/dio-challenges-coding-the-future-with-baires-dev/refs/heads/main/projeto_5/output/predictions_got-the-end_1_.gif)

Aqui, mais um trecho do vídeo de teste.
![prediction 2](https://raw.githubusercontent.com/FlaviaLopes/dio-challenges-coding-the-future-with-baires-dev/refs/heads/main/projeto_5/output/predictions_got-the-end_2_.gif)
