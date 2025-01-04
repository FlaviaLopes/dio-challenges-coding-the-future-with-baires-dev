# Redução de dimensionalidade em imagens para redes neurais

A solução está no notebook do Google Colab `Redução de dimensionalidade de imagens.ipynb`.


$Mas$ $antes$ $disso$...

**Entenda o sistema RGB**

As cores vermelha, verde e azul são as cores primárias do sistema RGB, no qual todas as demais cores são obtidas a partir de combinações dessas três. Uma imagem RGB é composta por três canais, cada um representando uma dessas cores primárias.

Cada pixel da imagem é representado por três valores correspondentes às intensidades de luminosidade nos canais vermelho (Red), verde (Green) e azul (Blue). Esses valores variam de 0 a 255, onde 0 representa a ausência de luz (escuridão total) e 255 representa a intensidade máxima de luminosidade em cada canal.

Portanto, na imagem RGB, cada pixel tem uma combinação de vermelho, verde e azul que, juntos, criam uma vasta gama de cores. Por exemplo, o pixel (255, 0, 0) é vermelho puro, enquanto o pixel (0, 255, 0) é verde puro.

**O que são as escalas de Cinza e binária**

A escala de cinza é gerada quando as três cores primárias (vermelho, verde e azul) têm a mesma intensidade. Por exemplo, o pixel RGB (128, 128, 128) resulta em um tom de cinza médio. Contudo, o olho humano percebe a intensidade de brilho dessas cores de maneira desigual. Para equilibrar essa percepção, é utilizada a fórmula da NTSC:

$Luminância = 0.299 × Red + 0.587 × Green + 0.114 × Blue$

Essa transformação ajusta a contribuição de cada canal para criar uma representação precisa e equilibrada da imagem em tons de cinza.

Uma imagem em preto e branco (ou binária) é ainda mais simplificada, pois possui apenas dois níveis de intensidade: preto (0) e branco (255). Para converter uma imagem em tons de cinza para binária, utiliza-se um limiar (threshold). Por exemplo, valores maiores que 127 são convertidos para 255 (branco), enquanto os menores ou iguais a 127 são convertidos para 0 (preto).

Essa abordagem é útil para destacar contrastes marcantes em imagens e simplificar o processamento em tarefas como reconhecimento de padrões ou segmentação de objetos.

**Redução de Dimensionalidade**

Ao converter uma imagem RGB para tons de cinza, estamos realizando uma forma de redução de dimensionalidade. Uma imagem RGB tem três canais (dimensão $m \times n \times 3$), enquanto uma imagem em escala de cinza tem apenas um canal (dimensão $m \times n$).

Essa redução diminui significativamente a quantidade de dados que precisa ser processada, o que pode ser útil em aplicações de aprendizado de máquina e visão computacional. Entretanto, essa simplificação também implica em perda de informações, como a distinção entre cores que possuem a mesma luminância. Por isso, essa transformação só deve ser aplicada em problemas onde a cor não é uma característica importante.

Da mesma forma, ao converter uma imagem em tons de cinza para binária, reduzimos ainda mais os dados processados, simplificando o problema para dois valores possíveis por pixel (preto ou branco). Essa etapa é crucial em contextos onde o contraste é mais importante do que as informações detalhadas, como na leitura de códigos de barras ou na identificação de formas básicas em uma imagem.
