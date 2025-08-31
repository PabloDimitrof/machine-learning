## Introdução
O objetivo deste roteiro é utilizar as bibliotecas `pandas`, `numpy`, `matplotlib` e `scikit-learn`, além da base de dados escolhida no [Kagle](https://www.kaggle.com/datasets/mohamadsallah5/english-premier-league-stats20212024), para treinar e avaliar um algoritmo de árvore de decisão.  
A proposta é verificar como variáveis como estádio, público, posse de bola, passes e chances criadas influenciam no resultado final da partida (`class`), permitindo explorar o uso de modelos de Machine Learning no contexto esportivo.

## Base de dados
A base de dados utilizada neste projeto contém informações de partidas de futebol, totalizando **1140 linhas** e **40 colunas**. Entre as variáveis estão posse de bola, número de passes e chances criadas.  
A variável alvo escolhida é a coluna **`class`**, que indica o resultado da partida (vitória do mandante, empate ou vitória do visitante), sendo o objeto da classificação pelo algoritmo de árvore de decisão.

## Exploração dos Dados
A seguir foi feita uma análise do significado e composição de cada coluna presente na base, com a finalidade de identificar possíveis problemas a serem tratados posteriormente. As visualizações e estatísticas descritivas ajudam a compreender a natureza dos dados e orientar decisões de pré-processamento e modelagem.

=== "class"

    A coluna `class` é a variável alvo do projeto e representa o resultado da partida: vitória do mandante (*"h"*), empate (*"d"*) ou vitória do visitante (*"a"*). 
    Trata-se de uma variável categórica com três possíveis valores, sendo o objeto de classificação do modelo de árvore de decisão. 
    A análise exploratória dessa coluna é essencial para observar o balanceamento do conjunto de dados, isto é, se há proporções semelhantes ou discrepantes entre os resultados possíveis.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/class.py"
    ```

=== "attendance"

    A coluna `attendance` representa o público presente em cada partida, um indicador de contexto do jogo que pode refletir fatores como mando de campo, relevância do confronto e engajamento da torcida. Em bases de futebol, o público tende a variar bastante entre estádios e rodadas, podendo apresentar assimetria (jogos muito cheios em arenas grandes) e valores atípicos (clássicos, finais).

    Do ponto de vista analítico, é uma variável contínua útil para observar a distribuição de torcedores ao longo das partidas e investigar relações com o resultado (`class`).

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/attendance.py"
    ```

=== "stadium"

    A coluna `stadium` identifica o estádio onde a partida foi disputada. Trata-se de uma variável categórica, associada ao contexto e à capacidade do local, podendo refletir fatores como mando de campo, perfil da torcida, relevo do gramado e até particularidades logísticas. 
    Na análise exploratória, é útil observar a **frequência de jogos por estádio**, verificando balanceamento da amostra (quais arenas têm mais/menos partidas) e possíveis vieses (ex.: concentração em poucos estádios). 
    Relações mais profundas com o resultado (`class`) podem ser investigadas em etapas seguintes, mas aqui focamos em entender a **composição dessa variável** no conjunto de dados.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/stadium.py"
    ```

=== "home_possessions"

    A coluna `home_possessions` representa a porcentagem de posse de bola do time mandante em cada partida. 
    Trata-se de uma variável numérica contínua, normalmente variando entre 30% e 70% na maioria dos jogos, podendo indicar estilos de jogo (times que mantêm a bola ou que jogam mais reativamente). 
    A análise exploratória dessa variável permite observar a distribuição da posse de bola entre os mandantes, identificar valores atípicos e comparar a média do controle de jogo ao longo das rodadas. 
    Posteriormente, poderá ser interessante relacionar essa posse com o resultado final (`class`) para verificar padrões.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/home_possessions.py"
    ```

=== "away_possessions"

    A coluna `away_possessions` representa a porcentagem de posse de bola do time visitante em cada partida. 
    Por ser uma variável numérica contínua, sua distribuição ajuda a observar o comportamento dos visitantes em termos de controle de jogo, identificar valores atípicos e comparar a tendência média de posse fora de casa. 
    Em etapas posteriores, pode ser relacionada ao resultado (`class`) para investigar padrões de desempenho como visitante.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/away_possessions.py"
    ```

=== "Home Team"

    A coluna `Home Team` identifica o time mandante na partida. Embora esteja codificada numericamente no dataset, sua natureza é categórica (IDs de times). 
    Na análise exploratória, é útil observar a frequência de jogos por mandante, verificando o balanceamento da amostra entre os times que atuam em casa e possíveis concentrações. 
    Relações com o resultado (`class`) podem ser exploradas depois; aqui focamos em entender a composição dessa variável.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/home_team.py"
    ```

=== "Away Team"

    A coluna `Away Team` identifica o time visitante na partida. Apesar de codificada como número, sua natureza é categórica (IDs de times). 
    Na análise exploratória, observar a frequência de jogos por visitante ajuda a avaliar o balanceamento da amostra e possíveis concentrações de partidas em determinados clubes.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/away_team.py"
    ```

=== "home_pass"

    A coluna `home_pass` indica o número de passes realizados pelo time mandante em cada partida. 
    É uma variável numérica contínua que ajuda a caracterizar o estilo de jogo do mandante, podendo variar bastante entre equipes mais ou menos dependentes da posse de bola. 
    Na análise exploratória, observar a distribuição dos passes permite identificar médias, dispersão e valores atípicos.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/home_pass.py"
    ```

=== "away_pass"

    A coluna `away_pass` indica o número de passes realizados pelo time visitante em cada partida. 
    Sendo uma variável numérica contínua, sua distribuição mostra como os visitantes se comportam em termos de construção de jogadas e controle de posse fora de casa. 
    A análise exploratória ajuda a entender a média de passes, variações entre os jogos e eventuais valores extremos.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/away_pass.py"
    ```

=== "home_chances"

    A coluna `home_chances` representa a quantidade de chances de gol criadas pelo time mandante durante a partida. 
    É uma variável numérica discreta que indica o nível de ofensividade da equipe jogando em casa. 
    A análise exploratória permite identificar a frequência de jogos com poucas ou muitas oportunidades e verificar a dispersão desse tipo de estatística.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/home_chances.py"
    ```

=== "away_chances"

    A coluna `away_chances` indica a quantidade de chances de gol criadas pelo time visitante durante a partida. 
    É uma variável numérica discreta que ajuda a compreender a ofensividade dos times jogando fora de casa. 
    A análise exploratória mostra como os visitantes se comportam em termos de criação de oportunidades, permitindo identificar padrões de equilíbrio ou diferenças marcantes em relação aos mandantes.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-decisao/colunas/away_chances.py"
    ```


## Pré-processamento

Após a exploração inicial da base, foram aplicados procedimentos de pré-processamento para preparar os dados para o treinamento do modelo.  
Entre as etapas realizadas estão:

- **Conversão de tipos**: variáveis originalmente em texto com valores numéricos (como `attendance`) foram transformadas em formato numérico.  
- **Tratamento de valores categóricos**: colunas como `stadium`, `Home Team` e `Away Team` foram mantidas como categóricas, sendo posteriormente convertidas em variáveis numéricas por meio de técnicas de codificação.  
- **Remoção de colunas irrelevantes ou redundantes**: colunas de identificação e de tempo (`date`, `clock`, `links`), bem como estatísticas que representam vazamento de informação do resultado (ex.: `Goals Home`, `Away Goals`), foram descartadas do conjunto de treino.  
- **Separação entre features e target**: as variáveis explicativas (`X`) foram definidas a partir de aproximadamente dez colunas relevantes da base, enquanto a variável alvo (`y`) é a coluna `class`.  
- **Divisão em treino e teste**: o conjunto de dados foi dividido em duas partes, garantindo estratificação do alvo para manter o equilíbrio das classes.

Com essas etapas, os dados foram organizados de forma consistente, reduzindo ruídos e preparando a base para a etapa seguinte: o treinamento do modelo de árvore de decisão.

=== "Base preparada"

    ```python exec="1"
    --8<-- "docs/arvore-decisao/pre.py"
    ```
=== "code"

    ```python exec="0"
    --8<-- "docs/arvore-decisao/pre.py"
    ```
=== "Base original"

    ```python exec="1"
    --8<-- "docs/arvore-decisao/base0.py"
    ```

## Divisão dos Dados

Com a base pré-processada, realizou-se a divisão entre conjuntos de **treinamento** e **teste**.  
O objetivo dessa etapa é garantir que o modelo seja avaliado em dados que ele nunca viu durante o treinamento, permitindo uma medida mais confiável de sua capacidade de generalização.  

Foi utilizada a função `train_test_split` da biblioteca *scikit-learn*, com os seguintes critérios:  
- **70% dos dados** destinados ao treinamento, para que o modelo aprenda os padrões da base;  
- **30% dos dados** destinados ao teste, para avaliar o desempenho em novos exemplos;  
- **Estratificação pelo alvo (`class`)**, garantindo que a proporção entre vitórias do mandante, empates e vitórias do visitante fosse mantida em ambos os conjuntos;  
- **Random State** fixado, assegurando reprodutibilidade na divisão.  

Dessa forma, o conjunto de treinamento foi usado para ajustar os parâmetros da árvore de decisão, enquanto o conjunto de teste serviu para medir a precisão e robustez do modelo.

```python exec="0"
--8<-- "docs/arvore-decisao/div.py"
```

## Treinamento do Modelo

=== "Modelo da Árvore"
    ```python exec="on" html="1"    
    --8<-- "docs/arvore-decisao/arvore.py"
    ```
=== "code"
    ```python exec="0"    
    --8<-- "docs/arvore/arvore.py"
    ```

## Avaliação do Modelo

Após o treinamento da árvore de decisão, o modelo foi avaliado no conjunto de teste, obtendo uma **precisão (accuracy) de aproximadamente 0.48**.  
Esse valor mostra que o modelo acerta menos da metade das previsões, o que indica uma performance limitada.  

Contudo, esse resultado deve ser entendido à luz da **natureza do problema**: prever o resultado de uma partida de futebol é uma tarefa de alta complexidade e incerteza.  
Além das estatísticas presentes na base, fatores externos como clima, lesões, decisões de arbitragem, motivação individual e até elementos de acaso influenciam diretamente no desfecho do jogo.  
Mesmo com acesso a uma base de dados mais ampla, a previsão exata de partidas continuaria sendo altamente incerta, pois o futebol não é uma ciência exata, mas um evento esportivo com variáveis muitas vezes imprevisíveis.  

Algumas observações complementares ajudam a interpretar esse desempenho:
- **Baixa capacidade preditiva**: a árvore não conseguiu capturar padrões fortes o suficiente para discriminar corretamente as três classes (`h`, `d`, `a`).  
- **Balanceamento do alvo**: embora tenha sido aplicada estratificação, empates tendem a ser mais raros e difíceis de prever, prejudicando a acurácia global.  
- **Variáveis contextuais**: as colunas usadas (como estádio, público, posse de bola, passes e chances) explicam parte do comportamento da partida, mas não determinam o resultado sozinhas.  
- **Complexidade inerente ao domínio**: a imprevisibilidade faz parte da própria essência do futebol, o que limita naturalmente a capacidade de qualquer modelo estatístico atingir altas taxas de acerto.  

Portanto, a precisão de 0.48 reflete tanto as limitações do modelo quanto a complexidade do fenômeno analisado. O exercício é válido não apenas para medir desempenho, mas para evidenciar os desafios de aplicar técnicas de *Machine Learning* em contextos de elevada incerteza como o esporte.

## Relatório Final

O projeto teve como objetivo aplicar um algoritmo de árvore de decisão sobre uma base de partidas de futebol, explorando o uso de técnicas de *Machine Learning* em um contexto esportivo.  

A análise foi conduzida em etapas bem definidas:

- **Exploração dos Dados (EDA)**: foram selecionadas e analisadas cerca de dez variáveis relevantes, entre elas `stadium`, `attendance`, `Home Team`, `Away Team`, `home_possessions`, `away_possessions`, `home_pass`, `away_pass`, `home_chances` e `away_chances`. A partir de visualizações e estatísticas descritivas, foi possível compreender melhor a natureza de cada coluna e avaliar o balanceamento da variável alvo `class`, que representa o resultado da partida.  

- **Pré-processamento**: nesta fase, colunas de identificação e tempo (`date`, `clock`, `links`), bem como estatísticas diretamente ligadas ao resultado (como gols, finalizações e cartões), foram removidas para evitar vazamento de informação. Além disso, foram realizadas conversões de tipos e preparação de variáveis categóricas para futura codificação numérica.  

- **Divisão dos Dados**: o conjunto foi separado em treino (70%) e teste (30%), com estratificação do alvo para preservar a proporção entre vitórias do mandante, empates e vitórias do visitante. Essa divisão assegurou reprodutibilidade e uma avaliação justa do modelo.  

- **Treinamento e Avaliação**: a árvore de decisão foi treinada com os dados de treino e avaliada no conjunto de teste. O modelo alcançou uma acurácia de aproximadamente **0.48**, valor que indica desempenho limitado, mas esperado dentro do contexto.  

A avaliação evidenciou alguns pontos importantes:
- O modelo não conseguiu capturar padrões suficientemente fortes, refletindo a alta complexidade da tarefa.  
- Empates, por serem menos frequentes, foram mais difíceis de prever, prejudicando a performance global.  
- As variáveis selecionadas (contexto e estatísticas gerais) ajudam a caracterizar as partidas, mas não são determinantes para o resultado final.  
- O futebol, por sua própria natureza, é altamente imprevisível e sujeito a variáveis externas que não estão presentes no dataset, como clima, arbitragem, lesões e fatores psicológicos.  

### Conclusão

A experiência demonstrou que, embora seja possível aplicar técnicas de *Machine Learning* ao domínio esportivo, os resultados precisam ser interpretados com cautela. O desempenho de 0.48 de acurácia não deve ser visto apenas como limitação do modelo, mas também como reflexo da complexidade e da imprevisibilidade inerente ao futebol.  

O projeto reforça a importância da análise exploratória, do cuidado no pré-processamento para evitar vazamento de dados e da avaliação crítica dos resultados. Além disso, abre espaço para trabalhos futuros que explorem variáveis adicionais — como forma recente dos times, desempenho de jogadores e fatores externos — e ajustes de hiperparâmetros para buscar melhorias no desempenho do modelo.  

Mais do que prever com exatidão os resultados, este trabalho mostra o potencial e os limites do uso de algoritmos de aprendizado de máquina em contextos reais, nos quais a incerteza e a imprevisibilidade são componentes inevitáveis.