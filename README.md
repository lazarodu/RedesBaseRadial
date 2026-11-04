# 🧠 Simulador Didático de Redes Neurais RBF (Radial Basis Function)

Aplicação web interativa, em arquivo único (`simulador_rbf.html`), desenvolvida para fins pedagógicos e experimentais sobre **Redes de Base Radial (RBF)** com **Treinamento Híbrido em Dois Estágios**, rigorosamente baseada no **Capítulo 6** do livro:

> **Silva, I. N.; Spatti, D. H.; Flauzino, R. A.**  
> *Redes Neurais Artificiais: para engenharia e ciências aplicadas*. Artliber Editora.

---

## 📋 Sumário
1. [Visão Geral](#-visão-geral)
2. [Fundamentação Teórica e Algoritmos](#-fundamentação-teórica-e-algoritmos)
   - [Topologia da Rede](#topologia-da-rede)
   - [Estágio I: Aprendizado Não-Supervisionado (k-means)](#estágio-i-aprendizado-não-supervisionado-k-means)
   - [Estágio II: Aprendizado Supervisionado (Regra Delta)](#estágio-ii-aprendizado-supervisionado-regra-delta)
   - [Solução Ótima por Mínimos Quadrados (Pseudo-Inversa)](#solução-ótima-por-mínimos-quadrados-pseudo-inversa)
3. [Problemas e Conjuntos de Dados Inclusos](#-problemas-e-conjuntos-de-dados-inclusos)
4. [Recursos da Interface Gráfica](#-recursos-da-interface-gráfica)
5. [Fenômenos Didáticos Observáveis](#-fenômenos-didáticos-observáveis)
6. [Como Executar](#-como-executar)
7. [Estrutura do Código](#-estrutura-do-código)

---

## 🌟 Visão Geral

Diferente de um Perceptron Multicamadas (PMC / MLP) com retropropagação clássica (Backpropagation) — que atualiza todas as camadas de forma supervisionada, não-linear e frequentemente lenta —, a **Rede RBF** desacopla a representação espacial e a saída linear por meio de um processo **híbrido**:

1. **Camada Intermediária:** Nós ocultos com funções de base radial (Gaussianas) que mapeiam o espaço de entrada de forma local (campos receptivos delimitados por centros $\mathbf{c}_j$ e raios $\sigma_j$).
2. **Camada de Saída:** Combinação linear com pesos $W^{(2)}$ e limiar $\theta$, treinada com a simples Regra Delta (Adaline/LMS) ou solução analítica por mínimos quadrados.

O simulador expõe essa dinâmica passo a passo através de renderizações em tempo real em HTML5 Canvas, painel matemático com equações síncronas e sondas interativas.

---

## 📐 Fundamentação Teórica e Algoritmos

### Topologia da Rede

A arquitetura geral é do tipo $n_{in} - n_1 - 1$:
- **Entrada:** Vetor $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$ (não possui pesos ou processamento; apenas distribui os sinais).
- **Camada Oculta:** $n_1$ neurônios com ativação gaussiana:
  $$g_j(\mathbf{x}) = \exp\left( -\frac{\|\mathbf{x} - \mathbf{W}_j^{(1)}\|^2}{2\sigma_j^2} \right), \quad j = 1, \dots, n_1$$
  onde $\mathbf{W}_j^{(1)} = \mathbf{c}_j$ representa o centro do $j$-ésimo neurônio e $\sigma_j^2$ é a variância (dispersão do campo receptivo).
- **Camada de Saída:** Neurônio linear que combina as saídas da camada oculta e subtrai o limiar $\theta$:
  $$y = \sum_{j=1}^{n_1} W_j^{(2)} \cdot g_j(\mathbf{x}) - \theta = \mathbf{z}^T \mathbf{W}^{(2)}$$
  com vetor pseudo-amostra expandido $\mathbf{z} = [g_1(\mathbf{x}), \dots, g_{n_1}(\mathbf{x}), -1]^T$ e pesos estendidos $\mathbf{W}^{(2)} = [W_1^{(2)}, \dots, W_{n_1}^{(2)}, \theta]^T$.

---

### Estágio I: Aprendizado Não-Supervisionado (k-means)

Os parâmetros da camada intermediária ($\mathbf{W}_j^{(1)}$ e $\sigma_j^2$) são determinados sem supervisão (ou apenas sobre as amostras de presença no caso de detecção/classificação):

1. **Inicialização:** Os centros $\mathbf{W}_j^{(1)}$ recebem as $n_1$ primeiras amostras do conjunto de treinamento.
2. **Particionamento:** Cada amostra $\mathbf{x}_k$ é associada ao agrupamento $\Omega^{(j)}$ cujo centro $\mathbf{W}_j^{(1)}$ for euclidianamente mais próximo:
   $$\mathbf{x}_k \in \Omega^{(j)} \iff \|\mathbf{x}_k - \mathbf{W}_j^{(1)}\| = \min_l \|\mathbf{x}_k - \mathbf{W}_l^{(1)}\|$$
3. **Atualização dos Centros:**
   $$\mathbf{W}_j^{(1)} = \frac{1}{m^{(j)}} \sum_{\mathbf{x} \in \Omega^{(j)}} \mathbf{x}$$
   onde $m^{(j)}$ é o número de amostras contidas no grupo $\Omega^{(j)}$.
4. **Critério de Parada:** As iterações encerram quando nenhum ponto muda de agrupamento (estabilidade dos grupos) ou ao atingir o limite estipulado.
5. **Cálculo da Dispersão ($\sigma_j^2$):** A variância de cada campo gaussiano é calculada pela distância quadrática média dos pontos pertencentes ao agrupamento até o seu respectivo centro:
   $$\sigma_j^2 = \frac{1}{m^{(j)}} \sum_{\mathbf{x} \in \Omega^{(j)}} \|\mathbf{x} - \mathbf{W}_j^{(1)}\|^2$$
   *(No simulador, um piso de segurança e um fator multiplicativo $\beta$ são disponibilizados para prevenir singularidade quando $\sigma^2 \to 0$)*.

Ao concluir o Estágio I, **a camada intermediária é congelada**.

---

### Estágio II: Aprendizado Supervisionado (Regra Delta)

Com os centros e variâncias fixados, as ativações $g_j(\mathbf{x}_k)$ comportam-se como novas variáveis de entrada linearmente desacopladas. O ajuste dos pesos da saída $\mathbf{W}^{(2)}$ segue a **Regra Delta (LMS)**:

1. Apresentação da $k$-ésima amostra com entrada expandida $\mathbf{z}^{(k)} = [g_1(\mathbf{x}^{(k)}), \dots, g_{n_1}(\mathbf{x}^{(k)}), -1]^T$.
2. Cálculo da saída linear e do erro instantâneo:
   $$y^{(k)} = \mathbf{z}^{(k)T} \mathbf{W}^{(2)}$$
   $$e^{(k)} = d^{(k)} - y^{(k)}$$
3. Atualização dos pesos da camada de saída e do limiar:
   $$\Delta W_j^{(2)} = \eta \cdot e^{(k)} \cdot g_j(\mathbf{x}^{(k)}), \quad \Delta\theta = -\eta \cdot e^{(k)}$$
4. Ao final de cada época, afere-se o Erro Médio Quadrático ($E_M$):
   $$E_M = \frac{1}{2p} \sum_{k=1}^p \left( d^{(k)} - y^{(k)} \right)^2$$
5. Critério de parada:
   $$|E_M(t) - E_M(t-1)| \le \varepsilon \quad \text{ou} \quad t \ge \text{máx\_épocas}$$

---

### Solução Ótima por Mínimos Quadrados (Pseudo-Inversa)

Como a saída é estritamente linear em relação a $\mathbf{z}$, o problema de mínimos quadrados admite solução analítica global direta:
$$\mathbf{W}_{\text{ótimo}} = (\mathbf{Z}^T \mathbf{Z})^{-1} \mathbf{Z}^T \mathbf{d}$$
O simulador implementa essa solução por meio de equações normais resolvidas por **Eliminação Gaussiana com Pivotação Parcial**, permitindo comparar instantaneamente o aprendizado gradativo (Regra Delta) com a melhor solução teórica possível para aquela camada intermediária.

---

## 📂 Problemas e Conjuntos de Dados Inclusos

O simulador traz três estudos de caso com reprodutibilidade determinística via gerador pseudoaleatório (PRNG):

1. **★ Projeto 1: Radiação Nuclear (Classificação 2D)**
   - Caso clássico do Capítulo 6 do livro.
   - 50 situações de monitoramento (25 com presença de radiação $y=+1$, distribuídas em 2 núcleos, e 25 de ausência $y=-1$).
   - Topologia clássica 2-2-1 ($n_{in}=2$, $n_1=2$ gaussianas, 1 saída).
   - Opção para treinar o Estágio I apenas com as amostras positivas ($y = +1$), reproduzindo a metodologia original do livro.

2. **Questão 2: Porta Lógica XOR (Não-Linearmente Separável)**
   - As 4 combinações binárias padrão: $(0,0) \to -1$, $(0,1) \to +1$, $(1,0) \to +1$, $(1,1) \to -1$.
   - Demonstra a capacidade das funções de base radial em criar superfícies fechadas com apenas 2 neurônios intermediários sobre os pontos de interesse.

3. **Projeto 2: Aproximação de Função 1D Não-Linear**
   - 40 amostras ruidosas de $f(x) = 0.9 \cdot \sin(2\pi x)$.
   - Permite investigar a interpolação/superposição de múltiplas gaussianas ponderadas ($W_j \cdot g_j(x)$) variando $n_1 \in \{5, 8, 10, 15\}$.

---

## 🖥️ Recursos da Interface Gráfica

A interface foi projetada em tema escuro moderno com tipografia técnica (*Plus Jakarta Sans* e *JetBrains Mono*):

- **Visualização do Espaço de Entrada (Canvas 2D / 1D):**
  - Mapa de calor de predição contínua e fronteira de decisão $y=0$.
  - Círculos de campos receptivos com raio $\sigma_j$ (1 desvio) e $2\sigma_j$ (2 desvios).
  - Traçado dos centros $\mathbf{W}^{(1)}$ identificados como $\text{G}_1, \dots, \text{G}_{n_1}$.
  - Anéis coloridos indicando a que cluster $\Omega^{(j)}$ cada amostra pertence.
  - Sonda de teste interativa: clique ou arraste o mouse sobre qualquer coordenada para inspecionar em tempo real.
- **Topologia Dinâmica da Rede:**
  - Diagrama esquemático dos neurônios conectados.
  - Pulso e espessura das arestas de acordo com a magnitude e polaridade dos pesos $W_j^{(2)}$ e limiar $\theta$.
  - Efeito hover com *tooltips* detalhados de cada nó (coordenadas de centros, variância, ativação e peso).
- **Curva de Aprendizado em Escala Logarítmica:**
  - Traçado dinâmico da evolução de $E_M$ versus época.
  - Linha tracejada verde indicando o $E_M$ ótimo analítico para referência imediata.
- **Painel de Equações Didáticas (Coluna Direita):**
  - Equações renderizadas e tabelas numéricas atualizadas em tempo de execução.
  - Detalhamento de cada amostra processada no Estágio II: vetor $\mathbf{z}$, erro instantâneo $e = d - y$ e gradiente $\Delta \mathbf{W}$.

---

## 🔬 Fenômenos Didáticos Observáveis

O simulador facilita a observação de nuances teóricas fundamentais das RBFs:

- **Efeito Ponto Cego (Blind Spot):**
  Ao testar uma amostra muito afastada dos centros gaussianos, todas as ativações decaem exponencialmente para zero ($\max g_j \approx 0$). A saída da rede converge exclusivamente para $- \theta$. No caso de sistemas de alerta crítico (como detecção de radiação), isso acarreta um **falso negativo**, ressaltando a natureza de resposta estritamente local da RBF em contraste com a resposta global de planos de corte em redes MLP.
- **Convergência Acelerada:**
  Após a convergência do k-means, o treinamento da camada de saída não sofre com estagnação de platôs ou mínimos locais acentuados típicos de MLPs profundas, pois a superfície de erro para a camada linear é uma parábola convexa única.
- **Impacto do Fator $\beta$ (Extensão do Campo):**
  Modificar o parâmetro $\beta$ altera o espalhamento de cada gaussiana. Valores muito pequenos geram ativações pontuais com lacunas entre centros; valores muito altos perdem a especificidade local.

---

## 🚀 Como Executar

Por ser uma aplicação web puramente *client-side* (Zero-Dependency Vanilla JavaScript, HTML5 e CSS3):

1. Clone o repositório ou faça o download dos arquivos:
   ```bash
   git clone https://github.com/seu-usuario/RedesBaseRadial.git
   cd RedesBaseRadial
   ```
2. Abra o arquivo `simulador_rbf.html` diretamente em qualquer navegador moderno (Google Chrome, Firefox, Safari, Edge, etc.):
   - No terminal Linux/WSL:
     ```bash
     xdg-open simulador_rbf.html
     # ou no WSL abrindo no Windows:
     explorer.exe simulador_rbf.html
     ```
   - Ou simplesmente clique duas vezes no arquivo `simulador_rbf.html`.

> [!NOTE]
> Não é necessária instalação de Node.js, Python, servidores web ou gerenciadores de pacotes.

---

## 🧱 Estrutura do Código

O arquivo `simulador_rbf.html` é modularizado internamente com separação rígida entre modelo matemático e interface gráfica:

```text
simulador_rbf.html
├── <head> e <style>        -> Estilização visual moderna (CSS Glassmorphism e Grid/Flex)
├── <body> / <header>       -> Barra superior de status do treinamento e controle de estado
├── <main>                  -> Grid responsivo em 3 colunas
│   ├── Coluna Esquerda     -> Parâmetros (n₁, η, ε, β, dataset, botões de épocas e auto-treino)
│   ├── Coluna Central      -> Canvas da topologia, Canvas do espaço de decisão e cartões de métricas
│   └── Coluna Direita      -> Painel didático matemático (fórmulas, tabelas numéricas e sonda de teste)
└── <script>
    ├── NÚCLEO MATEMÁTICO   -> Sem dependência de DOM:
    │   ├── mulberry32 / gauss01 -> PRNG e geradores estocásticos reproduzíveis
    │   ├── buildDataset         -> Construtores dos datasets (Radiação, XOR, Função 1D)
    │   ├── Estágio I            -> initStage1, kmStep, afterStage1Change
    │   ├── Estágio II           -> resetStage2, deltaSample, runEpochs, calcEM
    │   └── solveLS              -> Solução por mínimos quadrados (Gauss Elimination)
    └── CAMADA DE INTERFACE -> Renderização gráfica, eventos de mouse/touch e sincronia de dados
```

---

## 📚 Referência Bibliográfica

- **Silva, I. N.; Spatti, D. H.; Flauzino, R. A.** *Redes Neurais Artificiais: para engenharia e ciências aplicadas*. São Paulo: Artliber Editora, 2010. Capítulo 6: Redes de Base Radial (RBF).

