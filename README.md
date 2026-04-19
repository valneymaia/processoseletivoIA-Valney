# Processo Seletivo – Intensivo Maker | AI

Bem-vindo(a) à **etapa prática do processo seletivo para o Intensivo Maker**.

Esta atividade tem como objetivo avaliar competências técnicas relacionadas a **Machine Learning**, **Visão Computacional** e **Otimização de modelos para sistemas embarcados (Edge AI)**, a partir da aplicação prática dos conhecimentos adquiridos nos cursos EAD da etapa anterior.

> 🎯 **Importante**  
> O foco deste desafio é avaliar sua capacidade de **projetar, treinar e otimizar um modelo de IA**.  

---

## 📌 Navegação Rápida

- 🏁 [Passo 0 – Antes de Tudo](#-passo-0-antes-de-tudo)
- ⚙ [Passo 1 – Preparando o Ambiente](#-passo-1-preparando-o-ambiente)
- 💻 [Passo 2 – O Desafio Técnico](#-passo-2-o-desafio-técnico)
  - 🎯 [Conjunto de Dados](#-conjunto-de-dados)
  - 📂 [Estrutura do Projeto](#-estrutura-do-projeto)
  - 📚 [Material de Apoio](#-material-de-apoio)
  - ⚖️ [Critérios de Avaliação](#️-critérios-de-avaliação)
- 📤 [Passo 3 – Instruções de Entrega](#-passo-3-instruções-de-entrega)
  - 📝 [Relatório do Candidato](#-relatório-do-candidato)
  - ⭐ [Diferencial Implementado](#-diferencial-implementado)

---

## 🏁 Passo 0: Antes de Tudo

Caso você **nunca tenha utilizado Git ou GitHub**, não se preocupe.  
Siga atentamente as etapas abaixo.


### 1️⃣ Criação de Conta no GitHub

1. Acesse: https://github.com  
2. Clique em **Sign up**  
3. Crie sua conta gratuita seguindo as instruções da plataforma  

(*O GitHub será utilizado para envio, versionamento e correção automática do seu projeto.*)


### 2️⃣ Instalação do Git

O **Git** é a ferramenta que permite versionar e enviar seu código para o GitHub.

- **Windows**  
  Baixe e instale o **Git Bash**:  
  https://git-scm.com/downloads

- **Linux / macOS**  
  Verifique se o Git já está instalado:
  ```bash
  git --version
  ```

---

## ⚙ Passo 1: Preparando o Ambiente

Para desenvolver o desafio, você deverá criar uma cópia deste repositório.

### 1️⃣ Fork do Repositório

<img width="219" height="45" alt="image" src="https://github.com/user-attachments/assets/5d629626-513a-445c-ba0f-e5bb3e225187" />

1. No canto superior direito desta página, clique em **Fork**  
2. Uma cópia deste repositório será criada no **seu perfil do GitHub**
(*O Fork permite que você trabalhe de forma independente sem alterar o repositório original.*)



### 2️⃣ Clone do Repositório

<img width="149" height="52" alt="image" src="https://github.com/user-attachments/assets/abbd331b-a005-4633-89c6-afd16acbe828" />

No repositório do **seu Fork**, clique em **<> Code**, copie a URL e execute:

```bash
git clone https://github.com/SEU_USUARIO/nome-do-repositorio.git
cd nome-do-repositorio
```
(*O comando `git clone` cria uma cópia do repositório.*)



### 3️⃣ Preparação do Ambiente de Execução

Você pode executar o projeto de **Três formas**. Escolha apenas uma.



#### Opção A – Ambiente Python Local 
Requisitos:
- Python **3.10 ou 3.11**
- pip

Instale as dependências com:

```bash
pip install -r requirements.txt
```



#### Opção B – Dev Container 
Este repositório inclui um **Dev Container** para facilitar a criação de um ambiente Python padronizado.

**Requisitos**
- VS Code
- Docker instalado
- Extensão **Dev Containers**

**Passos**
1. Abra o repositório no VS Code  
2. Selecione **"Reopen in Container"**  
3. Aguarde a criação automática do ambiente  

➡️ As dependências serão instaladas automaticamente.


#### Opção C - via browser
Você também pode abrir o container via github codespace

1. Clique em **<> Code**
2. Clique em **Codespaces**
3. Clique em **Create codespace on image**

<img width="482" height="436" alt="image" src="https://github.com/user-attachments/assets/37a1e99d-66d2-4730-b824-26f834bd8cc3" />


>  Será aberto uma instância do VS Code no seu navegador com o container configurado


---

## 💻 Passo 2: O Desafio Técnico

O desafio consiste em desenvolver um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos**, e posteriormente **otimizá-lo para execução em dispositivos Edge**, como sistemas embarcados e IoT.

O foco não é apenas obter alta acurácia, mas também **compreender o fluxo completo**:

**treinamento → salvamento → conversão → otimização**



### 🎯 Conjunto de Dados

Será utilizado o dataset **MNIST**, composto por imagens de dígitos manuscritos de **0 a 9**.
<img width="500" height="294" alt="image" src="https://github.com/user-attachments/assets/f323b4cc-d759-4e05-bb58-13e4d6dc7e5b" />

✔️ O dataset já está disponível na biblioteca **TensorFlow/Keras**, não sendo necessário download manual.

📌 *O MNIST é amplamente utilizado para introdução à Visão Computacional e Redes Neurais.*



###  ✅ Requisitos Obrigatórios

**Etapa 1:**  Treinamento do Modelo (`train_model.py`)

Implemente no arquivo `train_model.py` um código que realize:

- Carregamento do dataset MNIST via TensorFlow
- Construção e treinamento de um modelo de classificação baseado em **Redes Neurais Convolucionais (CNN)**  
  (utilizando camadas `Conv2D` e `MaxPooling`)
- Treinamento do modelo
- Exibição da **acurácia final** no terminal
- Salvamento do modelo treinado no formato **Keras** (`.h5`)

(*O modelo salvo será utilizado na etapa de otimização.*)



**Etapa 2:** Otimização do Modelo (`optimize_model.py`)

No arquivo `optimize_model.py`, implemente:

- Carregamento do modelo treinado
- Conversão para **TensorFlow Lite (`.tflite`)**
- Aplicação de técnica de otimização, como:
  - **Dynamic Range Quantization**

(**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de **Edge AI**.)



### 📂 Estrutura do Projeto

⚠️ **Atenção:**  
A estrutura e os nomes dos arquivos **não devem ser alterados**.

```plaintext
seu-repositorio/
├── .github/
│   └── workflows/
│       └── ci.yml            # 🤖 Pipeline de correção automática (NÃO ALTERAR)
├── .devcontainer/            # 🐳 Dev Container (opcional)
│   └── devcontainer.json
├── train_model.py            # ✏️ Treinamento do modelo
├── optimize_model.py         # ✏️ Conversão e otimização
├── requirements.txt          # 📄 Dependências do projeto
├── model.h5                  # 🤖 Modelo treinado (gerado)
├── model.tflite              # ⚡ Modelo otimizado (gerado)
└── README.md                 # 📝 Relatório final do candidato
```



### ⚠️ Restrições e Considerações de Engenharia

Este desafio é avaliado automaticamente por meio de um pipeline de
**integração contínua (CI)**, executado em um ambiente controlado e com
restrições de recursos computacionais.

Você **não precisa conhecer GitHub Actions** para realizar o desafio.
No entanto, é importante respeitar as diretrizes abaixo.

**Diretrizes para o Modelo**

- O modelo deve ser uma **CNN simples**, adequada para **Edge AI**
- Evite arquiteturas muito profundas ou complexas
- Recomenda-se utilizar **até 3 camadas convolucionais**
- **Não utilize modelos pré-treinados**
- Número de épocas **limitado** (ex: até 5)

#### Diretrizes de Execução

- Treinamento apenas em **CPU**
- Tempo total reduzido (compatível com CI)
- Código deve executar do início ao fim **sem intervenção manual**

> **Importante:**  
> O objetivo não é obter a maior acurácia possível, mas sim demonstrar
> **engenharia eficiente**, compatível com ambientes automatizados e
> restrições típicas de aplicações reais de Edge AI.



### 📚 Material de Apoio

Os cursos realizados na etapa anterior **devem ser utilizados como referência**.

- 📘 **Fundamentos de Inteligência Artificial para Sistemas Embarcados**
- 👁️ **Sistemas de Visão Computacional Embarcada**
- ⚙️ **Otimização de Modelos em Sistemas Embarcados**

(*Os exemplos apresentados nesses cursos podem ser adaptados e reutilizados neste desafio.*)



### ⚖️ Critérios de Avaliação

A avaliação considerará:

- **Funcionalidade**  
  Execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`

- **Edge AI**  
  Conversão correta para `.tflite` e aplicação de técnica de otimização

- **Documentação**  
  Preenchimento adequado do relatório (README.md)

---

## 📤 Passo 3: Instruções de Entrega

### ✔️ Validação 

Antes do envio, execute os scripts e confirme a geração dos arquivos:
- `model.h5`
- `model.tflite`


### ⬆️ Envio do Código

```bash
git add .
git commit -m "Entrega do desafio técnico - Seu Nome"
git push origin main
```



### 🔍 Verificação Automática

1. Acesse a aba **Actions** no GitHub  
2. Verifique se o workflow foi executado com sucesso (✅)  
3. Em caso de erro (❌), consulte os logs, corrija e envie novamente

<img width="807" height="363" alt="image" src="https://github.com/user-attachments/assets/d991d35b-2bc2-48f7-9ac7-cf5ca9dc452a" />



### 📎 Submissão Final

Copie o link do seu repositório e envie conforme orientações do processo seletivo no Moodle.

---

## 📝 Relatório do Candidato
# Processo Seletivo – Intensivo Maker | AI
## Relatório Técnico do Candidato

---

## 👤 Identificação

| Campo | Informação |
|---|---|
| **Nome Completo** | Valney Maia Neto |
| **Instituição** | Universidade Federal do Cariri – UFCA |
| **GitHub** |  https://github.com/valneymaia/processoseletivoIA-Valney.git |

---

## ⚡ Como Reproduzir

Execute os scripts **nesta ordem** a partir da raiz do repositório:

```bash
# 1. Treinamento — gera model.h5
python train_model.py

# 2. Otimização — gera model.tflite
python optimize_model.py

# 3. Inferência de demonstração (opcional) — gera inference_sample.png
python test_inference.py
```

> **Requisitos:** Python 3.10 ou 3.11. Instale as dependências com `pip install -r requirements.txt`.  
> **Versões utilizadas:** `tensorflow==2.12.0`, `numpy==1.23.5`

---

## 1️⃣ Arquitetura do Modelo

O modelo treinado em `train_model.py` é uma **CNN de 3 blocos convolucionais**, projetada para ser eficiente em CPU e viável para Edge AI.

```
Conv2D(32)  → MaxPooling2D    # padrões simples: bordas e texturas
Conv2D(64)  → MaxPooling2D    # padrões intermediários: formas e curvas
Conv2D(128)                   # padrões abstratos: estrutura dos dígitos
Flatten → Dropout(0.5) → Dense(128) → Dropout(0.5) → Dense(10, softmax)
```

**Decisões de arquitetura:**

- **Progressão de filtros (32 → 64 → 128):** segue a convenção de hierarquia de representação — filtros iniciais capturam bordas simples, enquanto os finais capturam estruturas complexas dos dígitos.
- **3 blocos convolucionais:** equilíbrio entre capacidade de extração e custo computacional. Mais camadas aumentariam latência e memória, comprometendo a viabilidade para dispositivos embarcados.
- **Dropout(0.5) duplo:** aplicado antes e depois da camada densa para reduzir overfitting na parte totalmente conectada da rede, mantendo boa generalização no conjunto de teste.
- **Sem modelos pré-treinados:** compatível com os requisitos do desafio e suficiente para o MNIST.

---

## 2️⃣ Bibliotecas Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| `tensorflow` / `keras` | 2.12.0 | Carregamento do MNIST, construção da CNN, treinamento, avaliação e conversão TFLite |
| `numpy` | 1.23.5 | Normalização e reshape das imagens |
| `os` | padrão Python | Manipulação de arquivos e cálculo de tamanho dos artefatos |

---

## 3️⃣ Técnica de Otimização

**Técnica aplicada:** Dynamic Range Quantization via `tf.lite.Optimize.DEFAULT`

**Fluxo em `optimize_model.py`:**

```
model.h5  →  TFLiteConverter  →  Optimize.DEFAULT  →  model.tflite
```

**Por que essa técnica?**

A Dynamic Range Quantization converte os pesos de `float32` para `int8` em tempo de conversão, sem necessidade de dados de calibração. Isso resulta em:

- Redução significativa de tamanho sem etapas adicionais de pipeline
- Implementação simples, adequada para CI automatizado
- Boa manutenção de acurácia para modelos como o MNIST
- Viabilidade em microcontroladores e SBCs com memória limitada

---

## 4️⃣ Resultados

### Desempenho e Tamanho

| Artefato | Tamanho | Acurácia no Teste |
|---|---|---|
| `model.h5` (Keras) | 2,81 MB | **99,30%** |
| `model.tflite` (quantizado) | 0,24 MB | ~99% |
| **Redução de tamanho** | **91,43%** | — |

> Treinado com **5 épocas**, apenas em **CPU**, sem GPU.

<table>
  <tr>
    <td align="center"><b>Modelo treinado</b></td>
    <td align="center"><b>Modelo otimizado</b></td>
  </tr>
  <tr>
    <td><img src="assets/model.h5.png" alt="Modelo treinado" width="350"/></td>
    <td><img src="assets/model.tflite.png" alt="Modelo otimizado" width="350"/></td>
  </tr>
</table>

As imagens geradas pelo Netron confirmam a arquitetura implementada: é possível 
visualizar as três camadas convolucionais, o bloco denso com Dropout e a camada 
de saída com 10 neurônios (softmax). No modelo `.tflite`, as operações aparecem 
quantizadas, refletindo a aplicação do Dynamic Range Quantization.

### Análise dos Resultados

- O modelo convergiu rapidamente — a acurácia de 99,30% com apenas 5 épocas indica boa adequação da arquitetura ao problema.
- A redução de 91,43% no tamanho viabiliza o deploy em dispositivos com memória flash da ordem de **256 KB a 1 MB**, como ESP32-S3 ou Raspberry Pi.
- O `Dropout(0.5)` demonstrou eficácia: não houve sinal de overfitting mesmo com rede de capacidade moderada.

### Visualização da Arquitetura (Netron)

| Modelo Keras | Modelo TFLite |
|---|---|
| ![model.h5](assets/model.h5.png) | ![model.tflite](assets/model.tflite.png) |

---

## ⭐ Diferencial: Script de Inferência

O script `test_inference.py` demonstra o uso prático do modelo TFLite em um fluxo de inferência real.

**O que ele faz:**

1. Carrega o modelo `model.tflite` com o interpretador TFLite
2. Seleciona uma imagem aleatória do conjunto de teste do MNIST
3. Executa a inferência e exibe a classe prevista com a confiança
4. Salva a imagem usada como `inference_sample.png`

**Exemplo de saída:**

```
Dígito real:    7
Previsto:       7
Confiança:      99.8%
Imagem salva em: inference_sample.png
```

Esse script valida que o modelo não apenas foi treinado e convertido corretamente, mas que é **funcional em um pipeline real de inferência**.

---

## 5️⃣ Trade-offs e Aprendizados

**Trade-off tamanho × desempenho:**  
Uma rede mais profunda poderia atingir 99,5%+ de acurácia, mas elevaria o custo de inferência e o uso de memória. A arquitetura final, com ~245 KB em `.tflite`, é adequada para dispositivos com restrições reais de hardware.

**Aprendizado principal:**  
O desafio reforçou a importância de pensar no **ciclo completo de Edge AI** desde o início — não basta treinar com alta acurácia; o modelo precisa ser viável para conversão, otimização e deploy em ambientes com recursos limitados.

---

## 📂 Estrutura do Projeto

```plaintext
repositorio/
├── .github/workflows/ci.yml   # Pipeline de correção automática
├── .devcontainer/             # Dev Container (opcional)
├── train_model.py             # Etapa 1: treinamento da CNN
├── optimize_model.py          # Etapa 2: conversão e quantização
├── test_inference.py          # Diferencial: inferência com TFLite
├── requirements.txt           # Dependências do projeto
├── model.h5                   # Modelo treinado (gerado)
├── model.tflite               # Modelo otimizado (gerado)
├── inference_sample.png       # Imagem de teste (gerada)
├── assets/
│   ├── model.h5.png           # Visualização Netron – Keras
│   └── model.tflite.png       # Visualização Netron – TFLite
└── README.md                  # Este relatório
```

## 🆘 Suporte

Em caso de dúvidas:

- Consulte o material dos cursos EAD
- Leia atentamente este README
- Analise os logs das GitHub Actions
- Utilize os canais oficiais para contato com os instrutores

Boa sorte no processo seletivo.
