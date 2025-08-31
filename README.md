# 🚀 Projeto Final – Bootcamp de Ciência de Dados e IA  

## 📌 Contexto  
Este projeto foi desenvolvido como parte do **Bootcamp de Ciência de Dados e Inteligência Artificial (CDIA)**.  
O desafio consiste em criar um sistema de **manutenção preditiva** para máquinas industriais, capaz de prever falhas com base em dados sensorizados.  

- **Objetivo principal:** prever se ocorrerá falha em uma máquina (`falha_maquina`).  
- **Objetivo extra:** identificar probabilidades associadas a diferentes tipos de falhas.  
- **Dados:** fornecidos pela empresa parceira, contendo medições de sensores IoT.  

---

## 📂 Estrutura do repositório  

projeto-bootcamp-cdia/
│
├── data/
│ ├── maquinas_train.csv # dados de treino (com rótulo)
│ └── maquinas_test.csv # dados de teste (sem rótulo)
│
├── src/
│ ├── SCRIPT - Projeto Final Bootcamp.py # versão inicial do projeto
│ └── SCRIPT - Projeto Final Bootcamp - vF.py # versão final consolidada
│
├── predicoes_maquinas_test.csv # resultados de predição do conjunto de teste
│
├── README.md # documentação do projeto
└── requirements.txt # bibliotecas necessárias


---

## ⚙️ Tecnologias utilizadas  
- **Linguagem:** Python 3.10+  
- **Principais bibliotecas:**  
  - pandas, numpy → manipulação de dados  
  - matplotlib, seaborn → visualização  
  - scikit-learn → métricas e avaliação  
  - statsmodels → modelagem estatística (GLM binomial)  
  - statstests → seleção de variáveis via Stepwise  
  - unidecode → padronização de colunas  

---

## 📊 Metodologia  
1. Análise exploratória e tratamento de dados  
2. Modelagem preditiva com Regressão Logística (GLM Binomial)  
3. Seleção de variáveis com Stepwise  
4. Avaliação com matriz de confusão, curva ROC, AUC e Gini  
5. Predição no conjunto de teste e exportação  

---

## 📈 Resultados e Avaliação  
- **Métricas utilizadas:** Acurácia, Sensitividade, Especificidade, ROC/AUC, Gini  
- **Visualizações:**  
  - Distribuição das probabilidades previstas  
  - Sensitividade e especificidade em função do cutoff  
  - Curva ROC  

---

## ▶️ Como executar  

1. Clone este repositório:  
```bash
git clone https://github.com/SEU_USUARIO/projeto-bootcamp-cdia.git
cd projeto-bootcamp-cdia

2. Instale as dependências:
pip install -r requirements.txt

3. Execute o script final:
python src/"SCRIPT - Projeto Final Bootcamp - vF.py"

4. Verifique as predições em:
predicoes_maquinas_test.csv

👤 Autor

Projeto desenvolvido por Fredson Luis Torres Alves no âmbito do Bootcamp CDIA – 2025.