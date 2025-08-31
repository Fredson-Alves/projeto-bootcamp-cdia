# RESIDÊNCIA EM CIÊNCIA DE DADOS E INTELIGÊNCIA ARTIFICAL SENAI/SENAC
# BOOTCAMP EM CDIA
# SUPERVISED MACHINE LEARNING: MODELOS LOGÍSTICO BINÁRIO
# FREDSON LUIS TORRES ALVES

#!/usr/bin/env python
# coding: utf-8

# In[0.1]: Instalação dos pacotes

!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install statstests

# In[0.2]: Importação dos pacotes

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada
import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from statstests.process import stepwise # procedimento Stepwise
from scipy import stats # estatística chi2
import plotly.graph_objects as go # gráficos 3D
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
#from statsmodels.discrete.discrete_model import MNLogit # estimação do modelo
                                                        #logístico multinomial
import warnings
warnings.filterwarnings('ignore')

# In[1.0]: Etapa 1: Carregamento e pré-processamento dos dados
import pandas as pd

df = pd.read_csv('maquinas_train.csv', delimiter=',')

# Verificando estrutura
print(df.info())
print(df.describe())
print(df.head())

# In[1.1]: Padronização de nomes das colunas

import pandas as pd
import unidecode

# Padronizar nomes de colunas
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace(r'[\(\)]', '', regex=True)
    .map(unidecode.unidecode)  # remove acentos
)

# Conferir resultado
print(df.columns)

# In[1.2]: Função: tratar_e_explorar_variaveis_numericas(df)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def tratar_e_explorar_variaveis_numericas(df, tratar_nulos='drop'):
    """
    Executa o tratamento e a análise exploratória de variáveis numéricas.
    
    Parâmetros:
        df (DataFrame): conjunto de dados original.
        tratar_nulos (str): 'drop' para remover ou 'fill' para preencher com mediana.
    
    Retorna:
        DataFrame limpo (com possíveis linhas removidas ou preenchidas).
    """

    # 1. Converter 'id' para string, se existir
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)

    # 2. Verificar e tratar valores nulos
    print("🔍 Verificando valores nulos:\n")
    print(df.isnull().sum())

    if tratar_nulos == 'drop':
        df = df.dropna()
        print("\n✅ Nulos removidos.")
    elif tratar_nulos == 'fill':
        df.fillna(df.median(numeric_only=True), inplace=True)
        print("\n✅ Nulos preenchidos com mediana.")
    else:
        print("\n⚠️ Opção inválida para tratamento de nulos. Nenhuma ação aplicada.")

    # 3. Verificar linhas duplicadas
    duplicadas = df.duplicated().sum()
    print(f"\n🔁 Linhas duplicadas: {duplicadas}")

    # 4. Corrigir valores inconsistentes
    if 'temperatura_ar' in df.columns:
        df = df[df['temperatura_ar'] > 270]
    if 'temperatura_processo' in df.columns:
        df = df[df['temperatura_processo'] > 273]
    if 'velocidade_rotacional' in df.columns:
        df = df[df['velocidade_rotacional'] >= 0]
    if 'desgaste_da_ferramenta' in df.columns:
        df = df[df['desgaste_da_ferramenta'] >= 0]

    # 5. Identificar variáveis numéricas
    atributos_numericos = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 6. Estatísticas descritivas
    print("\n📊 Estatísticas descritivas:")
    display(df[atributos_numericos].describe())

    # 7. Intervalos
    print("\n📌 Intervalos de valores:")
    intervalo = pd.DataFrame({
        'mínimo': df[atributos_numericos].min(),
        'máximo': df[atributos_numericos].max(),
        'amplitude': df[atributos_numericos].max() - df[atributos_numericos].min()
    })
    display(intervalo)

    # 8. Histogramas
    for col in atributos_numericos:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Histograma - {col}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 9. Boxplots individuais
    for col in atributos_numericos:
        plt.figure(figsize=(10, 2))
        sns.boxplot(x=df[col], color='salmon')
        plt.title(f'Boxplot - {col}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 10. Boxplots lado a lado
    df_melt = df[atributos_numericos].melt()
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_melt, x='variable', y='value', palette='Set3')
    plt.title('Boxplots das Variáveis Numéricas')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 11. Mapa de correlação
    print("\n🔗 Mapa de Correlação entre Variáveis Numéricas:")
    plt.figure(figsize=(10, 8))
    matriz_corr = df[atributos_numericos].corr()
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
    plt.title("Mapa de Correlação")
    plt.tight_layout()
    plt.show()

    return df


# In[1.3]: Tratar e explorar vairiáveis numéricas
df = tratar_e_explorar_variaveis_numericas(df, tratar_nulos='drop')  # ou 'fill'

# In[1.4]: Etapa de tratamento das variáveis qualitativas
for col in ['tipo', 'falha_maquina', 'fdf_falha_desgaste_ferramenta', 'fdc_falha_dissipacao_calor',
            'fp_falha_potencia', 'fte_falha_tensao_excessiva', 'fa_falha_aleatoria']:
    print(f"\n🔎 Valores únicos em '{col}':\n", df[col].value_counts(dropna=False))

# In[1.5]: Função tratar_variaveis_binarias(df)

def tratar_variaveis_binarias(df):
    """
    Padroniza variáveis binárias para valores 0 e 1, 
    converte 'falha_maquina' para int (target),
    e mantém as demais como object (para dummificação futura).
    """

    # Mapeamento completo de valores para 0 e 1
    mapa_binario = {
        'sim': 1, 'true': 1, '1': 1, 1: 1, True: 1, 'y': 1,
        'não': 0, 'nao': 0, 'n': 0, 'false': 0, '0': 0, 0: 0, False: 0, '-': 0
    }

    # Lista de colunas binárias
    colunas_binarias = [
        'falha_maquina',
        'fdf_falha_desgaste_ferramenta',
        'fdc_falha_dissipacao_calor',
        'fp_falha_potencia',
        'fte_falha_tensao_excessiva',
        'fa_falha_aleatoria'
    ]

    # Aplicar transformação
    for col in colunas_binarias:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(mapa_binario)
        )

    # Converter apenas a variável-alvo para int
    df['falha_maquina'] = df['falha_maquina'].astype(int)

    # As demais permanecem como object
    for col in colunas_binarias[1:]:
        df[col] = df[col].astype(int).astype('object')

    return df

# In[1.6]: Tratar variáveis binárias

df = tratar_variaveis_binarias(df)
    
# In[1.7]: Validar após o tratamento
# Lista de colunas binárias (repita fora da função)
colunas_binarias = [
    'falha_maquina',
    'fdf_falha_desgaste_ferramenta',
    'fdc_falha_dissipacao_calor',
    'fp_falha_potencia',
    'fte_falha_tensao_excessiva',
    'fa_falha_aleatoria'
]

# Validação
for col in colunas_binarias:
    print(f"{col}: {df[col].unique()}")


# In[1.8]: Tabela de frequências absolutas das variáveis qualitativas referentes
#aos atributos do dataset

df['tipo'].value_counts().sort_index()
df['falha_maquina'].value_counts().sort_index()
df['fdf_falha_desgaste_ferramenta'].value_counts().sort_index()
df['fdc_falha_dissipacao_calor'].value_counts().sort_index()
df['fp_falha_potencia'].value_counts().sort_index()
df['fte_falha_tensao_excessiva'].value_counts().sort_index()
df['fa_falha_aleatoria'].value_counts().sort_index()

# In[1.9]: Função dummizar_variaveis(df, colunas)
def dummizar_variaveis(df, colunas):
    """
    Aplica pd.get_dummies às colunas categóricas especificadas,
    usando drop_first=True para evitar multicolinearidade
    e dtype=int para garantir compatibilidade com modelos estatísticos.

    Parâmetros:
        df (DataFrame): O DataFrame original.
        colunas (list): Lista de colunas categóricas a dummificar.

    Retorna:
        DataFrame com as colunas dummificadas.
    """
    df_dummies = pd.get_dummies(df, columns=colunas, drop_first=True, dtype=int)
    return df_dummies

# In[1.10]: Dummizando as variáveis
colunas_dummies = [
    'tipo',
    'fdf_falha_desgaste_ferramenta',
    'fdc_falha_dissipacao_calor',
    'fp_falha_potencia',
    'fte_falha_tensao_excessiva',
    'fa_falha_aleatoria'
]

df_dummies = dummizar_variaveis(df, colunas_dummies)

# In[1.11]: Estimação do modelo logístico binário para falha em máquinas

# Definindo a fórmula com base nas variáveis do seu dataset dummificado
lista_colunas = list(df_dummies.drop(columns=['id', 'id_produto', 'falha_maquina'], errors='ignore').columns)

# Criando a fórmula no formato patsy
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = 'falha_maquina ~ ' + formula_dummies_modelo
print("📌 Fórmula utilizada:", formula_dummies_modelo)

# Estimando o modelo logístico binário

modelo_falha = smf.glm(formula=formula_dummies_modelo,
                       data=df_dummies,
                       family=sm.families.Binomial()).fit()

# Sumário dos parâmetros
modelo_falha.summary()

# In[1.12]: Procedimento Stepwise

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

from statstests.process import stepwise

#Estimação do modelo por meio do procedimento Stepwise
step_modelo_falha = stepwise(modelo_falha, pvalue_limit=0.05)

# In[1.13]: Após o Stepwise: avaliar o modelo refinado
# Parâmetros do novo modelo
step_modelo_falha.summary()

# Adicionar predições à base
df_dummies['phat'] = step_modelo_falha.predict()

# In[1.14]: Construção de função para a definição da matriz de confusão

from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.title(f'Matriz de Confusão (cutoff = {cutoff})')
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores

# In[1.15]: Construção da matriz de confusão
# Matriz de confusão para cutoff = 0.3
matriz_confusao(observado=df_dummies['falha_maquina'],
                predicts=df_dummies['phat'],
                cutoff=0.30)

# Matriz de confusão para cutoff = 0.5
matriz_confusao(observado=df_dummies['falha_maquina'],
                predicts=df_dummies['phat'],
                cutoff=0.5)

# Matriz de confusão para cutoff = 0.7
matriz_confusao(observado=df_dummies['falha_maquina'],
                predicts=df_dummies['phat'],
                cutoff=0.7)

# In[1.16]: Igualando critérios de especificidade e de sensitividade

# Tentaremos estabelecer um critério que iguale a probabilidade de
#acerto das máquinas que apresentaram falha (sensitividade) e a probabilidade de
#acerto daquelas que não apresentaram falha(especificidade).


def espec_sens(observado,predicts):
    
    # adicionar objeto com os valores dos predicts
    values = predicts.values
    
    # range dos cutoffs a serem analisados em steps de 0.01
    cutoffs = np.arange(0,1.01,0.01)
    
    # Listas que receberão os resultados de especificidade e sensitividade
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        
        predicao_binaria = []
        
        # Definindo resultado binário de acordo com o predict
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
                
        # Cálculo da sensitividade e especificidade no cutoff
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        # Adicionar valores nas listas
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
        
    # Criar dataframe com os resultados nos seus respectivos cutoffs
    resultado = pd.DataFrame({'cutoffs':cutoffs,'sensitividade':lista_sensitividade,'especificidade':lista_especificidade})
    return resultado

# In[1.17]: Até o momento, foram extraídos 3 vetores: 'sensitividade',
#'especificidade' e 'cutoffs'. Assim, criamos um dataframe que contém
#os vetores mencionados

dados_plotagem = espec_sens(observado = df_dummies['falha_maquina'],
                            predicts = df_dummies['phat'])
dados_plotagem

# In[1.18]: Plotagem de um gráfico que mostra a variação da especificidade e da
#sensitividade em função do cutoff

plt.figure(figsize=(15,10))
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.sensitividade, marker='o',
         color='indigo', markersize=8)
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.especificidade, marker='o',
         color='limegreen', markersize=8)
plt.xlabel('Cuttoff', fontsize=20)
plt.ylabel('Sensitividade / Especificidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.legend(['Sensitividade', 'Especificidade'], fontsize=20)
plt.show()

# In[1.19]: Construção da curva ROC

from sklearn.metrics import roc_curve, auc

# Função 'roc_curve' do pacote 'metrics' do sklearn

fpr, tpr, thresholds =roc_curve(df_dummies['falha_maquina'],
                                df_dummies['phat'])
roc_auc = auc(fpr, tpr)

# Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

# Plotando a curva ROC
plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) +
          ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

# In[1.20]: Criando um novo exemplo com os mesmos preditores do modelo final
novo_exemplo = pd.DataFrame({
    'velocidade_rotacional': [1500],
    'torque': [45],
    'desgaste_da_ferramenta': [0.003],
    'fdc_falha_dissipacao_calor_1': [1],   # 1 se falha ocorreu, 0 se não
    'fte_falha_tensao_excessiva_1': [0]    # 1 se falha ocorreu, 0 se não
})

# Adiciona a constante (intercepto)
novo_exemplo = sm.add_constant(novo_exemplo, has_constant='add')

# Predição da probabilidade de falha
prob_falha = step_modelo_falha.predict(novo_exemplo)

# Exibir resultado
print(f"🔧 Probabilidade estimada de falha: {prob_falha.values[0]:.4f}")

