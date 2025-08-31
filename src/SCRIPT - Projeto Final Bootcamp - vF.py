# ===============================================================
# RESIDÊNCIA EM CIÊNCIA DE DADOS E INTELIGÊNCIA ARTIFICIAL
# Projeto: Regressão Logística Binária (GLM Binomial + Stepwise)
# Arquivos esperados:
#   - maquinas_train.csv (treino)
#   - maquinas_test.csv  (teste)
# ===============================================================

# --------------------- Instalação (opcional) -------------------
!pip install pandas numpy seaborn matplotlib plotly scipy statsmodels scikit-learn statstests unidecode

# --------------------------- Imports ---------------------------
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import unidecode

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statstests.process import stepwise

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    recall_score, accuracy_score, roc_curve, auc
)

# --------------------- Funções utilitárias ---------------------
def padroniza_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[\(\)]', '', regex=True)
        .map(unidecode.unidecode)
    )
    return df

def tratar_e_explorar_variaveis_numericas(df: pd.DataFrame, tratar_nulos='drop') -> pd.DataFrame:
    df = df.copy()
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)

    # Tratamento de nulos
    if tratar_nulos == 'drop':
        df = df.dropna()
    elif tratar_nulos == 'fill':
        df.fillna(df.median(numeric_only=True), inplace=True)

    # Regras básicas de sanidade (iguais às do treino)
    if 'temperatura_ar' in df.columns:
        df = df[df['temperatura_ar'] > 270]
    if 'temperatura_processo' in df.columns:
        df = df[df['temperatura_processo'] > 273]
    if 'velocidade_rotacional' in df.columns:
        df = df[df['velocidade_rotacional'] >= 0]
    if 'desgaste_da_ferramenta' in df.columns:
        df = df[df['desgaste_da_ferramenta'] >= 0]
    return df

def tratar_variaveis_binarias(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapa = {
        'sim': 1, 'true': 1, '1': 1, 1: 1, True: 1, 'y': 1,
        'nao': 0, 'não': 0, 'n': 0, 'false': 0, '0': 0, 0: 0, False: 0, '-': 0
    }
    cols_bin = [
        'falha_maquina',
        'fdf_falha_desgaste_ferramenta',
        'fdc_falha_dissipacao_calor',
        'fp_falha_potencia',
        'fte_falha_tensao_excessiva',
        'fa_falha_aleatoria'
    ]
    for c in cols_bin:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().replace(mapa)

    if 'falha_maquina' in df.columns:
        df['falha_maquina'] = df['falha_maquina'].astype(int)

    # Demais como object (para dummificação)
    for c in cols_bin[1:]:
        if c in df.columns:
            df[c] = df[c].astype(int).astype('object')
    return df

def dummizar_variaveis(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
    colunas_existentes = [c for c in colunas if c in df.columns]
    if len(colunas_existentes) == 0:
        return df.copy()
    return pd.get_dummies(df, columns=colunas_existentes, drop_first=True, dtype=int)

def matriz_confusao(predicts: pd.Series, observado: pd.Series, cutoff: float = 0.5) -> pd.DataFrame:
    pred_bin = (predicts >= cutoff).astype(int)
    cm = confusion_matrix(observado, pred_bin)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Matriz de Confusão (cutoff = {cutoff})')
    plt.xlabel('Predito'); plt.ylabel('Real')
    plt.gca().invert_xaxis(); plt.gca().invert_yaxis()
    plt.show()

    sens = recall_score(observado, pred_bin, pos_label=1)
    esp  = recall_score(observado, pred_bin, pos_label=0)
    acc  = accuracy_score(observado, pred_bin)
    return pd.DataFrame({'Cutoff':[cutoff], 'Sensitividade':[sens], 'Especificidade':[esp], 'Acurácia':[acc]})

def curva_roc(predicts: pd.Series, observado: pd.Series, titulo='Curva ROC'):
    fpr, tpr, thr = roc_curve(observado, predicts)
    roc_auc = auc(fpr, tpr)
    gini = 2*roc_auc - 1
    plt.figure(figsize=(10,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.3f} | Gini={gini:.3f}')
    plt.plot([0,1],[0,1], color='gray', linestyle='--')
    plt.title(titulo); plt.xlabel('1 - Especificidade'); plt.ylabel('Sensitividade')
    plt.legend(loc='lower right'); plt.grid(True); plt.show()
    return roc_auc, gini

# remove o wrapper Q('...') que pode aparecer nos nomes do modelo
def strip_q(name: str) -> str:
    if isinstance(name, str) and name.startswith("Q('") and name.endswith("')"):
        return name[3:-2]
    return name

# ----------------------------- Treino --------------------------
# 1) Carregar treino
df_train = pd.read_csv('maquinas_train.csv', delimiter=',')
df_train = padroniza_colunas(df_train)
df_train = tratar_e_explorar_variaveis_numericas(df_train, tratar_nulos='drop')
df_train = tratar_variaveis_binarias(df_train)

# 2) Dummificar (se existirem categóricas)
colunas_dummies = [
    'tipo',
    'fdf_falha_desgaste_ferramenta',
    'fdc_falha_dissipacao_calor',
    'fp_falha_potencia',
    'fte_falha_tensao_excessiva',
    'fa_falha_aleatoria'
]
df_tr = dummizar_variaveis(df_train, colunas_dummies)

# 3) Fórmula GLM Binomial (usa tudo exceto id's e alvo)
cols_modelo = list(df_tr.drop(columns=['falha_maquina','id','id_produto'], errors='ignore').columns)
formula = 'falha_maquina ~ ' + ' + '.join(cols_modelo) if len(cols_modelo) > 0 else 'falha_maquina ~ 1'
print('Fórmula inicial:', formula)

# 4) Estimar GLM Binomial
modelo_glm = smf.glm(formula=formula, data=df_tr, family=sm.families.Binomial()).fit()
print(modelo_glm.summary())

# 5) Stepwise (remove variáveis não significativas)
modelo_sw = stepwise(modelo_glm, pvalue_limit=0.05)
print(modelo_sw.summary())

# 6) Predições no TREINO (corrigido para Patsy: passar DF completo)
#    - obter nomes finais (sem Intercept e sem Q(...))
terms_final = [strip_q(n) for n in modelo_sw.model.exog_names if n != 'Intercept']
#    - garanta que todas as colunas necessárias existam (se faltar dummy, cria 0)
for c in terms_final:
    if c not in df_tr.columns:
        df_tr[c] = 0
#    - predict com DF completo
df_tr['phat'] = modelo_sw.predict(df_tr)

# 7) Sensibilidade e Especificidade por cutoff (TREINO) ======

from sklearn.metrics import recall_score
import numpy as np

def espec_sens(observado, predicts):
    # vetor de cutoffs de 0.00 a 1.00 (step = 0.01)
    cutoffs = np.arange(0, 1.01, 0.01)

    lista_sensitividade = []
    lista_especificidade = []

    values = predicts.values  # mais rápido para iterar
    for cutoff in cutoffs:
        pred_bin = (values >= cutoff).astype(int)
        sens = recall_score(observado, pred_bin, pos_label=1)  # TPR
        esp  = recall_score(observado, pred_bin, pos_label=0)  # TNR
        lista_sensitividade.append(sens)
        lista_especificidade.append(esp)

    return pd.DataFrame({
        'cutoffs': cutoffs,
        'sensitividade': lista_sensitividade,
        'especificidade': lista_especificidade
    })

# gerar a tabela com os valores (treino)
dados_plotagem = espec_sens(observado=df_tr['falha_maquina'],
                            predicts=df_tr['phat'])

# plotar o gráfico
plt.figure(figsize=(15, 10))
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.plot(dados_plotagem.cutoffs, dados_plotagem.sensitividade,
             marker='o', color='indigo', markersize=5, label='Sensitividade')
    plt.plot(dados_plotagem.cutoffs, dados_plotagem.especificidade,
             marker='o', color='limegreen', markersize=5, label='Especificidade')

plt.xlabel('Cutoff', fontsize=20)
plt.ylabel('Sensitividade / Especificidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

# 8) Avaliação no TREINO (opcional)
if 'falha_maquina' in df_tr.columns:
    print(matriz_confusao(df_tr['phat'], df_tr['falha_maquina'], cutoff=0.5))
    curva_roc(df_tr['phat'], df_tr['falha_maquina'], titulo='Curva ROC - Treino')

#%% ------------------------------ Teste --------------------------
# 8) Carregar TESTE
df_test = pd.read_csv('maquinas_test.csv', delimiter=',')
df_test = padroniza_colunas(df_test)
df_test = tratar_e_explorar_variaveis_numericas(df_test, tratar_nulos='fill')
df_test = tratar_variaveis_binarias(df_test)  # se não tiver categóricas, ok

# 9) Dummificar TESTE (se não houver categóricas, não altera)
df_te = dummizar_variaveis(df_test, colunas_dummies)

# 10) Garantir TODAS as colunas do modelo no TESTE (se faltar, cria 0)
for c in terms_final:
    if c not in df_te.columns:
        df_te[c] = 0

# 11) Predições no TESTE (corrigido para Patsy: passar DF completo)
df_te['phat'] = modelo_sw.predict(df_te)

# 12) Mostrar amostra das predições
cols_show = ['phat']
if 'falha_maquina' in df_te.columns:
    cols_show = ['falha_maquina', 'phat']
print(df_te[cols_show].head())

# 13) Avaliar no TESTE (se houver rótulo)
if 'falha_maquina' in df_te.columns:
    print(matriz_confusao(df_te['phat'], df_te['falha_maquina'], cutoff=0.5))
    curva_roc(df_te['phat'], df_te['falha_maquina'], titulo='Curva ROC - Teste')

# 14) Exportar predições do TESTE
df_te.to_csv('predicoes_maquinas_test.csv', index=False)
print('Arquivo salvo: predicoes_maquinas_test.csv')

#%% --------------------- Exemplo manual (opcional) ---------------
# Criar dicionário SOMENTE com as variáveis finais do modelo (terms_final)
novo_exemplo = pd.DataFrame({
    'velocidade_rotacional': [1500],
    'torque': [45],
    'desgaste_da_ferramenta': [0.003],
    'fdc_falha_dissipacao_calor_1': [1],
    'fte_falha_tensao_excessiva_1': [0]
})

# garantir todas as colunas finais do modelo
for c in terms_final:
    if c not in novo_exemplo.columns:
        novo_exemplo[c] = 0

# reordenar as colunas na mesma ordem usada no modelo
novo_exemplo = novo_exemplo.reindex(columns=terms_final)

# predição
print('Prob. falha (exemplo):', float(modelo_sw.predict(novo_exemplo)))
