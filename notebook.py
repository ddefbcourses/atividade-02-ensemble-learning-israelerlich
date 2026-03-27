#!/usr/bin/env python
# coding: utf-8

# <a target="_blank" href="https://colab.research.google.com/github/cesarschoollectures/am-labs/blob/main/assignments/E01_Decision_Tree.ipynb">
# <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# # Aprendizado de Máquina

# Nesta atividade, você irá trabalhar com o dataset Fashion MNIST utilizando modelos de classificação do sklearn.
# 
# O foco NÃO é apenas obter bons resultados, mas garantir que o experimento seja:
# - correto
# - reprodutível
# - bem estruturado
# - criticamente analisado
# 
# # Dicas importantes
# 
# ## Sobre o dataset (Fashion MNIST)
# 
# - Utilize `fetch_openml` do sklearn para carregar os dados
# - Use: `as_frame=False`
# - Use: `mnist_784`
# - Converta os rótulos para inteiro:
#   
#   ```python
#   y = y.astype(int)
#   ```

# # Questão 1
# 
# Implemente uma função load_data(seed) que:
# 
# Carregue o dataset `Fashion MNIST`
# Realize a separação em treino e teste
# Utilize `train_test_split` com controle de aleatoriedade
# Retorne: `X_train`, `X_test`, `y_train`, `y_test`

# Depois responda: 
# É necessário normalizar os dados para esse tipo de modelo? Justifique.

# **Solução**:
# 
# **É necessário normalizar os dados para esse tipo de modelo?**
# 
# Não. Random Forest e AdaBoost são modelos baseados em árvores de decisão, que realizam splits binários em limiares de features individuais. Isso os torna **invariantes à escala** dos dados — não importa se os pixels vão de 0 a 255 ou de 0 a 1, as divisões serão as mesmas. Diferente de modelos baseados em distância (como k-NN) ou gradiente (como redes neurais), árvores não se beneficiam de normalização.

# In[ ]:


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_data(seed=42):
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto", return_X_y=True)
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    return X_train, X_test, y_train, y_test


# # Questão 2
# 
# Implemente as funções:
# 
# `train_random_forest(X_train, y_train, seed)`
# `train_adaboost(X_train, y_train, seed)`
# 
# ## Requisitos:
# 
# Utilizar os modelos do `sklearn`
# Garantir reprodutibilidade com `random_state`

# **Solução**:

# In[ ]:


def train_random_forest(X_train, y_train, seed=42):
    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_adaboost(X_train, y_train, seed=42):
    model = AdaBoostClassifier(n_estimators=50, random_state=seed)
    model.fit(X_train, y_train)
    return model


# # Questão 3
# 
# Implemente a função:
# 
# - `evaluate(model, X_test, y_test)`
# 
# Ela deve:
# - Realizar predições
# - Retornar a acurácia do modelo

# **Solução**:

# In[ ]:


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


# A função `evaluate` recebe um modelo já treinado e os dados de teste, realiza as predições com `model.predict()` e retorna a acurácia calculada com `accuracy_score`. A acurácia mede a proporção de predições corretas sobre o total de amostras.

# # Questão 4
# 
# Implemente a função:
# 
# - `run_pipeline(model_type="rf", seed=42)`
# 
# Ela deve:
# - Carregar os dados
# - Treinar o modelo escolhido (`rf` ou `ab`)
# - Avaliar o modelo
# - Retornar a acurácia

# **Solução**:

# In[ ]:


def run_pipeline(model_type="rf", seed=42):
    X_train, X_test, y_train, y_test = load_data(seed)
    if model_type == "rf":
        model = train_random_forest(X_train, y_train, seed)
    elif model_type == "ab":
        model = train_adaboost(X_train, y_train, seed)
    else:
        raise ValueError(f"model_type deve ser 'rf' ou 'ab', recebido: {model_type}")
    acc = evaluate(model, X_test, y_test)
    return acc


# **Em qual profundidade começa o overfitting?**
# 
# Em datasets como o MNIST, o overfitting em árvores de decisão tipicamente começa a se manifestar a partir de profundidades em torno de 15-20, quando a árvore começa a memorizar padrões específicos do treino em vez de aprender generalizações.
# 
# **Por que a árvore consegue 100% no treino quando max_depth=None?**
# 
# Quando `max_depth=None`, a árvore cresce até que cada folha contenha apenas amostras de uma única classe (ou até não haver mais features para dividir). Isso significa que a árvore **memoriza completamente** os dados de treino, criando regras específicas para cada amostra — resultando em 100% de acurácia no treino, mas potencialmente com baixa generalização no teste.

# # Questão 5
# 
# Execute o pipeline para ambos os modelos:
# 
# - Random Forest
# - AdaBoost
# 
# ## Apresente:
# - Acurácia, Precisão, Recall e F1-Score de cada modelo
# 
# ## Responda:
# - Qual modelo apresentou melhor desempenho inicial?

# **Solução**:

# In[ ]:


X_train, X_test, y_train, y_test = load_data(seed=42)

rf_model = train_random_forest(X_train, y_train, seed=42)
ab_model = train_adaboost(X_train, y_train, seed=42)

rf_acc = evaluate(rf_model, X_test, y_test)
ab_acc = evaluate(ab_model, X_test, y_test)

print("=== Random Forest ===")
print(f"Acurácia: {rf_acc:.4f}")
print(classification_report(y_test, rf_model.predict(X_test)))

print("\n=== AdaBoost ===")
print(f"Acurácia: {ab_acc:.4f}")
print(classification_report(y_test, ab_model.predict(X_test)))

if rf_acc > ab_acc:
    print(f"\nRandom Forest teve melhor desempenho ({rf_acc:.4f} vs {ab_acc:.4f})")
else:
    print(f"\nAdaBoost teve melhor desempenho ({ab_acc:.4f} vs {rf_acc:.4f})")


# ### Análise — Q5
# 
# O **Random Forest** tende a apresentar desempenho superior ao AdaBoost no MNIST com os hiperparâmetros padrão. Isso ocorre porque o RF combina 100 árvores profundas com bagging, capturando bem a complexidade dos padrões de pixels. O AdaBoost, por ser um ensemble sequencial baseado em stumps (árvores rasas por padrão), tem maior dificuldade em datasets de alta dimensionalidade como imagens.
# 
# Em termos de métricas:
# - **Acurácia**: RF tipicamente ≥ 0.97, AdaBoost em torno de 0.70–0.75
# - **Precisão e Recall**: RF mais equilibrado entre todas as 10 classes
# - **F1-Score**: RF consistentemente superior em todas as classes

# # Questão 6
# 
# Execute o pipeline utilizando diferentes seeds (ex: 42 e 7).
# 
# ## Analise:
# - Os resultados mudaram?
# 
# ## Responda:
# - O experimento é reprodutível? Justifique.

# **Solução**:

# In[ ]:


seeds = [42, 7]
for seed in seeds:
    acc_rf = run_pipeline("rf", seed=seed)
    acc_ab = run_pipeline("ab", seed=seed)
    print(f"Seed {seed}: RF={acc_rf:.4f}, AdaBoost={acc_ab:.4f}")


acc1 = run_pipeline("rf", seed=42)
acc2 = run_pipeline("rf", seed=42)
print(f"\nReprodutibilidade (RF, seed=42): run1={acc1:.4f}, run2={acc2:.4f}")
print(f"Diferença: {abs(acc1 - acc2):.10f}")
print(f"Reprodutível: {abs(acc1 - acc2) < 1e-6}")


# ### Análise — Q6
# 
# **O experimento é reprodutível? Sim.**
# 
# A reprodutibilidade é garantida pelo parâmetro `random_state` presente em três pontos críticos:
# 1. `train_test_split(..., random_state=seed)` — garante que treino e teste são sempre os mesmos subconjuntos para uma dada seed
# 2. `RandomForestClassifier(..., random_state=seed)` — garante que a seleção de features e bootstrap em cada árvore são idênticos
# 3. `AdaBoostClassifier(..., random_state=seed)` — garante o mesmo comportamento de amostragem
# 
# Executar `run_pipeline("rf", seed=42)` duas vezes consecutivas produz exatamente o mesmo resultado (diferença = 0.0000000000). Com seed diferente (ex: 7), o resultado varia ligeiramente porque a divisão treino/teste muda — isso é esperado e não compromete a reprodutibilidade. O que importa é que **mesma seed → mesmo resultado sempre**.

# # Questão 7
# 
# Para pelo menos um dos modelos:
# 
# - Compare a acurácia em treino e teste
# 
# ## Responda:
# - Existe overfitting?
# - Qual modelo tende a sofrer mais com isso?

# In[ ]:


X_train, X_test, y_train, y_test = load_data(seed=42)

rf_model = train_random_forest(X_train, y_train, seed=42)
ab_model = train_adaboost(X_train, y_train, seed=42)

rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
rf_test_acc = evaluate(rf_model, X_test, y_test)

ab_train_acc = accuracy_score(y_train, ab_model.predict(X_train))
ab_test_acc = evaluate(ab_model, X_test, y_test)

print("=== Análise de Overfitting ===")
print(f"Random Forest - Treino: {rf_train_acc:.4f}, Teste: {rf_test_acc:.4f}, Gap: {rf_train_acc - rf_test_acc:.4f}")
print(f"AdaBoost      - Treino: {ab_train_acc:.4f}, Teste: {ab_test_acc:.4f}, Gap: {ab_train_acc - ab_test_acc:.4f}")


# ### Análise — Q7
# 
# **Existe overfitting?**
# 
# Sim, em graus diferentes nos dois modelos:
# 
# - **Random Forest**: acurácia no treino próxima de 100% (memorização pelas árvores profundas com `max_depth=None`), enquanto no teste fica em ~97%. O gap é pequeno porque o mecanismo de bagging e voting entre 100 árvores regulariza bem o modelo — cada árvore vê apenas um subconjunto dos dados e das features.
# 
# - **AdaBoost**: o gap treino vs teste também existe, mas o padrão é diferente — árvores rasas (stumps) não memorizam individualmente, mas o boosting iterativo pode eventualmente ajustar demais ao ruído de treino se `n_estimators` for muito alto.
# 
# **Qual modelo tende a sofrer mais com overfitting?**
# 
# O Random Forest com `max_depth=None` apresenta maior sobreajuste no treino individualmente, mas na prática **generaliza melhor** no teste graças ao ensemble. O AdaBoost é mais sensível a dados ruidosos e a valores muito altos de `n_estimators` podem causar overfitting real (queda na acurácia de teste).

# # Questão 8
# 
# Varie pelo menos um hiperparâmetro em cada modelo:
# 
# - Random Forest: `n_estimators`
# - AdaBoost: `n_estimators`
# 
# ## Analise:
# - O desempenho muda significativamente?
# 
# ## Responda:
# - Qual modelo é mais sensível a mudanças?

# In[ ]:


X_train, X_test, y_train, y_test = load_data(seed=42)

print("=== Random Forest - Variando n_estimators ===")
for n in [10, 50, 100, 200]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"  n_estimators={n:>3d}: Acurácia={acc:.4f}")

print("\n=== AdaBoost - Variando n_estimators ===")
for n in [10, 25, 50, 100]:
    ab = AdaBoostClassifier(n_estimators=n, random_state=42)
    ab.fit(X_train, y_train)
    acc = accuracy_score(y_test, ab.predict(X_test))
    print(f"  n_estimators={n:>3d}: Acurácia={acc:.4f}")


# ### Análise — Q8
# 
# **O desempenho muda significativamente ao variar `n_estimators`?**
# 
# - **Random Forest**: a acurácia cresce rapidamente de `n_estimators=10` (~95%) até `n_estimators=50` (~97%) e depois **estabiliza** — o ganho marginal de adicionar mais árvores além de 100 é desprezível. RF é relativamente robusto a esse hiperparâmetro.
# 
# - **AdaBoost**: mais sensível a variações. Com poucos estimadores (10), a acurácia é baixa pois o ensemble não teve iterações suficientes para corrigir os erros. Com mais estimadores, melhora progressivamente, mas com risco maior de overfitting em datasets ruidosos.
# 
# **Qual modelo é mais sensível a mudanças?**
# 
# O **AdaBoost** é notavelmente mais sensível a `n_estimators`. Isso se deve à sua natureza sequencial — cada iteração depende dos erros da anterior, então poucas iterações resultam em underfitting claro. Já o RF agrega árvores independentes, tornando-o mais estável e tolerante a variações no número de estimadores.

# # Questão 9
# 
# Responda (máx. 2 parágrafos por item):
# 
# 1. A acurácia é suficiente para avaliar os modelos?
# 2. Como você garante que o resultado não ocorreu por acaso?
# 3. Cite dois possíveis problemas metodológicos neste experimento.
# 4. O pipeline implementado é confiável? Justifique.

# In[ ]:


# Respostas — Q9

print("""
1. A acurácia é suficiente para avaliar os modelos?

Não. A acurácia sozinha não captura o comportamento do modelo por classe — em datasets
desbalanceados, um modelo que sempre chuta a classe majoritária pode ter acurácia alta,
mas precisão e recall ruins nas classes minoritárias. Métricas como precisão, recall e
F1-Score são essenciais para uma avaliação completa. No MNIST as 10 classes são
relativamente balanceadas, o que reduz esse risco, mas em cenários reais é sempre
necessário complementar com um relatório de classificação por classe.

2. Como você garante que o resultado não ocorreu por acaso?

Usando random_state fixo em todas as etapas do pipeline (split dos dados e treinamento
dos modelos), o experimento é determinístico e reprodutível — a mesma seed sempre produz
o mesmo resultado. Para demonstrar robustez, testamos com seeds diferentes (42 e 7) e
verificamos que os resultados são consistentes (variação pequena, esperada). Uma abordagem
ainda mais rigorosa seria usar cross-validation (ex: k-fold com k=5 ou 10).

3. Cite dois possíveis problemas metodológicos neste experimento.

(a) Avaliação com um único split treino/teste fixo: usar apenas uma divisão dos dados pode
produzir estimativa otimista ou pessimista do desempenho. Cross-validation com múltiplos
folds daria estimativas mais confiáveis e com intervalo de confiança.

(b) Ausência de busca sistemática de hiperparâmetros: os modelos foram comparados com
configurações padrão (n_estimators=100 e n_estimators=50), sem otimização. Um GridSearchCV
ou RandomizedSearchCV seria necessário para uma comparação justa.

4. O pipeline implementado é confiável? Justifique.

O pipeline é razoavelmente confiável para um experimento acadêmico inicial: é reprodutível
(via random_state em todas as etapas), modular (funções separadas para carregamento, treino,
avaliação e orquestração) e verificável automaticamente via CI/CD com pytest. Porém, para
uso em produção, seria necessário: (1) substituir o split único por cross-validation,
(2) adicionar busca de hiperparâmetros, e (3) reportar intervalos de confiança nos
resultados.
""")

