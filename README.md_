# MVP Machine Learning & Analytics - Pressão de Vapor da Nafta

**Autor:** Fabiano da Mata Almeida  
**Matrícula:** 4052025000952  
**Curso:** Pós-graduação em Ciência de Dados e Analytics - PUC-RJ  
**Dataset:** Pressão de Vapor da Nafta (descaracterizado)

[![Abrir no Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fdamata/pucrj-machinelearning-mvp-ml/blob/main/mvp_sprint_02_fma_2025.ipynb)

## Descrição do Projeto

Este MVP apresenta um **workflow de machine learning para algoritmos de regressão** focado na predição de propriedades físico-químicas em processos industriais de refino de petróleo.

## 🔄 Principais Etapas do Notebook

### 1. Análise Exploratória e Preparação
- **Carregamento e inspeção:** Análise inicial do dataset com 906 registros e 12 variáveis
- **Análise visual:** Visualização de séries temporais e distribuição das variáveis
- **Avaliação de normalidade:** Teste de Kolmogorov-Smirnov nas variáveis 

### 2. Engenharia de Features e Pré-processamento
- **Análise de multicolinearidade:** Cálculo de VIF (Variance Inflation Factor)
- **Remoção de variáveis:** Eliminação de features com alta correlação (VIF > 5)
- **Divisão dos dados:** Split em conjuntos de treino (75%) e teste (25%)

### 3. Modelagem e Otimização
- **Comparativo de normalização:** Standard, MinMax e PowerTransformer (Yeo-Johnson)
- **Baseline:** Implementação de modelo linear (Elastic Net) como referência
- **Otimização bayesiana:** BayesSearchCV para otimização de hiperparâmetros
- **Validação cruzada:** K-fold com 10 folds para avaliação robusta
- **Algoritmos testados:** Modelos lineares, modelos baseados em árvores e ensemble

### 4. Avaliação e Implantação
- **Métricas comparativas:** R², MAE, MSE e RMSE entre modelos e scalers
- **Visualização de resultados:** Boxplots de performance por algoritmo/scaler
- **Análise de hiperparâmetros:** Impacto dos parâmetros na convergência
- **Persistência do modelo:** Salvamento via joblib para deployment
- **Demonstração de uso:** Exemplo de carregamento e predição

## 🚀 Como Executar o Projeto

### Instruções
1. Clone este repositório
2. Abra o notebook `mvp_sprint_02_fma_2025.ipynb` no Jupyter Notebook ou Google Colab
3. Execute as células sequencialmente para reproduzir a análise completa
4. Alternativamente, utilize o modelo já treinado carregando `modelo_final_pv_nafta.joblib`

```python
import joblib
# Carregar o modelo
modelo = joblib.load('modelo_final_pv_nafta.joblib')
# Fazer predições
predicoes = modelo.predict(dados_novos)