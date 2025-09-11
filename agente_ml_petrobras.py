"""
AGENTE INTELIGENTE DE MACHINE LEARNING - PETROBRAS MVP
=====================================================

Agente que automatiza todo o fluxo de ML do projeto, desde o carregamento
dos dados até a entrega do modelo final, mantendo compatibilidade total com
o código existente.

Autor: GitHub Copilot
Projeto: MVP Machine Learning & Analytics - Pressão de Vapor da Nafta
"""

import os
import sys
import json
import time
import joblib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Imports do projeto original (mantidos para compatibilidade)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate

# ML imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Importar XGBoost e LightGBM se disponíveis
try:
    from xgboost import XGBRegressor
    import xgboost as xgb
except ImportError:
    print("XGBoost não disponível")
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
except ImportError:
    print("LightGBM não disponível")
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    print("CatBoost não disponível")
    CatBoostRegressor = None

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import DeltaYStopper
from skopt.plots import plot_objective, plot_convergence

warnings.filterwarnings("ignore")


@dataclass
class ConfiguracaoProjeto:
    """Configurações centralizadas do projeto"""

    problema_tipo: str = "regressao"
    seed: int = 42
    split_teste: float = 0.3
    cv_folds: int = 10
    metrica_otimizar: str = "rmse"
    n_iter_otimizacao: int = 30
    url_dataset: str = "dataset_pv_nafta_ml.xlsx"
    arquivo_config_algos: str = "algo_configs.json"
    diretorio_cache: str = "cache_agente_ml"
    arquivo_modelo_final: str = "modelo_final_pv_nafta_agente.joblib"


# Funções de suporte replicadas do notebook original
def _deserialize_space(space_dict):
    """Deserializa espaços de busca do JSON"""
    if isinstance(space_dict, dict):
        space_type = space_dict.get("_type")
        if space_type == "Real":
            return Real(
                space_dict["low"],
                space_dict["high"],
                prior=space_dict.get("prior", "uniform"),
            )
        elif space_type == "Integer":
            return Integer(
                space_dict["low"],
                space_dict["high"],
                prior=space_dict.get("prior", "uniform"),
            )
        elif space_type == "Categorical":
            categories = space_dict["categories"]
            # Converter listas para tuplas para hidden_layer_sizes
            normalized_cats = []
            for cat in categories:
                if isinstance(cat, list):
                    normalized_cats.append(tuple(cat))
                else:
                    normalized_cats.append(cat)
            return Categorical(normalized_cats)
    return space_dict


def get_regression_metric(metric_name):
    """Retorna função de métrica para regressão"""
    from sklearn.metrics import make_scorer

    if metric_name.lower() == "rmse":
        return make_scorer(
            lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))
        )
    elif metric_name.lower() == "mse":
        return make_scorer(mean_squared_error, greater_is_better=False)
    elif metric_name.lower() == "mae":
        return make_scorer(mean_absolute_error, greater_is_better=False)
    elif metric_name.lower() == "r2":
        return make_scorer(r2_score)
    else:
        return make_scorer(
            lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))
        )


def compute_metrics(y_true, y_pred):
    """Calcula métricas de regressão"""
    return {
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


# Funções de análise exploratória (replicadas do notebook)
def teste_n(df, column_name, alpha=0.05):
    """Teste de normalidade Kolmogorov-Smirnov"""
    from scipy.stats import kstest

    stat, p_valor = kstest(
        (df[column_name] - np.mean(df[column_name])) / np.std(df[column_name], ddof=1),
        "norm",
    )

    if p_valor > alpha:
        print(
            "A amostra parece vir de uma distribuição normal (não podemos rejeitar a hipótese nula) p-valor:",
            f"{p_valor:.5f}",
        )
    else:
        print(
            "A amostra não parece vir de uma distribuição normal (rejeitamos a hipótese nula) p-valor:",
            f"{p_valor:.5f}",
        )
    return float(stat), float(p_valor)


def calcula_corr(df):
    """Matriz de correlação com Plotly"""
    df_corr = df.dropna().corr().abs()

    fig = px.imshow(
        img=df_corr,
        color_continuous_scale="Viridis",
        width=900,
        height=900,
        text_auto=".2f",
    )

    fig.update_traces(textfont_size=12)
    fig.show()


def calcula_vif(df, target):
    """Calcula VIF para identificar multicolinearidade"""
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X = df.drop(columns=[target]) if target in df.columns else df.copy()
        X_with_const = sm.add_constant(X)
        vif = pd.DataFrame()
        vif["Variable"] = X_with_const.columns
        vif["VIF"] = [
            variance_inflation_factor(X_with_const.values, i)
            for i in range(X_with_const.shape[1])
        ]

        vif.set_index("Variable", inplace=True)
        print("\nVIF das variáveis (ordem decrescente):\n")
        print(
            vif.query("Variable != 'const'")
            .sort_values(by="VIF", ascending=False)
            .head(15)
            .T
        )
    except ImportError:
        print("Statsmodels não disponível para cálculo de VIF")


def plot_boxplot_pdf(df, n_cols=4):
    """Plot de distribuições das variáveis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_vars = len(numeric_cols)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []

    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor="black")
            axes[i].set_title(f"Distribuição: {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequência")

    # Remover eixos vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


class AgenteMLPetrobras:
    """
    Agente Inteligente de Machine Learning para o Projeto Petrobras

    Este agente automatiza todo o pipeline de ML:
    1. Carregamento e pré-processamento de dados
    2. Análise exploratória automatizada
    3. Seleção e otimização de modelos
    4. Avaliação e entrega do modelo final
    5. Geração de relatórios automáticos
    """

    def __init__(self, config: Optional[ConfiguracaoProjeto] = None):
        self.config = config or ConfiguracaoProjeto()
        self.cache_dir = Path(self.config.diretorio_cache)
        self.cache_dir.mkdir(exist_ok=True)

        # Estado do agente
        self.df_original = None
        self.df_processado = None
        self.target = None
        self.features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.resultados_otimizacao = {}
        self.melhor_modelo = None
        self.pipeline_final = None

        # Configurações de algoritmos e scalers
        self.algo_configs = {}
        self.scalers = [
            ("Minmax", MinMaxScaler()),
            ("Standard", StandardScaler()),
            ("YeoJohn", PowerTransformer(method="yeo-johnson")),
        ]

        self._log("🤖 Agente ML Petrobras inicializado")

    def _log(self, mensagem: str, nivel: str = "INFO"):
        """Sistema de logging do agente"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {nivel}: {mensagem}")

    def _salvar_cache(self, nome: str, dados: Any):
        """Salva dados no cache"""
        arquivo_cache = self.cache_dir / f"{nome}.joblib"
        joblib.dump(dados, arquivo_cache)
        self._log(f"💾 Dados salvos no cache: {nome}")

    def _carregar_cache(self, nome: str) -> Optional[Any]:
        """Carrega dados do cache"""
        arquivo_cache = self.cache_dir / f"{nome}.joblib"
        if arquivo_cache.exists():
            self._log(f"📁 Carregando do cache: {nome}")
            return joblib.load(arquivo_cache)
        return None

    def carregar_dados(self, forcar_reprocessamento: bool = False) -> pd.DataFrame:
        """
        Etapa 1: Carregamento e pré-processamento dos dados
        Aplica exatamente o mesmo processamento do notebook original
        """
        self._log("📊 Iniciando carregamento e processamento dos dados")

        # Verificar cache
        if not forcar_reprocessamento:
            dados_cache = self._carregar_cache("dados_processados")
            if dados_cache is not None:
                self.df_original, self.df_processado, self.target, self.features = (
                    dados_cache
                )
                self._log("✅ Dados carregados do cache")
                return self.df_processado

        # Carregamento original
        try:
            self.df_original = pd.read_excel(self.config.url_dataset)
            self._log(f"📈 Dataset carregado: {self.df_original.shape}")
        except Exception as e:
            self._log(f"❌ Erro ao carregar dataset: {e}", "ERROR")
            raise

        # Aplicar o processamento exato do notebook
        tags = self.df_original.columns.to_list()
        self.target = tags[0]  # primeira coluna é o target

        # Remoção baseada em VIF (igual ao notebook)
        df1 = self.df_original.drop(columns=["t_topo_nafta", "t_esup_nafta"])
        df2 = df1.drop(columns=["t_aque_nafta"])

        # Criação da variável derivada (razão de refluxo)
        df2["r_refl_nafta"] = df2["f_refl_nafta"] / df2["f_carg_nafta"]
        self.df_processado = df2.drop(columns=["f_refl_nafta"])

        # Definir features finais
        self.features = [
            col for col in self.df_processado.columns if col != self.target
        ]

        # Salvar no cache
        dados_cache = (self.df_original, self.df_processado, self.target, self.features)
        self._salvar_cache("dados_processados", dados_cache)

        self._log(f"✅ Dados processados: {self.df_processado.shape}")
        self._log(f"🎯 Target: {self.target}")
        self._log(f"🔢 Features: {len(self.features)}")

        return self.df_processado

    def executar_eda_automatizada(self, completa: bool = True):
        """
        Etapa 2: Análise Exploratória Automatizada
        Executa as mesmas análises do notebook de forma automatizada
        """
        self._log("🔍 Executando análise exploratória automatizada")

        if self.df_processado is None:
            self._log(
                "❌ Dados não carregados. Execute carregar_dados() primeiro", "ERROR"
            )
            return

        print("\n" + "=" * 80)
        print("RELATÓRIO DE ANÁLISE EXPLORATÓRIA AUTOMATIZADA")
        print("=" * 80)

        # Estatísticas básicas
        print(
            f"Dataset: {self.df_processado.shape[0]} observações, {self.df_processado.shape[1]} variáveis"
        )
        print(f"Target: {self.target}")
        print(f"Features: {len(self.features)}")

        with pd.option_context("display.float_format", "{:.2f}".format):
            print("\nEstatísticas Descritivas:")
            print(self.df_processado.describe().T)

        if completa:
            # Teste de normalidade
            print(f"\nTeste de Normalidade (target: {self.target}):")
            teste_n(self.df_processado, self.target)

            # Correlação
            print("\nMatriz de Correlação:")
            calcula_corr(self.df_processado)

            # VIF
            print("\nAnálise de VIF:")
            calcula_vif(self.df_processado, self.target)

            # Distribuições
            print("\nDistribuições das Variáveis:")
            plot_boxplot_pdf(self.df_processado, n_cols=4)

        print("=" * 80)
        self._log("✅ Análise exploratória concluída")

    def preparar_dados_ml(self):
        """
        Etapa 3: Preparação dos dados para ML
        Divisão treino/teste mantendo a mesma estratégia do notebook
        """
        self._log("🔄 Preparando dados para Machine Learning")

        if self.df_processado is None:
            self._log(
                "❌ Dados não processados. Execute carregar_dados() primeiro", "ERROR"
            )
            return

        # Separação X e y (igual ao notebook)
        X = self.df_processado.drop(self.target, axis=1)
        y = self.df_processado[self.target].copy()

        # Divisão treino/teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config.split_teste, random_state=self.config.seed
        )

        self._log(
            f"✅ Dados preparados - Treino: {self.X_train.shape}, Teste: {self.X_test.shape}"
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def carregar_configuracoes_algoritmos(self):
        """
        Etapa 4: Carregamento das configurações dos algoritmos
        Mantém compatibilidade total com o arquivo JSON do projeto
        """
        self._log("⚙️ Carregando configurações dos algoritmos")

        try:
            if not os.path.exists(self.config.arquivo_config_algos):
                self._log(
                    "❌ Arquivo de configuração de algoritmos não encontrado", "ERROR"
                )
                raise FileNotFoundError(
                    f"Arquivo {self.config.arquivo_config_algos} não encontrado"
                )

            with open(self.config.arquivo_config_algos, "r", encoding="utf-8") as f:
                self.algo_configs = json.load(f)

            self._log(
                f"✅ Configurações carregadas: {len(self.algo_configs)} algoritmos"
            )

            # Listar algoritmos disponíveis
            print("\nAlgoritmos Disponíveis:")
            for i, (key, cfg) in enumerate(self.algo_configs.items()):
                print(f"  {i+1}. {cfg.get('alias', key)} ({key})")

        except Exception as e:
            self._log(f"❌ Erro ao carregar configurações: {e}", "ERROR")
            raise

    def executar_otimizacao_completa(
        self,
        usar_cache: bool = True,
        algoritmos_selecionados: Optional[List[str]] = None,
    ):
        """
        Etapa 5: Otimização de Hiperparâmetros Completa
        Replica exatamente o processo do notebook com otimizações
        """
        self._log("🚀 Iniciando otimização de hiperparâmetros completa")

        # Verificar pré-requisitos
        if self.X_train is None:
            self._log(
                "❌ Dados não preparados. Execute preparar_dados_ml() primeiro", "ERROR"
            )
            return

        if not self.algo_configs:
            self.carregar_configuracoes_algoritmos()

        # Verificar cache
        cache_key = (
            f"otimizacao_{self.config.metrica_otimizar}_{self.config.n_iter_otimizacao}"
        )
        if usar_cache:
            resultados_cache = self._carregar_cache(cache_key)
            if resultados_cache is not None:
                self.resultados_otimizacao = resultados_cache
                self._log("✅ Resultados de otimização carregados do cache")
                return self.resultados_otimizacao

        # Algoritmos a otimizar
        algos_para_otimizar = algoritmos_selecionados or list(self.algo_configs.keys())

        # Configuração CV
        cv_splitter = KFold(
            n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.seed
        )
        delta_metric = 0.05 if self.config.metrica_otimizar in ["r2"] else 0.5

        print(
            f"\n🎯 Otimizando {len(algos_para_otimizar)} algoritmos com {len(self.scalers)} scalers"
        )
        print(
            f"📊 Métrica: {self.config.metrica_otimizar} | Iterações: {self.config.n_iter_otimizacao}"
        )
        print("=" * 80)

        resultados = {}
        opt_results = {}
        total_combinacoes = len(algos_para_otimizar) * len(self.scalers)
        combinacao_atual = 0

        for scaler_name, scaler in self.scalers:
            for algo_name in algos_para_otimizar:
                combinacao_atual += 1
                if algo_name not in self.algo_configs:
                    self._log(
                        f"⚠️ Algoritmo {algo_name} não encontrado nas configurações",
                        "WARN",
                    )
                    continue

                config = self.algo_configs[algo_name]
                print(
                    f"[{combinacao_atual}/{total_combinacoes}] {config.get('alias')} - {scaler_name}"
                )

                try:
                    start_time = time.time()

                    # Importar classe do modelo
                    import importlib

                    model_class = config["model_class"]
                    module_name = config["module"]
                    model_cls = getattr(
                        importlib.import_module(module_name), model_class
                    )
                    model = model_cls(**config["default_params"])

                    # Criar pipeline
                    pipe = Pipeline([("scaler", scaler), ("model", model)])

                    # Search space
                    search_space = {
                        f"model__{k}": _deserialize_space(v)
                        for k, v in config.get("search_space", {}).items()
                    }

                    # Otimização Bayesiana
                    opt = BayesSearchCV(
                        estimator=pipe,
                        search_spaces=search_space,
                        n_iter=self.config.n_iter_otimizacao,
                        cv=cv_splitter,
                        scoring=get_regression_metric(self.config.metrica_otimizar),
                        random_state=self.config.seed,
                        n_jobs=-1,
                        return_train_score=True,
                    )

                    opt.fit(
                        self.X_train,
                        self.y_train,
                        callback=[DeltaYStopper(delta=delta_metric, n_best=10)],
                    )

                    # Avaliação
                    best_model = opt.best_estimator_
                    y_pred_train = best_model.predict(self.X_train)
                    y_pred_test = best_model.predict(self.X_test)

                    # Armazenar resultados
                    resultados[algo_name, scaler_name] = {
                        "params": opt.best_params_,
                        "scores_dev": compute_metrics(self.y_train, y_pred_train),
                        "scores_test": compute_metrics(self.y_test, y_pred_test),
                        "cv_score": opt.best_score_,
                        "n_iterations": len(opt.optimizer_results_[0].x_iters),
                    }

                    opt_results[algo_name, scaler_name] = opt.optimizer_results_[0]

                    elapsed = time.time() - start_time
                    score = resultados[algo_name, scaler_name]["scores_test"][
                        self.config.metrica_otimizar
                    ]
                    print(f"  ✅ Concluído em {elapsed:.1f}s | Score: {score:.4f}")

                except Exception as e:
                    self._log(f"❌ Erro em {algo_name}-{scaler_name}: {e}", "ERROR")
                    continue

        self.resultados_otimizacao = {
            "resultados": resultados,
            "opt_results": opt_results,
        }

        # Salvar cache
        self._salvar_cache(cache_key, self.resultados_otimizacao)

        print("=" * 80)
        self._log(f"✅ Otimização concluída: {len(resultados)} modelos otimizados")

        return self.resultados_otimizacao

    def identificar_melhor_modelo(self) -> Dict[str, Any]:
        """
        Etapa 6: Identificação e análise do melhor modelo
        Usa a mesma lógica do notebook para encontrar o melhor modelo
        """
        self._log("🏆 Identificando melhor modelo")

        if not self.resultados_otimizacao:
            self._log(
                "❌ Otimização não executada. Execute executar_otimizacao_completa() primeiro",
                "ERROR",
            )
            return {}

        resultados = self.resultados_otimizacao["resultados"]

        # Encontrar melhor modelo (mesma lógica do notebook)
        metric = self.config.metrica_otimizar.lower()
        minimize_metrics = ["rmse", "mae", "mse"]

        best_key = None
        best_score = None

        for (algo, scaler), res in resultados.items():
            score = res["scores_test"][metric]
            if metric in minimize_metrics:
                if (best_score is None) or (score < best_score):
                    best_score = score
                    best_key = (algo, scaler)
            else:
                if (best_score is None) or (score > best_score):
                    best_score = score
                    best_key = (algo, scaler)

        if best_key is None:
            self._log("❌ Não foi possível identificar o melhor modelo", "ERROR")
            return {}

        best_algo, best_scaler = best_key
        best_result = resultados[best_key]

        self.melhor_modelo = {
            "algoritmo": best_algo,
            "scaler": best_scaler,
            "score": best_score,
            "params": best_result["params"],
            "metricas_teste": best_result["scores_test"],
            "metricas_treino": best_result["scores_dev"],
        }

        # Relatório do melhor modelo
        print("\n" + "=" * 80)
        print("🏆 MELHOR MODELO IDENTIFICADO")
        print("=" * 80)
        print(f"Algoritmo: {self.algo_configs[best_algo]['alias']} ({best_algo})")
        print(f"Scaler: {best_scaler}")
        print(f"Score ({self.config.metrica_otimizar}): {best_score:.4f}")

        print("\nMétricas de Teste:")
        for metric, value in best_result["scores_test"].items():
            print(f"  {metric.upper()}: {value:.4f}")

        print("\nMelhores Hiperparâmetros:")
        for param, value in best_result["params"].items():
            print(f"  {param}: {value}")
        print("=" * 80)

        self._log("✅ Melhor modelo identificado")
        return self.melhor_modelo

    def criar_pipeline_final(self) -> Pipeline:
        """
        Etapa 7: Criação do pipeline final
        Reconstrói o pipeline exatamente como no notebook
        """
        self._log("🔧 Criando pipeline final")

        if not self.melhor_modelo:
            self.identificar_melhor_modelo()

        if not self.melhor_modelo:
            self._log("❌ Melhor modelo não identificado", "ERROR")
            return None

        # Recuperar componentes (mesma lógica do notebook)
        best_algo = self.melhor_modelo["algoritmo"]
        best_scaler = self.melhor_modelo["scaler"]
        best_params = self.melhor_modelo["params"]

        # Encontrar scaler
        scaler_obj = None
        for name, scaler in self.scalers:
            if name == best_scaler:
                scaler_obj = scaler
                break

        # Encontrar modelo
        config = self.algo_configs[best_algo]
        import importlib

        model_cls = getattr(
            importlib.import_module(config["module"]), config["model_class"]
        )
        model_obj = model_cls(**config["default_params"])

        # Criar pipeline
        self.pipeline_final = Pipeline(
            [("scaler", scaler_obj), ("regressor", model_obj)]
        )

        # Configurar hiperparâmetros
        final_params = {}
        for key, value in best_params.items():
            if not key.startswith("regressor__"):
                final_params[f'regressor__{key.replace("model__", "")}'] = value
            else:
                final_params[key] = value

        self.pipeline_final.set_params(**final_params)

        self._log("✅ Pipeline final criado")
        return self.pipeline_final

    def treinar_modelo_final(self) -> Dict[str, Any]:
        """
        Etapa 8: Treinamento final no dataset completo
        Replica o processo final do notebook
        """
        self._log("🎓 Treinando modelo final no dataset completo")

        if self.pipeline_final is None:
            self.criar_pipeline_final()

        # Treinar no dataset completo
        X = self.df_processado.drop(self.target, axis=1)
        y = self.df_processado[self.target].copy()

        start_time = time.time()
        self.pipeline_final.fit(X, y)
        tempo_treino = time.time() - start_time

        # Avaliação no dataset completo
        y_pred_full = self.pipeline_final.predict(X)
        metricas_completo = compute_metrics(y, y_pred_full)

        # Salvar modelo
        joblib.dump(self.pipeline_final, self.config.arquivo_modelo_final)

        resultado_final = {
            "pipeline": self.pipeline_final,
            "metricas_dataset_completo": metricas_completo,
            "tempo_treinamento": tempo_treino,
            "y_real": y,
            "y_predito": y_pred_full,
            "arquivo_modelo": self.config.arquivo_modelo_final,
        }

        # Relatório final
        print("\n" + "=" * 80)
        print("🎓 MODELO FINAL TREINADO")
        print("=" * 80)
        print(f"Tempo de treinamento: {tempo_treino:.2f}s")

        headers = ["Métrica", "Valor no Dataset Completo"]
        metrics_data = [
            ["R² (R-squared)", f"{metricas_completo['r2']:.4f}"],
            ["MSE (Mean Squared Error)", f"{metricas_completo['mse']:.4f}"],
            ["RMSE (Root Mean Squared Error)", f"{metricas_completo['rmse']:.4f}"],
            ["MAE (Mean Absolute Error)", f"{metricas_completo['mae']:.4f}"],
        ]
        print(tabulate(metrics_data, headers=headers, tablefmt="grid"))
        print(f"\nModelo salvo em: {self.config.arquivo_modelo_final}")
        print("=" * 80)

        self._log("✅ Modelo final treinado e salvo")
        return resultado_final

    def gerar_visualizacoes_finais(self, resultado_final: Dict[str, Any]):
        """
        Etapa 9: Geração de visualizações finais
        Replica os gráficos do notebook
        """
        self._log("📊 Gerando visualizações finais")

        y_real = resultado_final["y_real"]
        y_predito = resultado_final["y_predito"]

        # Gráfico de análise (igual ao notebook)
        prediction_error = y_real - y_predito

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                "Tendência do Erro de Predição (Real - Previsto)",
                "Comparativo: Valores Reais vs. Valores Previstos",
            ),
        )

        # Gráfico do erro
        fig.add_trace(
            go.Scatter(
                x=y_real.index,
                y=prediction_error,
                mode="lines",
                name="Erro",
                line=dict(color="indianred"),
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

        # Gráfico real vs predito
        fig.add_trace(
            go.Scatter(
                x=y_real.index,
                y=y_real,
                mode="lines",
                name="Valor Real",
                line=dict(color="royalblue"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=y_real.index,
                y=y_predito,
                mode="lines",
                name="Valor Previsto",
                line=dict(color="darkorange", dash="dot"),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=700,
            title_text="🎯 Análise Final do Modelo - Agente ML Petrobras",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.update_yaxes(title_text="Erro (Real - Previsto)", row=1, col=1)
        fig.update_yaxes(title_text="Valor da Variável Target", row=2, col=1)
        fig.update_xaxes(title_text="Índice da Amostra", row=2, col=1)

        fig.show()

        self._log("✅ Visualizações geradas")

    def gerar_visualizacoes_otimizacao(self):
        """
        Visualizações do processo de otimização
        """
        if not self.resultados_otimizacao:
            self._log("❌ Resultados de otimização não disponíveis", "ERROR")
            return

        self._log("📈 Gerando visualizações de otimização")

        opt_results = self.resultados_otimizacao["opt_results"]

        # Plots de convergência (limitado para não sobrecarregar)
        max_plots = 6
        count = 0
        for (algo, scaler), res in opt_results.items():
            if count >= max_plots:
                break
            print(f"Convergência: {algo} - {scaler}")
            plot_convergence(res)
            plt.title(f"Convergência: {algo} ({scaler})")
            plt.show()
            count += 1

    def executar_missao_completa(
        self,
        algoritmos_selecionados: Optional[List[str]] = None,
        usar_cache: bool = True,
        eda_completa: bool = True,
    ) -> Dict[str, Any]:
        """
        🚀 MISSÃO PRINCIPAL DO AGENTE

        Executa todo o pipeline de ML de forma automatizada:
        1. Carregamento e processamento de dados
        2. Análise exploratória
        3. Preparação para ML
        4. Otimização de hiperparâmetros
        5. Identificação do melhor modelo
        6. Treinamento final
        7. Visualizações e relatórios

        Returns:
            Dict contendo todos os resultados da missão
        """
        self._log("🚀 INICIANDO MISSÃO COMPLETA DO AGENTE ML PETROBRAS")
        print("\n" + "=" * 80)
        print("🤖 AGENTE ML PETROBRAS - MISSÃO AUTOMATIZADA")
        print("MVP Machine Learning & Analytics - Pressão de Vapor da Nafta")
        print("=" * 80)

        missao_start = time.time()

        try:
            # Etapa 1: Dados
            self.carregar_dados()

            # Etapa 2: EDA
            self.executar_eda_automatizada(completa=eda_completa)

            # Etapa 3: Preparação ML
            self.preparar_dados_ml()

            # Etapa 4: Configurações
            self.carregar_configuracoes_algoritmos()

            # Etapa 5: Otimização
            self.executar_otimizacao_completa(
                usar_cache=usar_cache, algoritmos_selecionados=algoritmos_selecionados
            )

            # Etapa 6: Melhor modelo
            melhor_modelo = self.identificar_melhor_modelo()

            # Etapa 7: Pipeline final
            self.criar_pipeline_final()

            # Etapa 8: Treinamento final
            resultado_final = self.treinar_modelo_final()

            # Etapa 9: Visualizações
            self.gerar_visualizacoes_finais(resultado_final)

            # Etapa 10: Visualizações de otimização (opcional)
            if len(self.resultados_otimizacao.get("opt_results", {})) <= 6:
                self.gerar_visualizacoes_otimizacao()

            missao_tempo = time.time() - missao_start

            # Relatório final da missão
            print("\n" + "=" * 80)
            print("🎉 MISSÃO COMPLETA - SUCESSO!")
            print("=" * 80)
            print(f"⏱️ Tempo total da missão: {missao_tempo:.1f}s")
            print(f"🏆 Melhor algoritmo: {melhor_modelo['algoritmo']}")
            print(
                f"📊 Score final ({self.config.metrica_otimizar}): {melhor_modelo['score']:.4f}"
            )
            print(f"💾 Modelo salvo em: {self.config.arquivo_modelo_final}")
            print(f"📁 Cache disponível em: {self.cache_dir}")
            print("=" * 80)

            self._log("🎉 Missão completa executada com sucesso!")

            return {
                "melhor_modelo": melhor_modelo,
                "resultado_final": resultado_final,
                "resultados_otimizacao": self.resultados_otimizacao,
                "tempo_total": missao_tempo,
                "pipeline_final": self.pipeline_final,
                "dados_processados": self.df_processado,
            }

        except Exception as e:
            self._log(f"❌ ERRO NA MISSÃO: {e}", "ERROR")
            raise

    def carregar_modelo_salvo(self, arquivo: Optional[str] = None) -> Pipeline:
        """
        Carrega modelo salvo para uso
        """
        arquivo_modelo = arquivo or self.config.arquivo_modelo_final

        if not os.path.exists(arquivo_modelo):
            self._log(f"❌ Modelo não encontrado: {arquivo_modelo}", "ERROR")
            return None

        try:
            modelo = joblib.load(arquivo_modelo)
            self._log(f"✅ Modelo carregado: {arquivo_modelo}")

            # Demonstração de uso
            if self.df_processado is not None:
                sample = self.df_processado.drop(self.target, axis=1).head(1)
                predicao = modelo.predict(sample)
                print(f"Exemplo de predição: {predicao[0]:.4f}")

            return modelo
        except Exception as e:
            self._log(f"❌ Erro ao carregar modelo: {e}", "ERROR")
            return None

    def status_agente(self):
        """
        Mostra status atual do agente
        """
        print("\n" + "=" * 60)
        print("🤖 STATUS DO AGENTE ML PETROBRAS")
        print("=" * 60)
        print(
            f"📊 Dados carregados: {'✅' if self.df_processado is not None else '❌'}"
        )
        print(f"🔄 Dados preparados ML: {'✅' if self.X_train is not None else '❌'}")
        print(f"⚙️ Algoritmos configurados: {'✅' if self.algo_configs else '❌'}")
        print(
            f"🚀 Otimização executada: {'✅' if self.resultados_otimizacao else '❌'}"
        )
        print(f"🏆 Melhor modelo identificado: {'✅' if self.melhor_modelo else '❌'}")
        print(f"🔧 Pipeline final criado: {'✅' if self.pipeline_final else '❌'}")

        if self.df_processado is not None:
            print(f"📈 Shape dos dados: {self.df_processado.shape}")
            print(f"🎯 Target: {self.target}")

        if self.melhor_modelo:
            print(f"🏆 Melhor algoritmo: {self.melhor_modelo['algoritmo']}")
            print(f"📊 Score: {self.melhor_modelo['score']:.4f}")

        # Status do cache
        cache_files = list(self.cache_dir.glob("*.joblib"))
        print(f"💾 Arquivos em cache: {len(cache_files)}")

        print("=" * 60)


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA PARA USO RÁPIDO
# =============================================================================


def exemplo_uso_agente():
    """
    Exemplo completo de como usar o agente
    """
    print("🚀 EXEMPLO DE USO DO AGENTE ML PETROBRAS")

    # 1. Criar agente com configurações personalizadas
    config = ConfiguracaoProjeto(
        metrica_otimizar="rmse",
        n_iter_otimizacao=20,  # Reduzido para exemplo
        cv_folds=5,
    )
    agente = AgenteMLPetrobras(config)

    # 2. Executar missão completa automatizada
    resultados = agente.executar_missao_completa(
        algoritmos_selecionados=[
            "linear",
            "ridge",
            "random_forest",
        ],  # Exemplo com 3 algoritmos
        usar_cache=True,
        eda_completa=False,  # EDA simplificada para exemplo
    )

    # 3. Usar modelo final
    modelo_carregado = agente.carregar_modelo_salvo()

    # 4. Status final
    agente.status_agente()

    return agente, resultados


def executar_agente_producao():
    """
    Execução para produção com todos os algoritmos
    """
    print("🏭 EXECUÇÃO COMPLETA PARA PRODUÇÃO")

    # Configuração para produção
    config = ConfiguracaoProjeto(
        metrica_otimizar="rmse", n_iter_otimizacao=30, cv_folds=10
    )

    # Criar e executar agente
    agente = AgenteMLPetrobras(config)

    # Missão completa com todos os algoritmos
    resultados = agente.executar_missao_completa(usar_cache=True, eda_completa=True)

    return agente, resultados


if __name__ == "__main__":
    print("🤖 AGENTE ML PETROBRAS - PRONTO PARA EXECUÇÃO")
    print("\nPara usar o agente:")
    print("1. agente, resultados = exemplo_uso_agente()  # Exemplo rápido")
    print("2. agente, resultados = executar_agente_producao()  # Produção completa")
    print("3. agente.status_agente()  # Ver status")
    print("4. agente.carregar_modelo_salvo()  # Carregar modelo")
