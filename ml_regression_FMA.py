# Imports necessários
import os
import json
import joblib
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skopt.plots import plot_convergence, plot_objective
import importlib
import time

# # 1. Otimização da Configuração de Modelos


# Criação de uma classe centralizada para gerenciar configurações de algoritmos e scalers.
@dataclass
class ModelConfig:
    """Classe para armazenar configuração de um modelo"""

    alias: str
    model_class: str
    module: str
    default_params: Dict[str, Any]
    search_space: Dict[str, Any]
    default_metric: str


class OptimizedModelManager:
    """Gerenciador otimizado para configurações de modelos e scalers"""

    def __init__(self, config_file: str = "algo_configs.json"):
        self.config_file = config_file
        self._algo_configs = {}
        self._model_classes = {}  # Cache para classes de modelo
        self._scalers = []
        self._load_configurations()

    def _load_configurations(self):
        """Carrega configurações uma única vez"""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as f:
                configs = json.load(f)

            for algo_name, config in configs.items():
                self._algo_configs[algo_name] = ModelConfig(**config)

    def get_model_class(self, algo_name: str):
        """Retorna classe do modelo com cache para evitar importações repetidas"""
        if algo_name not in self._model_classes:
            config = self._algo_configs[algo_name]
            module = importlib.import_module(config.module)
            self._model_classes[algo_name] = getattr(module, config.model_class)

        return self._model_classes[algo_name]

    def create_model_instance(self, algo_name: str):
        """Cria instância do modelo com parâmetros padrão"""
        config = self._algo_configs[algo_name]
        model_class = self.get_model_class(algo_name)
        return model_class(**config.default_params)

    def get_search_space(
        self, algo_name: str, prefix: str = "model__"
    ) -> Dict[str, Any]:
        """Retorna search space com prefixo adequado para Pipeline"""
        config = self._algo_configs[algo_name]
        return {
            f"{prefix}{k}": self._deserialize_space(v)
            for k, v in config.search_space.items()
        }

    def _deserialize_space(self, obj):
        """Deserializa objetos do search space (implementação da função original)"""
        from skopt.space import Real, Integer, Categorical

        if isinstance(obj, (Real, Integer, Categorical)):
            return obj
        if not isinstance(obj, dict):
            return obj

        t = obj.get("_type")
        if t == "Real":
            return Real(obj["low"], obj["high"], prior=obj.get("prior"))
        elif t == "Integer":
            return Integer(obj["low"], obj["high"], prior=obj.get("prior"))
        elif t == "Categorical":
            return Categorical(obj.get("categories", []))
        return obj

    def set_scalers(self, scalers: List[Tuple[str, Any]]):
        """Define lista de scalers"""
        self._scalers = scalers

    def get_scalers(self) -> List[Tuple[str, Any]]:
        """Retorna lista de scalers"""
        return self._scalers

    def get_all_algorithms(self) -> List[str]:
        """Retorna lista de todos os algoritmos disponíveis"""
        return list(self._algo_configs.keys())

    def get_algorithm_alias(self, algo_name: str) -> str:
        """Retorna alias do algoritmo"""
        return self._algo_configs[algo_name].alias


# Instância global do gerenciador
model_manager = OptimizedModelManager()


# # 2. Consolidação das Funções de Avaliação


# Desenvolvimento de uma função centralizada para calcular métricas, identificar o melhor modelo e retornar informações completas do modelo otimizado.
class MetricsCalculator:
    """Calculadora otimizada de métricas com cache e funcionalidades consolidadas"""

    # Definição de métricas uma única vez
    MINIMIZE_METRICS = {
        "rmse",
        "mae",
        "mse",
        "msle",
        "rmsle",
        "mape",
        "poisson",
        "gamma",
        "max_error",
        "medae",
    }
    MAXIMIZE_METRICS = {"r2", "var", "d2", "explained_variance"}

    @staticmethod
    def compute_all_metrics(y_true, y_pred) -> Dict[str, float]:
        """Calcula todas as métricas de uma vez"""
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
        }

    @classmethod
    def is_metric_to_minimize(cls, metric: str) -> bool:
        """Determina se métrica deve ser minimizada"""
        return metric.lower() in cls.MINIMIZE_METRICS

    @classmethod
    def is_metric_to_maximize(cls, metric: str) -> bool:
        """Determina se métrica deve ser maximizada"""
        return metric.lower() in cls.MAXIMIZE_METRICS


class OptimizedResultsAnalyzer:
    """Analisador otimizado de resultados com funcionalidades consolidadas"""

    def __init__(self, results: Dict, metric_to_optimize: str):
        self.results = results
        self.metric_to_optimize = metric_to_optimize.lower()
        self.calculator = MetricsCalculator()
        self._best_model_cache = None

    def find_best_model(self) -> Tuple[Tuple[str, str], Dict[str, Any]]:
        """
        Encontra o melhor modelo uma única vez e faz cache do resultado

        Returns:
            Tuple[(algo_name, scaler_name), model_info]
        """
        if self._best_model_cache is not None:
            return self._best_model_cache

        best_key = None
        best_score = None
        is_minimize = self.calculator.is_metric_to_minimize(self.metric_to_optimize)

        for (algo, scaler), result in self.results.items():
            score = result["scores_test"][self.metric_to_optimize]

            if best_score is None:
                best_score = score
                best_key = (algo, scaler)
                continue

            if is_minimize:
                if score < best_score:
                    best_score = score
                    best_key = (algo, scaler)
            else:
                if score > best_score:
                    best_score = score
                    best_key = (algo, scaler)

        # Cache do resultado
        self._best_model_cache = (best_key, self.results[best_key])
        return self._best_model_cache

    def get_best_model_summary(self) -> Dict[str, Any]:
        """Retorna resumo completo do melhor modelo"""
        best_key, best_result = self.find_best_model()
        algo_name, scaler_name = best_key

        return {
            "algorithm": algo_name,
            "scaler": scaler_name,
            "algorithm_alias": model_manager.get_algorithm_alias(algo_name),
            "best_params": best_result["params"],
            "test_score": best_result["scores_test"][self.metric_to_optimize],
            "all_test_metrics": best_result["scores_test"],
            "train_metrics": best_result["scores_dev"],
            "cv_scores": best_result.get("cv_scores", []),
        }

    def prepare_boxplot_data(self) -> pd.DataFrame:
        """Prepara dados para boxplot uma única vez"""
        data = []

        for (algo, scaler), result in self.results.items():
            # Scores de validação dos folds
            cv_scores = result.get("cv_scores", [{}])
            if cv_scores:
                for fold_score_key, fold_score_value in cv_scores[0].items():
                    if fold_score_key.startswith("fold_") and fold_score_key.endswith(
                        "_val"
                    ):
                        data.append(
                            {
                                "Algoritmo": algo,
                                "Scaler": scaler,
                                "Score": fold_score_value,
                                "Type": "CV",
                            }
                        )

            # Score de teste
            test_score = result["scores_test"][self.metric_to_optimize]
            data.append(
                {
                    "Algoritmo": algo,
                    "Scaler": scaler,
                    "Score": test_score,
                    "Type": "Test",
                }
            )

        return pd.DataFrame(data)

    def get_ordered_algorithms(
        self, df_box: pd.DataFrame, by_test: bool = True
    ) -> List[str]:
        """Retorna algoritmos ordenados por performance"""
        if by_test:
            test_df = df_box[df_box["Type"] == "Test"]
            test_df_merged = test_df.copy()
            test_df_merged["Algoritmo"] = (
                test_df_merged["Algoritmo"] + " (" + test_df_merged["Scaler"] + ")"
            )

            ascending = self.calculator.is_metric_to_minimize(self.metric_to_optimize)
            return (
                test_df_merged.groupby("Algoritmo")["Score"]
                .mean()
                .sort_values(ascending=ascending)
                .index.tolist()
            )
        else:
            cv_df = df_box[df_box["Type"] == "CV"]
            cv_df_merged = cv_df.copy()
            cv_df_merged["Algoritmo"] = (
                cv_df_merged["Algoritmo"] + " (" + cv_df_merged["Scaler"] + ")"
            )

            # Ajustar sinais para métricas de minimização
            if self.calculator.is_metric_to_minimize(self.metric_to_optimize):
                cv_df_merged["Score"] = -cv_df_merged["Score"]

            ascending = self.calculator.is_metric_to_minimize(self.metric_to_optimize)
            return (
                cv_df_merged.groupby("Algoritmo")["Score"]
                .median()
                .sort_values(ascending=ascending)
                .index.tolist()
            )


# # 3. Identificação do Melhor Modelo


# Implementação de função para encontrar o melhor modelo, scaler e hiperparâmetros.
class BestModelIdentifier:
    """Classe otimizada para identificação e manipulação do melhor modelo"""

    def __init__(self, results_analyzer: OptimizedResultsAnalyzer):
        self.analyzer = results_analyzer
        self._best_pipeline_cache = None

    def create_best_pipeline(self) -> Pipeline:
        """
        Cria pipeline do melhor modelo uma única vez com cache

        Returns:
            Pipeline configurado com melhor modelo e scaler
        """
        if self._best_pipeline_cache is not None:
            return self._best_pipeline_cache

        summary = self.analyzer.get_best_model_summary()

        # Encontrar scaler
        scaler_obj = None
        for name, scaler in model_manager.get_scalers():
            if name == summary["scaler"]:
                scaler_obj = scaler
                break

        if scaler_obj is None:
            raise ValueError(f"Scaler '{summary['scaler']}' não encontrado")

        # Criar instância do modelo
        model_obj = model_manager.create_model_instance(summary["algorithm"])

        # Criar pipeline
        pipeline = Pipeline([("scaler", scaler_obj), ("regressor", model_obj)])

        # Configurar hiperparâmetros
        final_params = {}
        for key, value in summary["best_params"].items():
            if not key.startswith("regressor__"):
                final_params[f'regressor__{key.replace("model__", "")}'] = value
            else:
                final_params[key] = value

        pipeline.set_params(**final_params)

        # Cache do pipeline
        self._best_pipeline_cache = pipeline
        return pipeline

    def train_and_evaluate_best_model(
        self, X_train, y_train, X_test, y_test
    ) -> Dict[str, Any]:
        """
        Treina e avalia o melhor modelo em uma única função

        Returns:
            Dicionário com métricas de treino e teste
        """
        pipeline = self.create_best_pipeline()
        summary = self.analyzer.get_best_model_summary()

        # Treinar
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Previsões
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Métricas
        train_metrics = MetricsCalculator.compute_all_metrics(y_train, y_pred_train)
        test_metrics = MetricsCalculator.compute_all_metrics(y_test, y_pred_test)

        return {
            "pipeline": pipeline,
            "summary": summary,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "training_time": training_time,
            "predictions": {"y_pred_train": y_pred_train, "y_pred_test": y_pred_test},
        }

    def retrain_on_full_dataset(self, X, y) -> Dict[str, Any]:
        """
        Retreina o melhor modelo no dataset completo

        Returns:
            Dicionário com pipeline treinado e métricas
        """
        pipeline = self.create_best_pipeline()

        # Retreinar
        start_time = time.time()
        pipeline.fit(X, y)
        training_time = time.time() - start_time

        # Previsões e métricas
        y_pred_full = pipeline.predict(X)
        full_metrics = MetricsCalculator.compute_all_metrics(y, y_pred_full)

        return {
            "pipeline": pipeline,
            "full_metrics": full_metrics,
            "training_time": training_time,
            "y_pred_full": y_pred_full,
        }


# # 4. Criação e Treinamento de Pipelines


# Criação de função factory para pipelines que elimina duplicação na criação de pipelines e padroniza o processo de treinamento.
class PipelineFactory:
    """Factory otimizada para criação e gerenciamento de pipelines"""

    @staticmethod
    def create_pipeline(algo_name: str, scaler_name: str) -> Pipeline:
        """
        Cria pipeline baseado em nomes de algoritmo e scaler

        Args:
            algo_name: Nome do algoritmo
            scaler_name: Nome do scaler

        Returns:
            Pipeline configurado
        """
        # Encontrar scaler
        scaler_obj = None
        for name, scaler in model_manager.get_scalers():
            if name == scaler_name:
                scaler_obj = scaler
                break

        if scaler_obj is None:
            raise ValueError(f"Scaler '{scaler_name}' não encontrado")

        # Criar modelo
        model_obj = model_manager.create_model_instance(algo_name)

        return Pipeline([("scaler", scaler_obj), ("regressor", model_obj)])

    @staticmethod
    def configure_pipeline_params(
        pipeline: Pipeline, params: Dict[str, Any]
    ) -> Pipeline:
        """
        Configura parâmetros de um pipeline

        Args:
            pipeline: Pipeline a ser configurado
            params: Dicionário de parâmetros

        Returns:
            Pipeline configurado
        """
        final_params = {}
        for key, value in params.items():
            if not key.startswith("regressor__"):
                final_params[f'regressor__{key.replace("model__", "")}'] = value
            else:
                final_params[key] = value

        pipeline.set_params(**final_params)
        return pipeline

    @staticmethod
    def get_search_space_for_pipeline(algo_name: str) -> Dict[str, Any]:
        """
        Retorna search space configurado para uso com Pipeline

        Args:
            algo_name: Nome do algoritmo

        Returns:
            Dicionário de search space
        """
        return model_manager.get_search_space(algo_name, prefix="model__")


class ModelTrainingOrchestrator:
    """Orquestrador otimizado para treinamento de múltiplos modelos"""

    def __init__(
        self, scalers: List[Tuple[str, Any]], cv_folds: int = 10, random_state: int = 42
    ):
        self.scalers = scalers
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv_splitter = KFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        model_manager.set_scalers(scalers)

    def train_single_model(
        self, algo_name: str, scaler_name: str, X_train, y_train, n_iter: int = 30
    ) -> Dict[str, Any]:
        """
        Treina um único modelo com otimização de hiperparâmetros

        Args:
            algo_name: Nome do algoritmo
            scaler_name: Nome do scaler
            X_train, y_train: Dados de treinamento
            n_iter: Número de iterações para otimização

        Returns:
            Dicionário com resultados do treinamento
        """
        from skopt import BayesSearchCV
        from skopt.callbacks import DeltaYStopper

        # Criar pipeline
        pipeline = PipelineFactory.create_pipeline(algo_name, scaler_name)

        # Obter search space
        search_space = PipelineFactory.get_search_space_for_pipeline(algo_name)

        # Configurar otimização Bayesiana
        opt = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=self.cv_splitter,
            scoring="neg_root_mean_squared_error",  # Pode ser parametrizado
            random_state=self.random_state,
            n_jobs=-1,
            return_train_score=True,
        )

        # Treinar
        start_time = time.time()
        opt.fit(X_train, y_train, callback=[DeltaYStopper(delta=0.05, n_best=10)])
        training_time = time.time() - start_time

        return {
            "optimizer": opt,
            "best_pipeline": opt.best_estimator_,
            "best_params": opt.best_params_,
            "best_score": opt.best_score_,
            "training_time": training_time,
            "n_iterations": len(opt.optimizer_results_[0].x_iters),
        }


# # 5. Otimização dos Gráficos e Visualizações


# Desenvolvimento de funções reutilizáveis para plots de convergência, análise de hiperparâmetros e comparação de modelos.
class OptimizedVisualization:
    """Classe otimizada para visualizações com reutilização de código"""

    def __init__(self, results_analyzer: OptimizedResultsAnalyzer):
        self.analyzer = results_analyzer

    def plot_convergence_multiple(
        self, opt_results: Dict[Tuple[str, str], Any], max_plots: int = 6
    ) -> None:
        """
        Plota convergência para múltiplos modelos de forma otimizada

        Args:
            opt_results: Dicionário com resultados de otimização
            max_plots: Número máximo de plots a exibir
        """
        n_plots = min(len(opt_results), max_plots)
        if n_plots == 0:
            return

        # Calcular layout
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for idx, ((algo, scaler), res) in enumerate(opt_results.items()):
            if idx >= max_plots:
                break

            ax = axes[idx] if n_plots > 1 else axes[0]

            # Plot de convergência usando skopt
            try:
                plot_convergence(res, ax=ax)
                ax.set_title(f"{algo} - {scaler}")
            except Exception as e:
                ax.text(0.5, 0.5, f"Erro: {str(e)}", ha="center", va="center")
                ax.set_title(f"{algo} - {scaler} (Erro)")

        # Remover eixos vazios
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    def plot_objective_multiple(
        self, opt_results: Dict[Tuple[str, str], Any], max_plots: int = 4
    ) -> None:
        """
        Plota objetivos para múltiplos modelos de forma otimizada

        Args:
            opt_results: Dicionário com resultados de otimização
            max_plots: Número máximo de plots a exibir
        """
        n_plots = min(len(opt_results), max_plots)
        if n_plots == 0:
            return

        for idx, ((algo, scaler), res) in enumerate(opt_results.items()):
            if idx >= max_plots:
                break

            print(f"Análise de Hiperparâmetros: {algo} - {scaler}")
            try:
                plot_objective(res, size=3)
                plt.suptitle(f"Impacto dos Hiperparâmetros - {algo} ({scaler})")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Erro ao plotar {algo} - {scaler}: {str(e)}")

    def create_performance_comparison_plot(
        self, y_true, y_pred_full, model_name: str = "Melhor Modelo"
    ) -> go.Figure:
        """
        Cria gráfico de comparação de performance otimizado

        Args:
            y_true: Valores reais
            y_pred_full: Valores preditos
            model_name: Nome do modelo para o título

        Returns:
            Figura do Plotly
        """
        prediction_error = y_true - y_pred_full

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

        # Gráfico de erro
        fig.add_trace(
            go.Scatter(
                x=(
                    y_true.index
                    if hasattr(y_true, "index")
                    else list(range(len(y_true)))
                ),
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
        x_axis = y_true.index if hasattr(y_true, "index") else list(range(len(y_true)))

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_true,
                mode="lines",
                name="Valor Real",
                line=dict(color="royalblue"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=y_pred_full,
                mode="lines",
                name="Valor Previsto",
                line=dict(color="darkorange", dash="dot"),
            ),
            row=2,
            col=1,
        )

        # Layout
        fig.update_layout(
            height=700,
            title_text=f"Análise de Performance - {model_name}",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.update_yaxes(title_text="Erro (Real - Previsto)", row=1, col=1)
        fig.update_yaxes(title_text="Valor da Variável Target", row=2, col=1)
        fig.update_xaxes(title_text="Índice da Amostra", row=2, col=1)

        return fig

    def create_summary_report(self, best_model_results: Dict[str, Any]) -> None:
        """
        Cria relatório resumido de forma otimizada

        Args:
            best_model_results: Resultados do melhor modelo
        """
        summary = best_model_results["summary"]
        test_metrics = best_model_results["test_metrics"]
        train_metrics = best_model_results["train_metrics"]

        print("=" * 80)
        print(f"RELATÓRIO DO MELHOR MODELO")
        print("=" * 80)
        print(f"Algoritmo: {summary['algorithm_alias']} ({summary['algorithm']})")
        print(f"Scaler: {summary['scaler']}")
        print(
            f"Tempo de treinamento: {best_model_results.get('training_time', 'N/A'):.2f}s"
        )
        print("-" * 80)

        print("MÉTRICAS DE TESTE:")
        for metric, value in test_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")

        print("\nMÉTRICAS DE TREINO:")
        for metric, value in train_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")

        print("\nMELHORES HIPERPARÂMETROS:")
        for param, value in summary["best_params"].items():
            print(f"  {param}: {value}")

        print("=" * 80)


# # 6. Implementação de Cache para Resultados


# Sistema de cache para armazenar resultados de otimização e evitar re-execução desnecessária de experimentos.
class ExperimentCache:
    """Sistema de cache para experimentos de machine learning"""

    def __init__(self, cache_dir: str = "ml_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Arquivos de cache
        self.results_cache_file = self.cache_dir / "experiment_results.pkl"
        self.models_cache_dir = self.cache_dir / "models"
        self.models_cache_dir.mkdir(exist_ok=True)

        # Cache em memória
        self._memory_cache = {}

    def _generate_experiment_key(
        self,
        X_train,
        y_train,
        algorithms: List[str],
        scalers: List[str],
        cv_folds: int,
        metric: str,
        n_iter: int,
    ) -> str:
        """Gera chave única para o experimento"""
        # Criar hash baseado nos dados e parâmetros
        data_hash = hashlib.md5(
            str(X_train.shape).encode()
            + str(y_train.shape).encode()
            + str(sorted(algorithms)).encode()
            + str(sorted(scalers)).encode()
            + str(cv_folds).encode()
            + str(metric).encode()
            + str(n_iter).encode()
        ).hexdigest()[:16]

        return f"exp_{data_hash}"

    def save_experiment_results(
        self, experiment_key: str, results: Dict[str, Any]
    ) -> None:
        """Salva resultados do experimento"""
        cache_data = {
            "experiment_key": experiment_key,
            "results": results,
            "timestamp": time.time(),
        }

        with open(self.results_cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        # Salvar também em memória
        self._memory_cache[experiment_key] = cache_data

        print(f"Resultados salvos no cache: {experiment_key}")

    def load_experiment_results(self, experiment_key: str) -> Optional[Dict[str, Any]]:
        """Carrega resultados do experimento"""
        # Verificar cache em memória primeiro
        if experiment_key in self._memory_cache:
            return self._memory_cache[experiment_key]["results"]

        # Verificar cache em disco
        if self.results_cache_file.exists():
            try:
                with open(self.results_cache_file, "rb") as f:
                    cache_data = pickle.load(f)

                if cache_data["experiment_key"] == experiment_key:
                    # Carregar para memória também
                    self._memory_cache[experiment_key] = cache_data
                    return cache_data["results"]
            except Exception as e:
                print(f"Erro ao carregar cache: {e}")

        return None

    def cache_exists(self, experiment_key: str) -> bool:
        """Verifica se cache existe para o experimento"""
        return experiment_key in self._memory_cache or (
            self.results_cache_file.exists() and self._check_disk_cache(experiment_key)
        )

    def _check_disk_cache(self, experiment_key: str) -> bool:
        """Verifica cache em disco"""
        try:
            with open(self.results_cache_file, "rb") as f:
                cache_data = pickle.load(f)
            return cache_data["experiment_key"] == experiment_key
        except:
            return False

    def save_model(self, model_key: str, pipeline: Pipeline) -> str:
        """
        Salva modelo treinado

        Args:
            model_key: Chave identificadora do modelo
            pipeline: Pipeline treinado

        Returns:
            Caminho do arquivo salvo
        """
        model_file = self.models_cache_dir / f"{model_key}.joblib"
        joblib.dump(pipeline, model_file)
        print(f"Modelo salvo: {model_file}")
        return str(model_file)

    def load_model(self, model_key: str) -> Optional[Pipeline]:
        """Carrega modelo salvo"""
        model_file = self.models_cache_dir / f"{model_key}.joblib"
        if model_file.exists():
            return joblib.load(model_file)
        return None

    def clear_cache(self) -> None:
        """Limpa todos os caches"""
        if self.results_cache_file.exists():
            self.results_cache_file.unlink()

        for model_file in self.models_cache_dir.glob("*.joblib"):
            model_file.unlink()

        self._memory_cache.clear()
        print("Cache limpo com sucesso")

    def get_cache_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o cache"""
        info = {
            "memory_cache_size": len(self._memory_cache),
            "cache_dir": str(self.cache_dir),
            "results_cache_exists": self.results_cache_file.exists(),
            "cached_models": len(list(self.models_cache_dir.glob("*.joblib"))),
        }

        if self.results_cache_file.exists():
            info["results_cache_size"] = self.results_cache_file.stat().st_size

        return info


class OptimizedMLWorkflow:
    """Workflow completo otimizado com cache"""

    def __init__(self, cache_dir: str = "ml_cache"):
        self.cache = ExperimentCache(cache_dir)
        self.results_analyzer = None
        self.best_model_identifier = None

    def run_experiment(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        algorithms: List[str],
        scalers: List[Tuple[str, Any]],
        metric_to_optimize: str = "rmse",
        cv_folds: int = 10,
        n_iter: int = 30,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Executa experimento completo com cache

        Args:
            X_train, y_train, X_test, y_test: Dados de treino e teste
            algorithms: Lista de algoritmos para testar
            scalers: Lista de scalers para testar
            metric_to_optimize: Métrica a otimizar
            cv_folds: Número de folds para CV
            n_iter: Iterações para otimização bayesiana
            use_cache: Se deve usar cache

        Returns:
            Dicionário com todos os resultados
        """
        # Gerar chave do experimento
        scaler_names = [name for name, _ in scalers]
        experiment_key = self.cache._generate_experiment_key(
            X_train,
            y_train,
            algorithms,
            scaler_names,
            cv_folds,
            metric_to_optimize,
            n_iter,
        )

        # Verificar cache
        if use_cache and self.cache.cache_exists(experiment_key):
            print(f"Carregando resultados do cache: {experiment_key}")
            cached_results = self.cache.load_experiment_results(experiment_key)
            if cached_results:
                self.results_analyzer = OptimizedResultsAnalyzer(
                    cached_results, metric_to_optimize
                )
                self.best_model_identifier = BestModelIdentifier(self.results_analyzer)
                return cached_results

        # Executar experimento
        print(f"Executando novo experimento: {experiment_key}")

        # Configurar model manager
        model_manager.set_scalers(scalers)

        # Orquestrador de treinamento
        orchestrator = ModelTrainingOrchestrator(scalers, cv_folds)

        # Treinar todos os modelos
        results = {}
        for algo_name in algorithms:
            for scaler_name, _ in scalers:
                print(f"Treinando: {algo_name} - {scaler_name}")

                try:
                    training_result = orchestrator.train_single_model(
                        algo_name, scaler_name, X_train, y_train, n_iter
                    )

                    # Avaliar no conjunto de teste
                    best_pipeline = training_result["best_pipeline"]
                    y_pred_train = best_pipeline.predict(X_train)
                    y_pred_test = best_pipeline.predict(X_test)

                    results[(algo_name, scaler_name)] = {
                        "params": training_result["best_params"],
                        "scores_dev": MetricsCalculator.compute_all_metrics(
                            y_train, y_pred_train
                        ),
                        "scores_test": MetricsCalculator.compute_all_metrics(
                            y_test, y_pred_test
                        ),
                        "training_time": training_result["training_time"],
                        "n_iterations": training_result["n_iterations"],
                    }

                except Exception as e:
                    print(f"Erro ao treinar {algo_name} - {scaler_name}: {e}")
                    continue

        # Salvar no cache
        if use_cache:
            self.cache.save_experiment_results(experiment_key, results)

        # Configurar analisadores
        self.results_analyzer = OptimizedResultsAnalyzer(results, metric_to_optimize)
        self.best_model_identifier = BestModelIdentifier(self.results_analyzer)

        return results

    def get_best_model_complete_analysis(
        self, X, y, X_train, y_train, X_test, y_test
    ) -> Dict[str, Any]:
        """Análise completa do melhor modelo"""
        if self.best_model_identifier is None:
            raise ValueError("Execute run_experiment primeiro")

        # Treinar e avaliar melhor modelo
        best_results = self.best_model_identifier.train_and_evaluate_best_model(
            X_train, y_train, X_test, y_test
        )

        # Retreinar no dataset completo
        full_results = self.best_model_identifier.retrain_on_full_dataset(X, y)

        # Salvar modelo final
        summary = best_results["summary"]
        model_key = f"best_model_{summary['algorithm']}_{summary['scaler']}"
        model_path = self.cache.save_model(model_key, full_results["pipeline"])

        return {
            "best_results": best_results,
            "full_results": full_results,
            "model_path": model_path,
            "model_key": model_key,
        }


# Instância global do workflow
workflow = OptimizedMLWorkflow()
