"""
TESTE RÁPIDO DO AGENTE ML PETROBRAS
==================================

Script de teste para verificar se o agente está funcionando corretamente
antes de executar o notebook completo.
"""


def teste_agente_basico():
    """
    Teste básico para verificar se o agente foi implementado corretamente
    """
    print("🧪 INICIANDO TESTE BÁSICO DO AGENTE ML PETROBRAS")
    print("=" * 50)

    try:
        # Importar componentes principais
        from agente_ml_petrobras import (
            AgenteMLPetrobras,
            ConfiguracaoProjeto,
            exemplo_uso_agente,
            executar_agente_producao,
        )

        print("✅ Importação bem-sucedida")

        # Testar criação da configuração
        config = ConfiguracaoProjeto()
        print(f"✅ Configuração criada: {config.metrica_otimizar}")

        # Testar criação do agente
        agente = AgenteMLPetrobras(config)
        print("✅ Agente criado com sucesso")

        # Verificar diretório de cache
        print(f"✅ Diretório de cache: {agente.cache_dir}")

        # Testar status inicial
        print("\n📊 Status inicial do agente:")
        agente.status_agente()

        print("\n🎉 TESTE BÁSICO CONCLUÍDO COM SUCESSO!")
        print("✅ O agente está pronto para uso")

        return True

    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("💡 Verifique se todos os pacotes estão instalados")
        return False

    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False


def verificar_dependencias():
    """
    Verifica se todas as dependências necessárias estão disponíveis
    """
    print("\n🔍 VERIFICANDO DEPENDÊNCIAS")
    print("=" * 30)

    dependencias = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "sklearn",
        "joblib",
        "tabulate",
    ]

    dependencias_opcionais = ["xgboost", "lightgbm", "catboost", "statsmodels", "scipy"]

    print("📦 Dependências obrigatórias:")
    for dep in dependencias:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} - NECESSÁRIO INSTALAR!")

    print("\n📦 Dependências opcionais:")
    for dep in dependencias_opcionais:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ⚠️ {dep} - opcional, mas recomendado")


def verificar_arquivos():
    """
    Verifica se os arquivos necessários estão presentes
    """
    print("\n📁 VERIFICANDO ARQUIVOS NECESSÁRIOS")
    print("=" * 35)

    import os

    arquivos_necessarios = ["dataset_pv_nafta_ml.xlsx", "algo_configs.json"]

    for arquivo in arquivos_necessarios:
        if os.path.exists(arquivo):
            print(f"  ✅ {arquivo}")
        else:
            print(f"  ❌ {arquivo} - ARQUIVO NECESSÁRIO!")


def main():
    """
    Função principal do teste
    """
    print("🤖 TESTE DO AGENTE ML PETROBRAS")
    print("=" * 50)

    # Verificar dependências
    verificar_dependencias()

    # Verificar arquivos
    verificar_arquivos()

    # Teste básico do agente
    sucesso = teste_agente_basico()

    print("\n" + "=" * 50)
    if sucesso:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ O agente está pronto para uso")
        print("\n📋 Próximos passos:")
        print("  1. Abrir demo_agente_ml_petrobras.ipynb")
        print("  2. Executar células de demonstração")
        print("  3. Escolher execução rápida ou completa")
    else:
        print("❌ ALGUNS TESTES FALHARAM")
        print("💡 Verifique as mensagens de erro acima")
    print("=" * 50)


if __name__ == "__main__":
    main()
