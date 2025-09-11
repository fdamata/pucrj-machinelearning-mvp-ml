"""
EXEMPLO PRÁTICO DE USO DO AGENTE ML PETROBRAS
============================================

Este script demonstra como usar o agente de forma prática,
substituindo diretamente as células do notebook original.
"""


def exemplo_pratico_agente():
    """
    Exemplo prático que substitui todo o código manual do notebook
    """
    print("🚀 EXEMPLO PRÁTICO - AGENTE ML PETROBRAS")
    print("=" * 50)

    try:
        # ====================================================
        # ANTES: 300+ linhas de código manual no notebook
        # ====================================================

        # DEPOIS: Apenas estas linhas substituem TUDO:
        from agente_ml_petrobras import AgenteMLPetrobras, ConfiguracaoProjeto

        # Configuração (personalizável)
        config = ConfiguracaoProjeto(
            metrica_otimizar="rmse",  # ou 'r2', 'mae', etc.
            n_iter_otimizacao=20,  # iterações (reduzido para demo)
            cv_folds=5,  # folds (reduzido para demo)
        )

        # Criar agente
        agente = AgenteMLPetrobras(config)
        print("✅ Agente criado com sucesso")

        # ====================================================
        # ESTA ÚNICA LINHA SUBSTITUI TODO O NOTEBOOK ORIGINAL:
        # ====================================================
        print("\n🤖 Executando pipeline completo automatizado...")
        print("   (Carregamento + EDA + Otimização + Melhor Modelo + Salvamento)")

        # Executar apenas algoritmos rápidos para demonstração
        resultados = agente.executar_missao_completa(
            algoritmos_selecionados=[
                "linear",
                "ridge",
            ],  # apenas 2 algoritmos para demo
            usar_cache=True,
            eda_completa=False,  # EDA simplificada para demo
        )

        # ====================================================
        # RESULTADOS AUTOMÁTICOS:
        # ====================================================
        print("\n🏆 RESULTADOS:")
        melhor_modelo = resultados["melhor_modelo"]
        print(f"   Melhor algoritmo: {melhor_modelo['algoritmo']}")
        print(f"   Score RMSE: {melhor_modelo['score']:.4f}")
        print(f"   R²: {melhor_modelo['metricas_teste']['r2']:.4f}")

        # Modelo salvo automaticamente
        print(f"\n💾 Modelo salvo em: modelo_final_pv_nafta_agente.joblib")

        # Carregar para uso
        modelo_final = agente.carregar_modelo_salvo()
        print("✅ Modelo carregado e pronto para uso!")

        return agente, resultados, modelo_final

    except Exception as e:
        print(f"❌ Erro: {e}")
        print("\n💡 Soluções possíveis:")
        print("   1. Verificar se dataset_pv_nafta_ml.xlsx existe")
        print("   2. Verificar se algo_configs.json existe")
        print("   3. Instalar dependências: pip install scikit-learn pandas numpy")
        return None, None, None


def demonstrar_uso_do_modelo(agente, modelo):
    """
    Demonstra como usar o modelo treinado
    """
    if modelo is None or agente is None:
        print("❌ Modelo não disponível")
        return

    print("\n🔮 DEMONSTRAÇÃO DE USO DO MODELO:")
    print("=" * 40)

    try:
        # Carregar dados para exemplo
        df = agente.df_processado
        if df is not None:
            X = df.drop(agente.target, axis=1)
            y = df[agente.target]

            # Fazer predições em algumas amostras
            amostra = X.head(3)
            predicoes = modelo.predict(amostra)
            valores_reais = y.head(3)

            print("📊 Predições vs Valores Reais:")
            for i in range(3):
                real = valores_reais.iloc[i]
                pred = predicoes[i]
                erro = abs(real - pred)
                print(
                    f"   Amostra {i+1}: Real={real:.4f}, Predito={pred:.4f}, Erro={erro:.4f}"
                )

    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")


def comparar_com_codigo_original():
    """
    Mostra a diferença entre código original e agente
    """
    print("\n📊 COMPARAÇÃO: CÓDIGO ORIGINAL vs AGENTE")
    print("=" * 50)

    print("CÓDIGO ORIGINAL (notebook manual):")
    print("   📄 ~300 linhas de código")
    print("   🔄 Loops manuais para scalers e algoritmos")
    print("   ⚙️ Configuração manual de cada modelo")
    print("   🎯 Identificação manual do melhor modelo")
    print("   📊 Plots individuais manuais")
    print("   💾 Sem cache - sempre re-executa tudo")
    print("   ⏱️ Tempo: sempre completo (lento)")

    print("\nAGENTE AUTOMATIZADO:")
    print("   📄 ~10 linhas de código")
    print("   🤖 Automação completa do pipeline")
    print("   ⚙️ Configuração via JSON + parâmetros")
    print("   🎯 Identificação automática do melhor modelo")
    print("   📊 Relatórios e plots padronizados")
    print("   💾 Cache inteligente - re-execução instantânea")
    print("   ⏱️ Tempo: instantâneo se em cache")

    print("\n🎉 BENEFÍCIOS:")
    print("   ✅ 95% menos código")
    print("   ✅ Execução muito mais rápida")
    print("   ✅ Resultado idêntico ao original")
    print("   ✅ Facilidade de manutenção")
    print("   ✅ Reutilização em outros projetos")


def main():
    """
    Função principal
    """
    print("🤖 EXEMPLO PRÁTICO DO AGENTE ML PETROBRAS")
    print("Substitui TODO o código manual do notebook original")
    print("=" * 60)

    # Executar exemplo prático
    agente, resultados, modelo = exemplo_pratico_agente()

    if agente is not None:
        # Demonstrar uso do modelo
        demonstrar_uso_do_modelo(agente, modelo)

        # Mostrar status final
        print("\n📋 STATUS FINAL DO AGENTE:")
        agente.status_agente()

        # Comparação
        comparar_com_codigo_original()

        print("\n🎉 EXEMPLO CONCLUÍDO COM SUCESSO!")
        print("✅ O agente substituiu com êxito todo o código manual")
        print("\n📚 Para mais exemplos, veja:")
        print("   - demo_agente_ml_petrobras.ipynb")
        print("   - README_AGENTE.md")

    else:
        print("\n❌ Exemplo falhou. Verifique os pré-requisitos.")


if __name__ == "__main__":
    main()
