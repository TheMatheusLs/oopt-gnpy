#!/usr/bin/env python3
"""
Exemplo de uso do Numba para otimização de código CPU-bound no projeto oopt-gnpy.

Este script demonstra como usar o decorador @jit do Numba para acelerar
cálculos numéricos intensivos, especialmente útil para otimizações
relacionadas ao GGN NLI e outras operações matemáticas.
"""

import numpy as np
import time
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba não está instalado. Instale com: uv pip install -e '.[performance]'")
    print("    Executando apenas versões sem otimização...\n")


# Exemplo 1: Cálculo de potência (sem Numba)
def calculate_power_slow(signal_array, exponent=2):
    """Versão não otimizada - apenas para comparação."""
    result = np.zeros_like(signal_array)
    for i in range(len(signal_array)):
        result[i] = signal_array[i] ** exponent
    return result


# Exemplo 1: Cálculo de potência (com Numba)
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def calculate_power_fast(signal_array, exponent=2):
        """Versão otimizada com Numba JIT."""
        result = np.zeros_like(signal_array)
        for i in range(len(signal_array)):
            result[i] = signal_array[i] ** exponent
        return result


# Exemplo 2: Integração numérica (sem Numba)
def numerical_integration_slow(frequencies, spectrum):
    """
    Integração numérica simples - similar ao que pode ser usado
    em cálculos de NLI no oopt-gnpy.
    """
    total = 0.0
    n = len(frequencies)
    for i in range(n - 1):
        df = frequencies[i + 1] - frequencies[i]
        avg_value = (spectrum[i] + spectrum[i + 1]) / 2
        total += avg_value * df
    return total


# Exemplo 2: Integração numérica (com Numba)
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def numerical_integration_fast(frequencies, spectrum):
        """
        Versão otimizada com Numba - pode ser até 100x mais rápida
        em arrays grandes.
        """
        total = 0.0
        n = len(frequencies)
        for i in range(n - 1):
            df = frequencies[i + 1] - frequencies[i]
            avg_value = (spectrum[i] + spectrum[i + 1]) / 2
            total += avg_value * df
        return total


# Exemplo 3: Cálculo de NLI simplificado (sem Numba)
def calculate_nli_coefficient_slow(frequencies, power_spectrum):
    """
    Versão simplificada de cálculo de coeficiente NLI
    (similar ao usado em _ggn_approx).
    """
    n = len(frequencies)
    nli_array = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Cálculo simplificado para demonstração
                delta_f = abs(frequencies[i] - frequencies[j])
                nli_array[i] += power_spectrum[j] / (1 + delta_f**2)
    
    return nli_array


# Exemplo 3: Cálculo de NLI simplificado (com Numba)
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def calculate_nli_coefficient_fast(frequencies, power_spectrum):
        """
        Versão otimizada e paralelizada com Numba.
        O parâmetro 'parallel=True' permite paralelização automática.
        """
        n = len(frequencies)
        nli_array = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    delta_f = abs(frequencies[i] - frequencies[j])
                    nli_array[i] += power_spectrum[j] / (1 + delta_f**2)
        
        return nli_array


def benchmark_function(func, *args, warmup=True):
    """
    Executa benchmark de uma função.
    
    Args:
        func: Função a ser testada
        *args: Argumentos para a função
        warmup: Se True, executa uma vez antes para compilação JIT
    
    Returns:
        tuple: (resultado, tempo_execução)
    """
    if warmup:
        # Primeira execução para compilação JIT (se Numba estiver ativo)
        _ = func(*args)
    
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    
    execution_time = end - start
    return result, execution_time


def main():
    """Função principal para executar os benchmarks."""
    print("=" * 70)
    print("BENCHMARK: Comparação de Performance - Numba vs Código Python Puro")
    print("=" * 70)
    
    # Configuração dos dados de teste
    print("\n📊 Configurando dados de teste...")
    array_size = 100000
    signal = np.random.randn(array_size)
    frequencies = np.linspace(191e12, 196e12, 1000)  # 191-196 THz (banda C)
    spectrum = np.abs(np.random.randn(1000)) * 1e-3
    
    print(f"   • Array size: {array_size:,}")
    print(f"   • Frequency points: {len(frequencies):,}")
    
    # Teste 1: Cálculo de Potência
    print("\n" + "─" * 70)
    print("Teste 1: Cálculo de Potência (signal² para array grande)")
    print("─" * 70)
    
    result_slow, time_slow = benchmark_function(calculate_power_slow, signal, 2, warmup=False)
    print(f"   🐢 Versão Python puro: {time_slow*1000:.2f} ms")
    
    if NUMBA_AVAILABLE:
        result_fast, time_fast = benchmark_function(calculate_power_fast, signal, 2, warmup=True)
        print(f"   ⚡ Versão Numba:       {time_fast*1000:.2f} ms")
        speedup = time_slow / time_fast
        print(f"   🚀 Speedup: {speedup:.1f}x mais rápido!")
    
    # Teste 2: Integração Numérica
    print("\n" + "─" * 70)
    print("Teste 2: Integração Numérica (método do trapézio)")
    print("─" * 70)
    
    result_slow, time_slow = benchmark_function(numerical_integration_slow, frequencies, spectrum, warmup=False)
    print(f"   🐢 Versão Python puro: {time_slow*1000:.2f} ms")
    print(f"      Resultado: {result_slow:.6e}")
    
    if NUMBA_AVAILABLE:
        result_fast, time_fast = benchmark_function(numerical_integration_fast, frequencies, spectrum, warmup=True)
        print(f"   ⚡ Versão Numba:       {time_fast*1000:.2f} ms")
        print(f"      Resultado: {result_fast:.6e}")
        speedup = time_slow / time_fast
        print(f"   🚀 Speedup: {speedup:.1f}x mais rápido!")
    
    # Teste 3: Cálculo NLI Simplificado
    print("\n" + "─" * 70)
    print("Teste 3: Cálculo NLI Simplificado (loops duplos)")
    print("─" * 70)
    
    # Usar array menor para NLI (operação mais cara)
    freq_small = frequencies[::10]  # Reduz pontos para teste
    spec_small = spectrum[::10]
    
    result_slow, time_slow = benchmark_function(calculate_nli_coefficient_slow, freq_small, spec_small, warmup=False)
    print(f"   🐢 Versão Python puro: {time_slow*1000:.2f} ms")
    
    if NUMBA_AVAILABLE:
        result_fast, time_fast = benchmark_function(calculate_nli_coefficient_fast, freq_small, spec_small, warmup=True)
        print(f"   ⚡ Versão Numba:       {time_fast*1000:.2f} ms")
        speedup = time_slow / time_fast
        print(f"   🚀 Speedup: {speedup:.1f}x mais rápido!")
    
    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    
    if NUMBA_AVAILABLE:
        print("✅ Numba está instalado e funcionando corretamente!")
        print("\n💡 Dicas de uso:")
        print("   • Use @jit(nopython=True) para máxima performance")
        print("   • Use @jit(parallel=True) para paralelização automática")
        print("   • A primeira execução será lenta (compilação JIT)")
        print("   • Execuções subsequentes serão muito mais rápidas")
        print("\n📚 Para mais informações, consulte:")
        print("   https://numba.pydata.org/numba-doc/latest/user/5minguide.html")
    else:
        print("⚠️  Numba NÃO está instalado.")
        print("\n📦 Para instalar, execute:")
        print("   uv pip install -e '.[performance]'")
        print("\n   Ou diretamente:")
        print("   uv pip install numba")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
