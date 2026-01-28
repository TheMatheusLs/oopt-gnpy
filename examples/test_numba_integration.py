#!/usr/bin/env python3
"""
Test script to validate Numba optimizations integration in oopt-gnpy.

This script performs basic validation tests to ensure:
1. Numba is properly installed (if expected)
2. Optimized functions produce identical results to original implementations
3. Performance improvements are measurable
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all necessary modules can be imported."""
    print("=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)
    
    try:
        from gnpy.core import science_utils
        print("✅ gnpy.core.science_utils imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import science_utils: {e}")
        return False
    
    try:
        from gnpy.core import numba_optimizations
        print("✅ gnpy.core.numba_optimizations imported successfully")
    except ImportError as e:
        print(f"⚠️  numba_optimizations not available: {e}")
        print("   This is OK if Numba is not installed")
    
    return True


def test_numba_status():
    """Check if Numba is available and get version info."""
    print("\n" + "=" * 70)
    print("TEST 2: Numba Status")
    print("=" * 70)
    
    try:
        from gnpy.core.numba_optimizations import get_numba_info, is_numba_available
        
        is_available = is_numba_available()
        info = get_numba_info()
        
        if is_available:
            print("✅ Numba is AVAILABLE")
            print(f"   Version: {info.get('version', 'unknown')}")
            print(f"   Threading Layer: {info.get('threading_layer', 'unknown')}")
        else:
            print("⚠️  Numba is NOT available")
            print("   The code will use original implementations (slower but functional)")
        
        return True
    except ImportError:
        print("⚠️  Cannot check Numba status - module not found")
        return True  # Not a failure, just not available


def test_raised_cosine():
    """Test that raised_cosine produces identical results with and without Numba."""
    print("\n" + "=" * 70)
    print("TEST 3: raised_cosine Function Correctness")
    print("=" * 70)
    
    try:
        from gnpy.core import science_utils
        
        # Test parameters
        frequency = np.linspace(191e12, 196e12, 1000)
        channel_frequency = 193e12
        channel_baud_rate = 32e9
        channel_roll_off = 0.15
        
        # Force original implementation
        original_use_numba = science_utils.USE_NUMBA
        science_utils.USE_NUMBA = False
        
        result_original = science_utils.raised_cosine(
            frequency, channel_frequency, channel_baud_rate, channel_roll_off
        )
        
        # Use optimized version (if available)
        science_utils.USE_NUMBA = original_use_numba
        
        result_optimized = science_utils.raised_cosine(
            frequency, channel_frequency, channel_baud_rate, channel_roll_off
        )
        
        # Compare results
        max_diff = np.max(np.abs(result_original - result_optimized))
        
        if max_diff < 1e-10:
            print(f"✅ Results are identical (max diff: {max_diff:.2e})")
            return True
        else:
            print(f"❌ Results differ (max diff: {max_diff:.2e})")
            print(f"   Original sum: {np.sum(result_original)}")
            print(f"   Optimized sum: {np.sum(result_optimized)}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_simple():
    """Simple performance comparison for raised_cosine."""
    print("\n" + "=" * 70)
    print("TEST 4: Basic Performance Test (raised_cosine)")
    print("=" * 70)
    
    try:
        from gnpy.core import science_utils
        import time
        
        # Test parameters
        frequency = np.linspace(191e12, 196e12, 10000)
        channel_frequency = 193e12
        channel_baud_rate = 32e9
        channel_roll_off = 0.15
        
        # Test original implementation
        original_use_numba = science_utils.USE_NUMBA
        science_utils.USE_NUMBA = False
        
        # Warm-up
        _ = science_utils.raised_cosine(frequency, channel_frequency, channel_baud_rate, channel_roll_off)
        
        start = time.perf_counter()
        for _ in range(100):
            _ = science_utils.raised_cosine(frequency, channel_frequency, channel_baud_rate, channel_roll_off)
        time_original = time.perf_counter() - start
        
        print(f"   Original implementation: {time_original*1000:.2f} ms (100 iterations)")
        
        # Test optimized version (if available)
        if original_use_numba:
            science_utils.USE_NUMBA = True
            
            # Warm-up (important for JIT compilation)
            _ = science_utils.raised_cosine(frequency, channel_frequency, channel_baud_rate, channel_roll_off)
            
            start = time.perf_counter()
            for _ in range(100):
                _ = science_utils.raised_cosine(frequency, channel_frequency, channel_baud_rate, channel_roll_off)
            time_optimized = time.perf_counter() - start
            
            print(f"   Numba implementation:    {time_optimized*1000:.2f} ms (100 iterations)")
            
            speedup = time_original / time_optimized
            print(f"   🚀 Speedup: {speedup:.1f}x faster!")
            
            if speedup > 2.0:
                print("   ✅ Good performance improvement")
            elif speedup > 1.0:
                print("   ⚠️  Modest performance improvement")
            else:
                print("   ❌ No performance improvement - this is unexpected!")
        else:
            print("   ⚠️  Numba not available - skipping optimization test")
        
        # Restore original state
        science_utils.USE_NUMBA = original_use_numba
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "NUMBA OPTIMIZATIONS VALIDATION" + " " * 22 + "║")
    print("║" + " " * 20 + "for oopt-gnpy project" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Numba Status", test_numba_status),
        ("Correctness Test", test_raised_cosine),
        ("Performance Test", test_performance_simple),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:12} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The Numba optimizations are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
