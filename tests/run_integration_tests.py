#!/usr/bin/env python3
"""
Integration test runner for the Stock Direction Predictor system.
Provides convenient execution of different test suites with proper configuration.
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any


def run_test_suite(test_file: str, markers: List[str] = None, verbose: bool = True, 
                  capture: str = "no") -> Dict[str, Any]:
    """
    Run a specific test suite and return results.
    
    Args:
        test_file: Path to the test file
        markers: List of pytest markers to filter tests
        verbose: Whether to run in verbose mode
        capture: Pytest capture mode ('no', 'sys', 'fd')
    
    Returns:
        Dictionary with test results
    """
    cmd = ["python", "-m", "pytest", test_file]
    
    if verbose:
        cmd.append("-v")
    
    if capture:
        cmd.extend(["-s", f"--capture={capture}"])
    
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # Add coverage if available
    try:
        import coverage
        cmd.extend(["--cov=stock_predictor", "--cov-report=term-missing"])
    except ImportError:
        pass
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        execution_time = time.time() - start_time
        
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': execution_time,
            'command': ' '.join(cmd)
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Test execution timed out after 30 minutes',
            'execution_time': time.time() - start_time,
            'command': ' '.join(cmd)
        }
    except Exception as e:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'execution_time': time.time() - start_time,
            'command': ' '.join(cmd)
        }


def print_test_results(suite_name: str, results: Dict[str, Any]) -> None:
    """Print formatted test results."""
    print(f"\n{'='*60}")
    print(f"TEST SUITE: {suite_name}")
    print(f"{'='*60}")
    print(f"Status: {'✓ PASSED' if results['success'] else '✗ FAILED'}")
    print(f"Return Code: {results['returncode']}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    print(f"Command: {results['command']}")
    
    if results['stdout']:
        print(f"\n--- STDOUT ---")
        print(results['stdout'])
    
    if results['stderr']:
        print(f"\n--- STDERR ---")
        print(results['stderr'])


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Integration test runner for Stock Direction Predictor"
    )
    
    parser.add_argument(
        '--suite', '-s',
        choices=['all', 'integration', 'e2e', 'benchmark', 'regression', 'quick'],
        default='all',
        help='Test suite to run (default: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Run tests in verbose mode'
    )
    
    parser.add_argument(
        '--capture',
        choices=['no', 'sys', 'fd'],
        default='no',
        help='Pytest capture mode (default: no)'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Run test suites in parallel (experimental)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Directory to save test results'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Timeout for each test suite in seconds (default: 1800)'
    )
    
    args = parser.parse_args()
    
    # Define test suites
    test_suites = {
        'integration': {
            'file': 'tests/test_integration_workflow.py',
            'markers': ['integration'],
            'description': 'Complete workflow integration tests'
        },
        'e2e': {
            'file': 'tests/test_end_to_end_scenarios.py',
            'markers': ['e2e'],
            'description': 'End-to-end tests with realistic market scenarios'
        },
        'benchmark': {
            'file': 'tests/test_performance_benchmarks.py',
            'markers': ['benchmark'],
            'description': 'Performance benchmarking and scalability tests'
        },
        'regression': {
            'file': 'tests/test_regression_consistency.py',
            'markers': ['regression'],
            'description': 'Regression tests for consistency validation'
        }
    }
    
    # Determine which suites to run
    if args.suite == 'all':
        suites_to_run = list(test_suites.keys())
    elif args.suite == 'quick':
        # Quick suite runs integration tests only
        suites_to_run = ['integration']
    else:
        suites_to_run = [args.suite]
    
    print(f"Stock Direction Predictor - Integration Test Runner")
    print(f"Running test suites: {', '.join(suites_to_run)}")
    print(f"Verbose mode: {args.verbose}")
    print(f"Capture mode: {args.capture}")
    
    # Check if test files exist
    missing_files = []
    for suite_name in suites_to_run:
        test_file = test_suites[suite_name]['file']
        if not Path(test_file).exists():
            missing_files.append(test_file)
    
    if missing_files:
        print(f"\nError: Missing test files:")
        for file in missing_files:
            print(f"  - {file}")
        sys.exit(1)
    
    # Run test suites
    all_results = {}
    overall_success = True
    total_time = 0
    
    for suite_name in suites_to_run:
        suite_config = test_suites[suite_name]
        print(f"\n{'='*60}")
        print(f"STARTING: {suite_name.upper()} - {suite_config['description']}")
        print(f"{'='*60}")
        
        results = run_test_suite(
            test_file=suite_config['file'],
            markers=suite_config['markers'],
            verbose=args.verbose,
            capture=args.capture
        )
        
        all_results[suite_name] = results
        total_time += results['execution_time']
        
        if not results['success']:
            overall_success = False
        
        print_test_results(suite_name.upper(), results)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Status: {'✓ ALL PASSED' if overall_success else '✗ SOME FAILED'}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Suites Run: {len(suites_to_run)}")
    
    print(f"\nDetailed Results:")
    for suite_name, results in all_results.items():
        status = "✓ PASSED" if results['success'] else "✗ FAILED"
        time_str = f"{results['execution_time']:.2f}s"
        print(f"  {suite_name:12} {status:10} {time_str:>8}")
    
    # Save results if output directory specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        from datetime import datetime
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_success': overall_success,
            'total_execution_time': total_time,
            'suites_run': suites_to_run,
            'results': all_results
        }
        
        output_file = output_dir / f"integration_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()