#!/usr/bin/env python3
"""
Simple test script to verify the DDR S-parameter analysis tool
"""

import os
import sys
from ddr_s_parameter_analyzer import DDRSParameterAnalyzer

def test_basic_functionality():
    """
    Test basic functionality of the analyzer
    """
    print("Testing DDR S-Parameter Analyzer...")
    
    # Check if Touchstone files exist
    if not os.path.exists("test_model1.s24p"):
        print("Test files not found. Generating them...")
        try:
            from generate_touchstone_files import create_small_test_files
            create_small_test_files()
        except ImportError:
            print("Error: Could not import generate_touchstone_files.py")
            return False
    
    # Create analyzer
    analyzer = DDRSParameterAnalyzer()
    
    # Load test models
    print("Loading test models...")
    success1 = analyzer.read_touchstone_file("test_model1.s24p", "Test_Model_1")
    success2 = analyzer.read_touchstone_file("test_model2.s24p", "Test_Model_2")
    
    if not (success1 and success2):
        print("Error: Failed to load Touchstone files")
        return False
    
    # Define DDR ports
    dq_ports = [1, 2, 3, 4, 5, 6, 7, 8]
    dqs_ports = [9, 10, 11, 12]
    dqs_pairs = [(9, 10), (11, 12)]
    analyzer.define_ddr_ports(dq_ports, dqs_ports, dqs_pairs)
    
    # Test calculations
    print("Testing calculations...")
    
    try:
        # Test insertion loss calculation
        il_data = analyzer.calculate_insertion_loss("Test_Model_1")
        print(f"  ✓ Insertion loss calculated for {len(il_data)} signals")
        
        # Test return loss calculation
        rl_data = analyzer.calculate_return_loss("Test_Model_1")
        print(f"  ✓ Return loss calculated for {len(rl_data)} ports")
        
        # Test TDR calculation
        tdr_data = analyzer.calculate_tdr("Test_Model_1")
        print(f"  ✓ TDR calculated for {len(tdr_data)-1} ports")  # -1 for time_axis
        
        # Test crosstalk calculation
        xtalk_data = analyzer.calculate_crosstalk("Test_Model_1")
        print(f"  ✓ Crosstalk calculated ({len(xtalk_data)} combinations)")
        
        # Test Excel export
        print("Testing Excel export...")
        analyzer.export_to_excel("test_results.xlsx")
        print("  ✓ Excel export successful")
        
        # Test plotting (without showing)
        print("Testing plotting...")
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        analyzer.plot_metric_comparison('insertion_loss', 'test_insertion_loss.png')
        analyzer.plot_metric_comparison('return_loss', 'test_return_loss.png')
        print("  ✓ Plot generation successful")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during calculations: {str(e)}")
        return False

def test_file_generation():
    """
    Test Touchstone file generation
    """
    print("\nTesting Touchstone file generation...")
    
    try:
        from generate_touchstone_files import create_small_test_files
        create_small_test_files()
        
        # Check if files were created
        if os.path.exists("test_model1.s24p") and os.path.exists("test_model2.s24p"):
            print("  ✓ Touchstone files generated successfully")
            return True
        else:
            print("  ✗ Touchstone files not created")
            return False
            
    except Exception as e:
        print(f"  ✗ Error generating files: {str(e)}")
        return False

def main():
    """
    Run all tests
    """
    print("DDR S-Parameter Analysis Tool - Test Suite")
    print("=" * 50)
    
    # Test file generation
    file_test = test_file_generation()
    
    # Test basic functionality
    func_test = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  File Generation: {'✓ PASS' if file_test else '✗ FAIL'}")
    print(f"  Basic Functionality: {'✓ PASS' if func_test else '✗ FAIL'}")
    
    if file_test and func_test:
        print("\n🎉 All tests passed! The tool is working correctly.")
        print("\nYou can now run:")
        print("  python main.py")
        print("  python example_usage.py")
        print("  run_analysis.bat")
        return True
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 