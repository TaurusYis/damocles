@echo off
echo DDR S-Parameter Analysis Tool
echo =============================

echo.
echo Step 1: Generating Touchstone files...
python generate_touchstone_files.py

echo.
echo Step 2: Running main analysis...
python main.py

echo.
echo Step 3: Running example usage...
python example_usage.py

echo.
echo Analysis complete! Check the generated files:
echo - *.png (comparison plots)
echo - *.xlsx (Excel reports)
echo - summary_statistics.txt
echo - touchstone_files_info.txt

pause 