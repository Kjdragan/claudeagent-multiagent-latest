#!/usr/bin/env python3
"""
Script to fix indentation issues in main_comprehensive_research.py
"""

import re

def fix_indentation():
    """Fix indentation issues in the main file"""

    with open('main_comprehensive_research.py', 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    in_class = False
    in_method = False
    method_indent = 0

    for i, line in enumerate(lines):
        line_num = i + 1

        # Skip comments and empty lines that don't need fixing
        if line.strip().startswith('#') or line.strip() == '':
            fixed_lines.append(line)
            continue

        # Check if we're in a method definition
        if re.match(r'async def |def ', line):
            # This is a method definition - should be indented with 4 spaces
            if not line.startswith('    async def ') and not line.startswith('    def '):
                line = '    ' + line.lstrip()
                print(f"Line {line_num}: Fixed method definition indentation")
            in_method = True
            method_indent = 8  # Method body should be 8 spaces

        elif in_method and line.strip():
            # This is inside a method body
            current_indent = len(line) - len(line.lstrip())
            expected_indent = method_indent

            # Handle docstrings specially
            if '"""' in line or "'''" in line:
                expected_indent = method_indent

            # Handle try/except blocks and other control structures
            if current_indent < expected_indent and (line.strip().startswith('try:') or
                                                   line.strip().startswith('except ') or
                                                   line.strip().startswith('finally:') or
                                                   line.strip().startswith('elif ') or
                                                   line.strip().startswith('else:')):
                expected_indent = method_indent

            if current_indent != expected_indent and current_indent < expected_indent:
                line = ' ' * expected_indent + line.lstrip()
                print(f"Line {line_num}: Fixed body indentation from {current_indent} to {expected_indent}")

        fixed_lines.append(line)

    # Write the fixed file
    with open('main_comprehensive_research.py', 'w') as f:
        f.writelines(fixed_lines)

    print("Indentation fix completed!")

if __name__ == "__main__":
    fix_indentation()