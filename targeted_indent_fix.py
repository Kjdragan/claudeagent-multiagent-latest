#!/usr/bin/env python3
"""
Targeted fix for indentation issues in helper methods section
"""

import re

def fix_helper_methods_indentation():
    """Fix indentation specifically for helper methods section"""

    with open('main_comprehensive_research.py', 'r') as f:
        lines = f.readlines()

    # Find helper methods section
    helper_start = None
    for i, line in enumerate(lines):
        if '# Helper methods for multi-agent workflow' in line:
            helper_start = i
            break

    if helper_start is None:
        print("Could not find helper methods section")
        return

    fixed_lines = lines[:helper_start + 1]  # Keep up to and including helper methods comment

    # Process each line in helper methods
    current_indent = 0
    in_method = False

    for i in range(helper_start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            fixed_lines.append(line)
            continue

        # Check for method definition
        if re.match(r'async def |def ', stripped):
            # Class method should have 4 spaces
            if not line.startswith('    async def ') and not line.startswith('    def '):
                line = '    ' + line.lstrip()
                print(f"Line {i+1}: Fixed method definition")
            in_method = True
            current_indent = 8  # Method body indentation

        # Check for docstring
        elif in_method and ('"""' in line or "'''" in line):
            # Docstring should be at method body level
            if not line.startswith('        '):
                line = '        ' + line.lstrip()
                print(f"Line {i+1}: Fixed docstring indentation")

        # Check for control structures
        elif in_method and (stripped.startswith('try:') or
                           stripped.startswith('except ') or
                           stripped.startswith('finally:') or
                           stripped.startswith('elif ') or
                           stripped.startswith('else:') or
                           stripped.startswith('with ') or
                           stripped.startswith('for ') or
                           stripped.startswith('if ') or
                           stripped.startswith('while ')):
            # Control structures should be at method body level
            if not line.startswith('        '):
                line = '        ' + line.lstrip()
                print(f"Line {i+1}: Fixed control structure indentation")

        # Regular method body
        elif in_method and stripped:
            # Method body should be indented with 8 spaces
            if not line.startswith('        '):
                line = '        ' + line.lstrip()
                print(f"Line {i+1}: Fixed method body indentation")

        fixed_lines.append(line)

    # Write the fixed file
    with open('main_comprehensive_research.py', 'w') as f:
        f.writelines(fixed_lines)

    print("Fixed indentation for helper methods section")

if __name__ == "__main__":
    fix_helper_methods_indentation()