#!/usr/bin/env python3
"""
Comprehensive script to fix all indentation issues in the helper methods section
"""

import re

def comprehensive_fix():
    """Fix all indentation issues comprehensively"""

    with open('main_comprehensive_research.py', 'r') as f:
        content = f.read()

    # Find the start of helper methods section
    helper_methods_start = content.find('# Helper methods for multi-agent workflow')
    if helper_methods_start == -1:
        print("Could not find helper methods section")
        return

    # Get the helper methods section
    lines = content.split('\n')
    start_line_num = content[:helper_methods_start].count('\n') + 1

    # Process each line in helper methods section
    fixed_lines = lines[:start_line_num]  # Keep lines before helper methods
    current_indent = 0
    in_method = False
    method_indent = 0

    for i in range(start_line_num, len(lines)):
        line = lines[i]
        line_num = i + 1

        # Skip empty lines and comments that don't need fixing
        if not line.strip() or line.strip().startswith('#'):
            fixed_lines.append(line)
            continue

        # Check for method definition
        if re.match(r'async def |def ', line):
            # This should be a class method - 4 spaces
            if not line.startswith('    async def ') and not line.startswith('    def '):
                line = '    ' + line.lstrip()
                print(f"Line {line_num}: Fixed method definition indentation")
            in_method = True
            method_indent = 8  # Method body should be 8 spaces

        # Check for docstring start
        elif in_method and ('"""' in line or "'''" in line):
            # Docstring should be at method body level
            current_indent = len(line) - len(line.lstrip())
            if current_indent != method_indent:
                line = ' ' * method_indent + line.lstrip()
                print(f"Line {line_num}: Fixed docstring indentation to {method_indent}")

        # Check for method body
        elif in_method and line.strip():
            current_indent = len(line) - len(line.lstrip())

            # Special handling for control structures
            if (line.strip().startswith('try:') or
                line.strip().startswith('except ') or
                line.strip().startswith('finally:') or
                line.strip().startswith('elif ') or
                line.strip().startswith('else:') or
                line.strip().startswith('with ') or
                line.strip().startswith('for ') or
                line.strip().startswith('if ') or
                line.strip().startswith('while ')):

                if current_indent < method_indent:
                    line = ' ' * method_indent + line.lstrip()
                    print(f"Line {line_num}: Fixed control structure indentation to {method_indent}")

            # Regular method body
            elif current_indent < method_indent:
                line = ' ' * method_indent + line.lstrip()
                print(f"Line {line_num}: Fixed method body indentation to {method_indent}")

        fixed_lines.append(line)

    # Write the fixed content
    fixed_content = '\n'.join(fixed_lines)
    with open('main_comprehensive_research.py', 'w') as f:
        f.write(fixed_content)

    print("Comprehensive indentation fix completed!")

if __name__ == "__main__":
    comprehensive_fix()