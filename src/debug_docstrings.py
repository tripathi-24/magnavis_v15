#!/usr/bin/env python3
"""Debug script to find unclosed docstrings."""

with open('data_convert_db_now.py', 'r') as f:
    lines = f.readlines()

# Track docstring state
in_string = False
string_start_line = None

for i, line in enumerate(lines, start=1):
    count = line.count('"""')
    
    if count == 0:
        continue
    
    for _ in range(count):
        if not in_string:
            in_string = True
            string_start_line = i
            if i >= 900:
                print(f"Line {i:5d}: OPEN  docstring")
        else:
            in_string = False
            if i >= 900:
                print(f"Line {i:5d}: CLOSE docstring (opened at {string_start_line})")

if in_string:
    print(f"\n❌ UNCLOSED: Docstring opened at line {string_start_line} is never closed!")
    print(f"\nContent at opening line {string_start_line}:")
    print(f"{lines[string_start_line-1]}")
    print(f"{lines[string_start_line]}")
else:
    print(f"\n✅ All docstrings properly closed")
