import os

EXCLUDED_DIRS = {'.venv', '__pycache__', '.git', '.mypy_cache', '.pytest_cache'}

def print_tree(start_path='.', prefix='', output_lines=None):
    entries = sorted(os.listdir(start_path))
    entries = [e for e in entries if not e.startswith('.') or e in EXCLUDED_DIRS]  # include only visible or explicitly allowed hidden dirs
    entries = [e for e in entries if e.endswith('.py') or os.path.isdir(os.path.join(start_path, e))]

    entries = [e for e in entries if e not in EXCLUDED_DIRS]

    for index, name in enumerate(entries):
        path = os.path.join(start_path, name)

        # Skip non-source directories
        if os.path.isdir(path) and not contains_python(path):
            continue

        connector = '└── ' if index == len(entries) - 1 else '├── '
        line = prefix + connector + name
        print(line)
        output_lines.append(line)

        if os.path.isdir(path):
            extension = '    ' if index == len(entries) - 1 else '│   '
            print_tree(path, prefix + extension, output_lines)

def contains_python(path):
    for root, dirs, files in os.walk(path):
        # Prune EXCLUDED_DIRS in-place to prevent descent
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        if any(file.endswith('.py') for file in files):
            return True
    return False

if __name__ == '__main__':
    output = []
    print_tree('.', '', output)

    with open('python_tree_clean.txt', 'w') as f:
        f.write('\n'.join(output))

    print("\n✅ Saved to: python_tree_clean.txt (no .venv junk this time)")
