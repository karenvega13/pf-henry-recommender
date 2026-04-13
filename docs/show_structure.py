"""
show_structure.py
-----------------
Corre este script desde la raíz de tu proyecto:
    python show_structure.py

Imprime el árbol completo de carpetas y archivos,
ignorando rutas irrelevantes (.git, __pycache__, etc.)
"""

import os

IGNORE = {
    ".git", "__pycache__", ".ipynb_checkpoints",
    ".venv", "venv", "env", ".env",
    "node_modules", ".DS_Store", ".vscode",
    ".idea", "*.egg-info",
}

def should_ignore(name):
    return name in IGNORE or name.endswith(".egg-info")

def print_tree(root=".", prefix=""):
    entries = sorted(os.listdir(root))
    entries = [e for e in entries if not should_ignore(e)]

    for i, entry in enumerate(entries):
        path = os.path.join(root, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry)

        if os.path.isdir(path):
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    root = "."
    print(f"\n📁 {os.path.abspath(root)}\n")
    print_tree(root)
    print()
