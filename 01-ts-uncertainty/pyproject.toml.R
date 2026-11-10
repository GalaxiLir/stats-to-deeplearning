[build-system]
requires = ["setuptools>=64"]          # outil qui construit/installe le paquet
build-backend = "setuptools.build_meta"

[project]
name = "tsunc"
dynamic = ["version"]                  # la version n'est écrite qu'une fois, dans __init__.py
requires-python = ">=3.10"
dependencies = ["numpy", "pandas", "matplotlib"]   # ≈ Imports: de DESCRIPTION

[tool.setuptools.dynamic]
version = { attr = "tsunc.__version__" }
[tool.setuptools.packages.find]
where = ["src"]                        # le code est dans src/ (disposition « src layout »)