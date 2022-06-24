from pathlib import Path

cwd = Path().cwd()
project_path = Path('/'.join(cwd.parts[:cwd.parts.index('PyCharmProject') + 1]))
