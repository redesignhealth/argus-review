"""Packaged reviewer prompt bodies (Markdown files), one per runtime prompt name.

Loaded via ``argus.prompts_runtime.fetch_prompt``. This ``__init__.py`` exists
so ``importlib.resources.files("argus.prompts")`` resolves reliably as a
package resource anchor across wheel and sdist builds (a plain namespace
directory works in editable/source checkouts but isn't a robust anchor once
packaged), not because this module exposes any Python API of its own.
"""
