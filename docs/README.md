# Kubeflow SDK Documentation

This directory contains the Sphinx documentation for the Kubeflow SDK.

## 📁 Directory Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation landing page
├── requirements.txt       # Documentation dependencies
├── Makefile              # Build automation (Linux/macOS)
├── source/
│   ├── installation.rst
│   ├── getting-started.rst
│   ├── overview.rst
│   ├── api-reference/    # Auto-generated API documentation
│   ├── guides/           # User guides
│   ├── examples/         # Code examples
│   └── future/           # Planned features
└── _build/               # Generated documentation (gitignored)
```

## 🚀 Quick Start

### Manual Build

If you prefer manual control:

```bash
# Create and activate virtual environment
python3 -m venv ../.venv
source ../.venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ..

# Build HTML
make html

# View the documentation
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux
```

## 🛠️ Available Make Commands

```bash
make html         # Build HTML documentation
make clean        # Remove build artifacts
make livehtml     # Build with live reload (auto-refresh)
make linkcheck    # Check for broken links
make coverage     # Check documentation coverage
make strict       # Build with warnings as errors
```

## 🔧 Development Workflow

### Adding New Documentation

1. **API Documentation**: Auto-generated from docstrings
   - Edit docstrings in `kubeflow/trainer/` source files
   - Rebuild docs to see changes

2. **User Guides**: Add to `source/guides/`
   - Create new `.rst` file
   - Add to `index.rst` toctree
   - Follow existing guide structure

3. **Examples**: Add to `source/examples/`
   - Create new `.rst` file
   - Add to `index.rst` toctree
   - Include complete, runnable code

### Live Development

Use live reload for rapid iteration:

```bash
make livehtml
```

This will:
- Start a local server at http://127.0.0.1:8000
- Open your browser automatically
- Auto-rebuild and refresh when you save changes

### Testing Your Changes

Before submitting a PR:

```bash
# Clean build
make clean && make html

# Check for broken links
make linkcheck

# Verify no unexpected warnings
make strict
```

## 📝 Documentation Style Guide

### reStructuredText (RST) Basics

```rst
Section Header
==============

Subsection
----------

**Bold text**
*Italic text*
``Code/literal text``

.. code-block:: python

    # Python code block
    from kubeflow.trainer import TrainerClient
    client = TrainerClient()

.. note::
   Important information

.. warning::
   Critical warnings
```

### Code Examples

- Use complete, runnable examples
- Include necessary imports
- Add comments explaining key concepts
- Show expected output when relevant

### Cross-References

```rst
:doc:`guide-name`                    # Link to another document
:doc:`../api-reference/types`        # Relative paths
:class:`kubeflow.trainer.CustomTrainer`  # Link to class
:func:`kubeflow.trainer.train`       # Link to function
```

### Docstring Format

We use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description.

    Longer description explaining what the function does,
    how it works, and when to use it.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When invalid input is provided.

    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    """
    pass
```

## 🔍 Troubleshooting

### Common Issues

**Import errors during build**
```bash
# Ensure SDK is installed in editable mode
pip install -e ..
```

**Missing dependencies**
```bash
# Reinstall documentation dependencies
pip install -r requirements.txt
```

**Warnings about missing files**
- Check that all referenced files exist
- Verify paths in `toctree` directives
- Some warnings are expected (see `.github/workflows/build-docs.yml`)

**Changes not appearing**
```bash
# Clean build cache
make clean
make html
```

## 🤖 CI/CD

Documentation is automatically built and validated on every PR:

- **Build check**: Ensures docs build without errors
- **Link validation**: Checks for broken links
- **Docstring coverage**: Verifies public APIs are documented
- **Spell check**: Catches typos

See `.github/workflows/build-docs.yml` for details.

## 📚 Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Furo Theme](https://pradyunsg.me/furo/)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

## 🤝 Contributing

1. Make your documentation changes
2. Test locally with `./build_docs_locally.sh`
3. Verify with `make linkcheck` and `make strict`
4. Submit a PR - CI will validate your changes
5. Address any CI failures

For major documentation changes, consider opening an issue first to discuss the approach.

## 📞 Getting Help

- GitHub Issues: https://github.com/kubeflow/sdk/issues
- Kubeflow Slack: #kubeflow-sdk channel
- Documentation: https://kubeflow-sdk.readthedocs.io
