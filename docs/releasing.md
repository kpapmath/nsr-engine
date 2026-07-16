# Releasing / updating the library

How to cut a new version of `nsr-engine` and publish it to PyPI.

Publishing is fully automated: **when you publish a GitHub Release, the
`Publish to PyPI` workflow builds the package and uploads it to PyPI** via
trusted publishing (OIDC — no API tokens stored). Your job is to bump the
version, verify the build, tag, and cut the release.

## Version source

The version is single-sourced. Edit **one** place:

```
src/nsr_engine/__init__.py   →   __version__ = "X.Y.Z"
```

`pyproject.toml` reads it dynamically (`version = { attr =
"nsr_engine.__version__" }`), so do **not** put a version string there.

Use [semantic versioning](https://semver.org/):

| Change | Bump | Example |
|--------|------|---------|
| Bug fix, no API change | patch | `0.1.0 → 0.1.1` |
| New backwards-compatible feature | minor | `0.1.0 → 0.2.0` |
| Breaking API change | major | `0.1.0 → 1.0.0` |

## Release steps

### 1. Start from a clean, up-to-date `main`

```bash
git checkout main
git pull
git status          # should be clean
```

### 2. Bump the version

Edit `src/nsr_engine/__init__.py` and set `__version__` to the new value.

### 3. Run the tests and the build locally

```bash
pip install -e ".[sympy,memmap,refine,dev]"
python -m pytest tests/ -v

# confirm the package builds and its metadata is valid
python -m pip install build twine
python -m build
python -m twine check dist/*
```

`twine check` catches README/metadata problems that would otherwise fail the
PyPI upload. Delete the local `dist/` afterward if you like — the release
workflow rebuilds from scratch.

### 4. Commit and push

```bash
git add src/nsr_engine/__init__.py
git commit -m "Release vX.Y.Z"
git push
```

CI (`.github/workflows/ci.yml`) runs the test matrix (Python 3.11 and 3.12)
and a build check on the push. **Wait for it to go green** before releasing.

### 5. Tag and create the GitHub Release

Use the tag `vX.Y.Z` (matching the version). Publishing the release is what
triggers the upload to PyPI.

With the GitHub CLI:

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes
```

Or in the web UI: **Releases → Draft a new release → new tag `vX.Y.Z` →
Publish release**.

### 6. Confirm publication

- Watch the **Publish to PyPI** workflow under the repo's Actions tab.
- Verify the new version at <https://pypi.org/project/nsr-engine/>.
- Sanity-check the install in a fresh environment:

  ```bash
  pip install --upgrade nsr-engine
  python -c "import nsr_engine; print(nsr_engine.__version__)"
  ```

## Updating dependencies

Runtime and optional dependencies are declared in **`pyproject.toml`**
(`dependencies` and `[project.optional-dependencies]`). This is the source of
truth for what users install.

`requirements.txt` is a convenience/pinning file for local development and
should be kept roughly in sync with `pyproject.toml` when you change deps.

After changing any dependency bounds:

```bash
pip install -e ".[sympy,memmap,refine,dev]"
python -m pytest tests/ -v
```

Then release with a version bump as above (a dependency change that affects
users warrants at least a patch release).

## Troubleshooting

- **PyPI upload fails with a permissions/OIDC error** — the `publish` job uses
  the `pypi` GitHub Actions environment and `id-token: write`. Confirm the
  trusted publisher is configured on PyPI for this repo and that the
  environment name matches `publish.yml`.
- **"File already exists" on PyPI** — that version was already uploaded. PyPI
  does not allow re-uploading a version; bump to a new one and re-release.
- **Build passes locally but CI fails** — check the Python version matrix
  (3.11 / 3.12); a feature or syntax may require a newer minimum than
  `requires-python = ">=3.11"`.
