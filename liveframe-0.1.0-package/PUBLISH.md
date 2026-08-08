# Manual PyPI release

PyPI project names are claimed by the first successful production upload. This
package is intended for a direct Twine upload, without GitHub Actions.

## Account preparation

1. Sign in to https://pypi.org and verify the account email.
2. Enable two-factor authentication.
3. In Account settings, create an API token scoped to the entire account. A
   project-scoped token cannot be created until the first `liveframe` release
   exists.

Do not put the token in this directory, shell history, Git, or `.pypirc`.

## Upload the validated files

From this directory, using an environment with Twine installed:

```bash
python -m twine check --strict dist/*
python -m twine upload --repository pypi dist/*
```

At Twine's prompts, enter `__token__` as the username and paste the PyPI API
token as the password. Upload exactly these two files together:

- `dist/liveframe-0.1.0-py3-none-any.whl`
- `dist/liveframe-0.1.0.tar.gz`

PyPI releases are immutable. Do not rebuild between validation and upload; a
changed `0.1.0` file cannot replace an uploaded one.

## Verify production

After upload succeeds:

```bash
python -m pip index versions liveframe
python -m pip install --no-cache-dir liveframe==0.1.0
liveframe version
```

The project page will be https://pypi.org/project/liveframe/0.1.0/.
After the first upload, replace the account-wide token with a project-scoped
token for future manual releases.
