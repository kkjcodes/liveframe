# LiveFrame arXiv Package

This directory is the self-contained manuscript source for the LiveFrame publication package.

## Before submission

1. Confirm the affiliation line, title, author order, and arXiv license selection.
2. Publish Tier A and Tier B evidence at durable URLs, complete preview content review, and add the URLs to the reproducibility section.
3. Re-run the literature search immediately before submission, especially for CUDA/MLX reproducibility, Apple-Silicon video benchmarks, video VAE acceleration, and causal-cache reuse.
4. Verify every citation against its primary source. Venue labels in this package are conservative; an arXiv comment is not treated as independent venue confirmation.
5. Compile from a clean directory and inspect every page of `main.pdf`.

## Build

The validated local build uses Tectonic:

```bash
tectonic -X compile main.tex --keep-logs --keep-intermediates
```

With a conventional TeX Live installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is unavailable:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Clean generated files with:

```bash
latexmk -C
```

## arXiv upload

Upload the contents of this directory, not the directory itself. The source package should contain:

- `main.tex`
- `references.bib`
- `README.md` (optional for arXiv, useful in the release archive)

No model weights, private paths, raw evidence corpora, generated videos, or third-party source are included. The manuscript uses only standard TeX packages available in current TeX Live distributions and has no external figure dependency.

Recommended arXiv category: `cs.CV` primary, with `cs.LG` as a cross-list if appropriate. The author must have an arXiv account and may need endorsement for the selected category.
