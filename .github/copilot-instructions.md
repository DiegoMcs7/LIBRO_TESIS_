## Purpose
Quick, actionable instructions for AI code agents working on this LaTeX thesis project. Focus on what to change, how to build locally, and project-specific conventions to avoid wasted edits.

## Project overview (big picture)
- This is a LaTeX thesis repository. `main.tex` orchestrates the document by calling many small chapter files (e.g. `introduccion.tex`, `fundamento.tex`, `metodologia.tex`).
- `preambulo.tex` contains the shared package imports, macros, and formatting rules. Avoid large, untested edits here — many downstream files rely on these macros (examples: `\inout{}`, `\trominodl`, `\chapterformatdefault`).
- Figures and graphics live in the `images/` folder (PDF/PNG files are referenced via `\includegraphics` and custom macros).
- Bibliography is `bibliography.bib` and the project uses biblatex with backend=biber (see `preambulo.tex`). Indexing uses `imakeidx` and an `index.ist` style (referenced via `\makeindex`).

## Build & debug (concrete commands)
- Preferred: use latexmk which the project has been run with (build artifacts are placed under `build/`). On Windows PowerShell a typical command is:

```powershell
latexmk -pdf -outdir=build main.tex
```

- If debugging build steps manually, follow the sequence observed in `build/main.fdb_latexmk`:
  1. `pdflatex` (produce `.aux`, `.bcf`)
  2. `biber build/main` (bibliography)
  3. `makeindex build/main.idx` (index)
  4. `pdflatex` (repeat until stable)

- Notes: the fdb shows MiKTeX paths on the original machine. Agents should not assume a particular TeX distribution — prefer latexmk and call biber on the `build/main` basename when needed.

## Files to edit vs files to avoid
- Edit these for content: `introduccion.tex`, `trabajos_relacionados.tex`, `fundamento.tex`, `metodologia.tex`, `resultados.tex`, `discusion.tex`, `conclusion.tex`, and the small front/backmatter files like `portada.tex`, `resumen_es.tex`.
- Edit `preambulo.tex` only for deliberate, small, well-tested changes (package additions, macro tweaks). If you change macros here, run a full build (`latexmk`) to verify no cascading failures.
- Do not hand-edit files in `build/` — they are generated. If you need to update the `.ist` index style or similar, add the file at repo root and reference it from `preambulo.tex`.

## Project-specific conventions and patterns
- Single-source chapter split: each chapter is a separate `.tex` included via `\input{...}`. Keep chapter-local figures and tables referenced with chapter numbering (the preamble sets counters by chapter).
- Spanish language: `babel` is set to `spanish`; algorithmicx texts and algorithm keywords have been redefined to Spanish (e.g., `\algorithmicif` → `\textbf{si}`). Preserve Spanish wording when editing algorithm environments.
- Custom problem statements: `\inout{<title>}{<input>}{<output>}` is used across the thesis for formal problem statements. Reuse rather than inventing new layouts.
- Figures: some macros insert small PDF fragments (e.g., `\trominodl` macros defined in `preambulo.tex`) — check `images/` for the referenced files before changing scale or names.
- Bibliography: `biblatex` with `backend=biber` and `style=numeric` — run `biber build/main` (note the `build/` prefix) after the first pdflatex pass.

## Integration points & external dependencies
- External tools required: latexmk (recommended), pdflatex/xelatex/pdflatex engine, biber (for biblatex), makeindex. On Windows typical TeX installations are MiKTeX or TeX Live.
- Graphics may reference `.pdf_tex` files (import package is used). If editing figures, ensure generated PDF and accompanying `.pdf_tex` are both placed under `images/`.

## Quick examples agents should follow
- To add a new subsection in Chapter 3, edit `metodologia.tex` and place figures in `images/`; then run `latexmk -pdf -outdir=build main.tex` and verify `build/main.pdf` updates.
- To update citations: update `bibliography.bib`, run `latexmk` (or: `pdflatex main.tex; biber build/main; pdflatex main.tex` twice) and check references in `build/main.pdf`.

## Common pitfalls to avoid
- Don’t modify `main.tex` to reorganize chapters unless you update `\tableofcontents` expectations — the style customizations in `preambulo.tex` assume the current split.
- Changing package options (e.g., switch from `biblatex` backend) will require adjusting build steps; prefer incremental changes and test with `latexmk`.

## When in doubt / next steps
- Run a full build (`latexmk`) after any change touching `preambulo.tex`, bibliography, or image includes.
- Ask for clarification when a requested change touches formatting macros in `preambulo.tex` — those are high-impact.

---
If you'd like, I can: (1) add a short PowerShell script to wrap the build steps, (2) create a brief CONTRIBUTING.md with LaTeX building/dependency notes, or (3) run a test build in the workspace and report results. Which should I do next?
