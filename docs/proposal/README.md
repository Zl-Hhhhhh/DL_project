# Project Proposal LaTeX

This folder contains an English proposal in the official NeurIPS 2026 preprint format. The bundled style file is unmodified.

## Files

- `project_proposal.tex`: main document; references are included directly with `thebibliography`.
- `neurips_2026.sty`: official style from [NeurIPS](https://neurips.cc/Conferences/2026/CallForPapers).
- `compile.ps1`: Windows build script using project-local Tectonic, installed Tectonic, or pdfLaTeX.

## Author information

Replace the rules in `\AuthorName` and `\StudentID` with your actual name and student ID. Recompile and check the page count after editing. Research methods and data collection in the proposal are planned work; prototype results are identified separately.

## Compile

From the project root in PowerShell:

```powershell
.\docs\proposal\compile.ps1
```

The PDF is written to `output/pdf/project_proposal.pdf`. Initial compilation may download TeX packages. Compiler and package caches remain under `.cache/latex-tools`.

For Overleaf, upload the accompanying source ZIP, set `project_proposal.tex` as the main document, select pdfLaTeX, and compile.

The proposal follows the course's one-page requirement; the final page count must be rechecked after content or author changes.

## Verification

Compiled with Tectonic 0.17.0 on 2026-10-08. The PDF has one page including all three references. The page was rendered and visually inspected. Body font sizes and page geometry follow the official style; references use its small font.

## VS Code

Open the entire `D:/mystudy/DSAA2012/project` folder in VS Code so its workspace settings apply. LaTeX Workshop is configured in `.vscode/settings.json` to use the portable Tectonic compiler and project-local package cache.

1. Open `docs/proposal/project_proposal.tex`.
2. Run `Developer: Reload Window` from the Command Palette if the settings have not refreshed.
3. Run `LaTeX Workshop: Build LaTeX project`, or press `Ctrl+Alt+B`. The recipe is `Tectonic (project)`. Saving the source also triggers a build.
4. Run `LaTeX Workshop: View LaTeX PDF file`, or press `Ctrl+Alt+V`, to open the built-in PDF preview.

PDF, log, auxiliary, and SyncTeX files are written to `output/pdf`. If another extension captures the shortcuts, invoke the LaTeX Workshop commands from the Command Palette.
