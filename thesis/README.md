IIT Kanpur Thesis LaTeX Template

Version: 1.0

Date: February 2026

Author & Maintainer

Utkarsh Tripathi PhD Scholar

Department of Chemical Engineering

Indian Institute of Technology Kanpur (IITK)

Email: utkarsht23@iitk.ac.in

Website: https://home.iitk.ac.in/~utkarsht23/

Overview

This repository contains a refined LaTeX template designed for writing theses (PhD, MTech, MS, MDes) at IIT Kanpur. It is structured to meet standard academic formatting requirements while solving specific compiling issues found in previous versions of the template.

Key Features & Fixes

This version addresses specific bugs related to page numbering in twoside printing formats:

Ghost Page Numbering Fixed: - Previous templates often printed headers and page numbers on the blank pages inserted automatically by LaTeX (to ensure chapters start on the right side).

This version uses a redefined \cleardoublepage command to ensure that blank pages are completely empty.

Dedication Page Formatting: - Fixed the issue where the "Dedication" page displayed a page number despite using \pagestyle{empty}. It is now strictly unnumbered.

Package Optimization: - Removed duplicate package calls.

Includes pre-configuration for TikZ, longtable, rotating (landscape figures), and natbib.

Directory Structure

main.tex: The master file. Compile this file to generate the PDF.

Thesis.cls: The class file defining the specific formatting styles (Margins, Fonts, Chapter Headers).

Chapters/: Contains separate .tex files for each chapter (e.g., Chapter1.tex, Chapter2.tex).

Appendices/: Contains appendix files.

Pictures/: Store all your images here.

references.bib: The BibTeX file for your citations.

Usage Instructions

Ensure you have a LaTeX distribution installed (TeX Live, MiKTeX, or Overleaf).

Open main.tex.

Update your personal details in the preamble:

\title{Your Thesis Title}
\author{Your Name}
% Update supervisor and department details in the Title Page section


Write your content in the files inside the Chapters/ folder.

Uncomment the \input lines in main.tex corresponding to the chapters you are working on.

Compile using XeLaTeX or PDFLaTeX.

Credits

Customized and maintained by Utkarsh Tripathi, PhD Scholar, ChE, IIT Kanpur.

Feel free to contact me for suggestions or improvements.