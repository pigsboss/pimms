all: perspective.pdf solvers.pdf
clean:
	rm -rf *.aux *.log *.bbl *.blg *.xdv
perspective.pdf: perspective.tex
	xelatex -no-pdf perspective.tex
	xelatex -no-pdf perspective.tex
	bibtex perspective
	xelatex -no-pdf perspective.tex
	xelatex -no-pdf perspective.tex
	xdvipdfmx perspective.xdv
solvers.pdf: solvers.tex
	xelatex solvers.tex

