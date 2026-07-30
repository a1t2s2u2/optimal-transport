$pdflatex       = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$lualatex       = 'lualatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex         = 'bibtex %O %B';
$pdf_mode       = 1;
$out_dir        = 'out';
@default_files  = ('main.tex');
$max_repeat     = 5;
