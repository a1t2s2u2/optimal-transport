# uplatex + dvipdfmx 用 latexmk 設定
$latex          = 'uplatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex         = 'upbibtex %O %B';
$biber          = 'biber %O --bblencoding=utf8 -u -U --output_safechars %B';
$makeindex      = 'upmendex %O -o %D %S';
$dvipdf         = 'dvipdfmx %O -o %D %S';
$pdf_mode       = 3;            # latex -> dvi -> pdf 経路
$out_dir        = 'out';
@default_files  = ('main.tex');
$max_repeat     = 5;
