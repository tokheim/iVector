% -----------------
% NTNU Thesis Class
% -----------------

% Provide
\ProvidesClass{thesis}
\NeedsTeXFormat{LaTeX2e}

% Base class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions
\LoadClass{memoir}

% Fixes
\RequirePackage{etex}


% -----------------
% Colors
% -----------------

\RequirePackage{xcolor}

% Color
\definecolor{red}{HTML}{660000}
\definecolor{green}{HTML}{336633}

% Black and white
\definecolor{black}{HTML}{000000}
\definecolor{lightgray}{HTML}{CCCCCC}
\definecolor{gray}{HTML}{999999}
\definecolor{darkgray}{HTML}{666666}


% -----------------
% Text
% -----------------

% Encoding
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8x]{inputenc}

% Standard fonts
\renewcommand{\rmdefault}{ppl} % font: palatino

% Commands
\newcommand{\lowercaps}[1]{\textsc{\MakeLowercase{#1}}}
\newcommand{\uppercaps}[1]{\textsc{\MakeUppercase{#1}}}

% Hyphenation, kerning
\RequirePackage{microtype}

% Leading (for b5/10pt)
\setSpacing{1.25}

% Paragraphs
\clubpenalty  = 5000
\widowpenalty = 5000
\setlength{\parindent}{2.5em}
\setlength{\parskip}{0.001em}
%\setlength{\parindent}{0em}
%\setlength{\parskip}{1em}


% Compact lists
\RequirePackage{mdwlist}
%\renewcommand{\labelitemi}{$\circ$}

% Smallcaps
\newcommand{\smallcaps}[1]{\textsc{\MakeUppercase{#1}}}

% Ornaments
\RequirePackage{fourier-orns}

% Columns
\RequirePackage{multicol}


% -----------------
% Math
% -----------------

\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{amssymb}


% -----------------
% Graphics
% -----------------

\RequirePackage{graphicx}
\RequirePackage{pdfpages}
\RequirePackage{wrapfig}
\RequirePackage{subfig}
\RequirePackage{caption}
\captionsetup{font={small,it},labelfont=it}

\RequirePackage{tikz}
\RequirePackage{pgf}
\RequirePackage{pgfplots}
\usetikzlibrary{arrows,shapes,positioning,plotmarks}


% -----------------
% Links
% -----------------

\RequirePackage{caption}
\RequirePackage{hyperref}
\RequirePackage{memhfixc}
\hypersetup{
  colorlinks=true, 
  urlcolor=red, 
  linkcolor=red,
  citecolor=red
}
\urlstyle{same}


% -----------------
% Page layout
% -----------------

% Page margins
\settypeblocksize{47pc}{32pc}{*} % (height,width,ratio)
\setlrmargins{*}{*}{1}
\setulmargins{7pc}{*}{*}
\raggedbottom
\checkandfixthelayout

\makepagestyle{ruled}
\makeevenfoot{ruled}{}{}{}
\makeoddfoot{ruled}{}{}{}
%\makeheadrule{ruled}{\textwidth}{0.2pt}
\makeevenhead{ruled}{{\small\thepage \ \ \ \MakeUppercase{\leftmark}}\vspace{1pt}}{}{}
\makeoddhead{ruled}{}{}{{\small\MakeUppercase{\rightmark}} \ \ \ \thepage\vspace{1pt}}
\makepsmarks{ruled}{%
\nouppercaseheads
\createmark{chapter}{left}{nonumber}{}{}
\createmark{section}{right}{nonumber}{}{}
%\createmark{chapter}{left}{shownumber}{\@chapapp\ }{. }
%\createmark{section}{right}{shownumber}{}{. }
\createplainmark{toc}{both}{\contentsname}
\createplainmark{lof}{both}{\listfigurename}
\createplainmark{lot}{both}{\listtablename}
\createplainmark{bib}{both}{\bibname}
\createplainmark{index}{both}{\indexname}
\createplainmark{glossary}{both}{\glossaryname}
}
\pagestyle{ruled}

% Pages with chapter headings
\makepagestyle{plain}
  \makeevenhead{plain}{}{}{}
  \makeoddhead{plain}{}{}{}
  %\makeevenfoot{plain}{\small\thepage}{}{}
  %\makeoddfoot{plain}{}{}{\small\thepage}

% Float positioning
\RequirePackage{afterpage}


% -----------------
% Divisors
% -----------------

\setsecnumdepth{subsection}

% Chapters (based on memoir/dash)
\makechapterstyle{thesis}{%
  \chapterstyle{default}
  \setlength{\beforechapskip}{-4em}
  \renewcommand*{\printchaptername}{}
  \renewcommand*{\chapternamenum}{}
  \renewcommand*{\chapnumfont}{\normalfont\LARGE}
  \settoheight{\midchapskip}{\chapnumfont 1}
  \renewcommand*{\printchapternum}{
    \centering \chapnumfont
    \rule[0.5\midchapskip]{1em}{0.4pt} \thechapter\
    \rule[0.5\midchapskip]{1em}{0.4pt}
    %\includegraphics[width=2em]{{\thechapter}}
  }
  \renewcommand*{\afterchapternum}{\par\nobreak
    \vskip 0 % change on indent
    \onelineskip
  }
  \renewcommand*{\printchapternonum}{\centering
                 \vphantom{\chapnumfont 1}\afterchapternum}
  \renewcommand*{\chaptitlefont}{\normalfont\huge}
  \renewcommand*{\printchaptertitle}[1]{\centering \chaptitlefont ##1}
  \setlength{\afterchapskip}{2.0\onelineskip}
}
\chapterstyle{thesis}

% Sections
\setsecheadstyle{\vspace{1em}\Large}

% Subsections
\setsubsecheadstyle{\large\itshape}

% Subsubsections
\setsubsubsecheadstyle{\itshape}

% Paragraph headings
\setparaheadstyle{\normalfont\itshape}


% -----------------
% Table of Contents
% -----------------

\setcounter{tocdepth}{1}

%\RequirePackage{titlesec}
%\RequirePackage{titletoc}
%\contentsmargin{2.55em}
%\titlecontents{section}
%  [1.5em]
%  {}
%  {\hspace*{2.55em}\color{red}\contentslabel{2.3em}}
%  {\hspace*{-2.3em}}
%  {\hspace*{1em}\color{gray}\contentspage}


% -----------------
% Frontmatter
% -----------------

% Abstract
\renewcommand{\abstractnamefont}{\normalfont\huge} 
\renewcommand{\abstracttextfont}{\normalfont\normalsize}
\setlength{\absleftindent}{6em}
\setlength{\absrightindent}{6em}


% -----------------
% References
% -----------------

%\RequirePackage{natbib}

%\renewcommand*{\bibname}{References}
%\renewenvironment{thebibliography}[1]{%
%  \setlength{\bibitemsep}{-\parsep - 0.5em}
%  \setlength\bibindent{0em}
%  \setSpacing{1}
%  \setlength{\parindent}{-1em}
%  \sloppy
%  \clubpenalty4000
%  \widowpenalty4000
%  \bibsection
%  \begin{bibitemlist}{#1}}{\end{bibitemlist}\postbibhook}

%\renewcommand*{\biblistextra}{% 
%  \vspace{1em}
%  \setlength{\leftmargin}{1.5em}% 
%  \setlength{\itemindent}{-1.5em}% 
%  \addtolength{\itemindent}{\labelsep}}


% -----------------
% Sourcecode
% -----------------

% Listings
\RequirePackage{listings}
\RequirePackage{courier}
\lstset{
  basicstyle=\scriptsize\ttfamily,
  numbers=right,
  numberstyle=\tiny\color{gray},
  %stepnumber=2,
  numbersep=-5pt,
  tabsize=2,
  extendedchars=true,
  breaklines=true,
  keywordstyle=\color{green},
  frame=lines,         
  stringstyle=\color{red}\ttfamily,
  showspaces=false,
  showtabs=false,
  xleftmargin=0pt,
  xrightmargin=0pt,
  framexleftmargin=0pt,
  framexrightmargin=3pt,
  framextopmargin=3pt,
  framexbottommargin=3pt,
  rulecolor=\color{black},
  framerule=1pt,
  backgroundcolor=\color{white},
  showstringspaces=false,
  abovecaptionskip=5pt,
  belowcaptionskip=0pt,
  captionpos=b,
  aboveskip=20pt,
  belowskip=5pt
}
\lstloadlanguages{XML,Ruby}

% Captions
\RequirePackage{caption}
\DeclareCaptionFormat{listing}{\colorbox[cmyk]{0,0,0,0}{\parbox{\textwidth}{\hspace{0pt}#1#2#3}}}
\captionsetup[lstlisting]{
  format=listing, font={it}, labelfont={it}, textfont={it}, 
  singlelinecheck=false, margin=0pt, font={footnotesize}}

% Sourcecode environment
\lstnewenvironment{sourcecode}[3][]{
  \setSpacing{1}
  \lstset{language={#1},label={#2},caption={#3}}
}{}
\lstnewenvironment{implementation}[2]{
  \setSpacing{1}
  \lstset{language={Ruby},label=#1,caption={#2}}
}{}



% Pseudocode
\RequirePackage{algorithm}
%\RequirePackage[boxed]{algorithm}
\RequirePackage{algorithmic}
\algsetup{
  indent=2em,
  linenosize=\small,
  linenodelimiter=. 
}
\renewcommand{\algorithmicrequire}{\textbf{Input:}}
\renewcommand{\algorithmicensure}{\textbf{Output:}}


% -----------------
% Environments
% -----------------

% Footnotes
\footmarkstyle{({#1}) \ } 
\setlength{\footmarkwidth}{1.5em} 
\setlength{\footmarksep}{0.5em} 
\setlength{\footparindent}{0em} 
\renewcommand*{\foottextfont}{\footnotesize}
\renewcommand*{\footnoterule}{%
  \vspace{0.5em}
  \color{lightgray}
  \hrule width \columnwidth 
  \kern 5pt
  \color{darkgray}
  \vspace{0.5em}
}

% Block quotes
\newenvironment{blockquote}{%
  \begin{quote}
  \itshape
}{%
  \end{quote}
}

% Quotes with citations
% \begin{citequote}[page]{ref}{text}..
\newenvironment{citequote}[3][]{%
  \citet[#1]{#2} #3:
  \begin{quote}
  \itshape
}{%
  \end{quote}
}

% Horizontal separator
\newcommand{\hr}{
  \vspace{1em}
  \begin{center}
    \color{black}
    \Huge
    \aldine
    \color{black}
  \end{center}
  \normalsize
  \vspace{2em}
}

% narrow column
\newenvironment{narrow}{
  \begin{adjustwidth}{8em}{8em} 
}{
  \end{adjustwidth}
}

% spaced equations
\newenvironment{eqsp}{
  \vspace{0.1em}
  \begin{equation*}
}{
  \end{equation*}
  \vspace{0.1em}
}

