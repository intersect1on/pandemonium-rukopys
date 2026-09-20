%% ucuproposal.cls
%% Клас для пропоузалів семестрових проєктів курсу «Штучний інтелект» (УКУ).
%% Уся верстка та брендинг живуть тут — main.tex містить лише метадані й контент.
%% Вимагає компіляції XeLaTeX (fontspec + polyglossia).

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ucuproposal}[2026/01/01 v1.0 UCU semester project proposal template]

\RequirePackage{etoolbox}

%% ------------------------------------------------------------------
%% Опції класу
%% ------------------------------------------------------------------
\newtoggle{ucu@english}
\togglefalse{ucu@english}
\newtoggle{ucu@slogan}
\togglefalse{ucu@slogan}

\DeclareOption{ukrainian}{\togglefalse{ucu@english}}
\DeclareOption{english}{\toggletrue{ucu@english}}
\DeclareOption{slogan}{\toggletrue{ucu@slogan}}
\DeclareOption*{%
  \ClassWarningNoLine{ucuproposal}{Невідома опція класу '\CurrentOption' буде проігнорована}%
}
\ExecuteOptions{ukrainian}
\ProcessOptions\relax

%% article підтримує лише 10/11/12pt. Раніше тут стояло 10.5pt — article мовчки
%% його ігнорував і вантажив size10.clo, тобто текст УЖЕ був 10pt, а в лозі
%% висіло "Unused global option(s): [10.5pt]". Прибрано, щоб опція збігалася
%% з реальністю; на вигляд документа це не впливає.
\LoadClass[10pt,a4paper]{article}

%% ------------------------------------------------------------------
%% Базові пакети
%% ------------------------------------------------------------------
\RequirePackage{geometry}
\geometry{a4paper,left=22mm,right=22mm,top=25mm,bottom=25mm,headheight=14pt,headsep=8mm}

\RequirePackage{fontspec}
\RequirePackage{polyglossia}

\iftoggle{ucu@english}{%
  \setdefaultlanguage{english}
  \setotherlanguage{ukrainian}
}{%
  \setdefaultlanguage{ukrainian}
  \setotherlanguage{english}
}

\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{fancyhdr}
%% Бібліографія — класичний BibTeX, а НЕ biblatex+biber. biblatex тягне за
%% собою окремий процес biber і два додаткові проходи XeLaTeX (через
%% .bcf/.run.xml) — це була найдорожча частина збірки, дорожча за весь
%% решту документа разом. Робочий процес для студента не змінюється:
%% той самий references.bib і ті самі \cite{}.
%% cite — коректне групування посилань: \cite{a,b,c} → [1--3].
\RequirePackage{cite}

%% ------------------------------------------------------------------
%% Фірмовий колір УКУ (єдиний акцентний колір шаблону)
%% ------------------------------------------------------------------
\definecolor{ucured}{HTML}{7F1716}
\definecolor{ucutext}{HTML}{2B2B2B}
\definecolor{ucugray}{HTML}{8A8A8A}

\AtBeginDocument{\color{ucutext}}

%% ------------------------------------------------------------------
%% Шрифти: Charis SIL (текст) + Fira Sans (заголовки/титулка) + Fira Mono
%%
%% ВАЖЛИВО ДЛЯ ШВИДКОСТІ: гарнітури вантажаться за ІМЕНЕМ ФАЙЛУ, а не за
%% назвою гарнітури. Пошук за назвою ("Charis SIL") віддається fontconfig і
%% на Overleaf коштує секунди на кожну гарнітуру; пошук за іменем файлу йде
%% через kpathsea в дереві TeX Live і фактично безкоштовний.
%% Перевірки наявності теж зроблені по файлу — резервний DejaVu лишається,
%% тож компіляція ніколи не падає через відсутній шрифт.
%% ------------------------------------------------------------------
\newtoggle{ucu@mainfontsc}
\togglefalse{ucu@mainfontsc}

\IfFontExistsTF{CharisSIL-Regular.ttf}{%
  \setmainfont{CharisSIL}[%
    Extension      = .ttf,
    UprightFont    = *-Regular,
    BoldFont       = *-Bold,
    ItalicFont     = *-Italic,
    BoldItalicFont = *-BoldItalic,
  ]%
  %% Капітельний варіант визначаємо ОДИН раз окремою гарнітурою замість
  %% \addfontfeature на кожному використанні — інакше XeLaTeX створює новий
  %% інстанс шрифту в кожному місці титулки.
  \newfontfamily\ucu@scfont{CharisSIL}[%
    Extension   = .ttf,
    UprightFont = *-Regular,
    BoldFont    = *-Bold,
    Letters     = SmallCaps,
  ]%
  \toggletrue{ucu@mainfontsc}%
}{%
  \ClassWarningNoLine{ucuproposal}{Шрифт 'Charis SIL' не знайдено. Використано резервний DejaVu Serif. Встановіть пакет TeX Live 'charissil' для коректного брендингу}%
  \IfFontExistsTF{DejaVuSerif.ttf}{%
    \setmainfont{DejaVuSerif}[%
      Extension      = .ttf,
      UprightFont    = *,
      BoldFont       = *-Bold,
      ItalicFont     = *-Italic,
      BoldItalicFont = *-BoldItalic,
    ]%
  }{%
    \ClassWarningNoLine{ucuproposal}{Резервний шрифт 'DejaVu Serif' також не знайдено. Використано типовий шрифт LaTeX}%
  }%
}

%% Малі капітелі для титулки: справжня OpenType-функція Charis SIL,
%% або класичний \scshape як безпечний резерв для інших шрифтів.
\newcommand{\ucu@sc}[1]{%
  \iftoggle{ucu@mainfontsc}{{\ucu@scfont#1}}{{\scshape#1}}%
}

\IfFontExistsTF{FiraSans-Regular.otf}{%
  \setsansfont{FiraSans}[%
    Extension      = .otf,
    UprightFont    = *-Regular,
    BoldFont       = *-Bold,
    ItalicFont     = *-Italic,
    BoldItalicFont = *-BoldItalic,
  ]%
  % Накреслення заголовків: у брендбуці — Fira Sans SemiBold.
  \newfontfamily\ucuheadfont{FiraSans}[%
    Extension   = .otf,
    UprightFont = *-SemiBold,
    ItalicFont  = *-SemiBoldItalic,
  ]%
}{%
  \ClassWarningNoLine{ucuproposal}{Шрифт 'Fira Sans' не знайдено. Використано резервний DejaVu Sans. Встановіть пакет TeX Live 'fira' для коректного брендингу}%
  \IfFontExistsTF{DejaVuSans.ttf}{%
    \setsansfont{DejaVuSans}[%
      Extension      = .ttf,
      UprightFont    = *,
      BoldFont       = *-Bold,
      ItalicFont     = *-Oblique,
      BoldItalicFont = *-BoldOblique,
    ]%
  }{%
    \ClassWarningNoLine{ucuproposal}{Резервний шрифт 'DejaVu Sans' також не знайдено. Використано типовий шрифт LaTeX}%
  }%
  \newcommand{\ucuheadfont}{\sffamily\bfseries}%
}

\IfFontExistsTF{FiraMono-Regular.otf}{%
  \setmonofont{FiraMono}[%
    Extension   = .otf,
    UprightFont = *-Regular,
    BoldFont    = *-Bold,
  ]%
}{%
  \ClassWarningNoLine{ucuproposal}{Шрифт 'Fira Mono' не знайдено. Використано резервний DejaVu Sans Mono}%
  \IfFontExistsTF{DejaVuSansMono.ttf}{%
    \setmonofont{DejaVuSansMono}[%
      Extension   = .ttf,
      UprightFont = *,
      BoldFont    = *-Bold,
    ]%
  }{}%
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.4em}
\linespread{1.15}

%% ------------------------------------------------------------------
%% Мовно-залежні підписи та ім'я файлу логотипу
%% ------------------------------------------------------------------
\iftoggle{ucu@english}{%
  \newcommand{\ucu@biblabel}{References}
  \newcommand{\ucu@labelteam}{Team}
  \newcommand{\ucu@labelmentor}{Mentor}
  \newcommand{\ucu@labelrepo}{Repository}
  \newcommand{\ucu@labeldate}{Submission date}
  \newcommand{\ucu@pagelabel}{page}
  \newcommand{\ucu@superheader}{AI Course \textperiodcentered\ Semester Project Proposal \textperiodcentered\ 2026}
  \newcommand{\ucu@uniname}{Ukrainian Catholic University}
  \newcommand{\ucu@submittedline}{Project proposal submitted for the}
  \newcommand{\ucu@courseline}{\textquotedblleft Artificial Intelligence\textquotedblright\ course}
  \newcommand{\ucu@inthe}{in the}
  \newcommand{\ucu@facultyname}{Faculty of Applied Sciences}
}{%
  \newcommand{\ucu@biblabel}{Список джерел}
  \newcommand{\ucu@labelteam}{Команда}
  \newcommand{\ucu@labelmentor}{Ментор}
  \newcommand{\ucu@labelrepo}{Репозиторій}
  \newcommand{\ucu@labeldate}{Дата подачі}
  \newcommand{\ucu@pagelabel}{сторінка}
  \newcommand{\ucu@superheader}{Курс «Штучний інтелект» \textperiodcentered\ Семестровий проєкт \textperiodcentered\ 2026}
  \newcommand{\ucu@uniname}{Український Католицький Університет}
  \newcommand{\ucu@submittedline}{Пропозиція проєкту, подана в межах курсу}
  \newcommand{\ucu@courseline}{«Штучний інтелект»}
  \newcommand{\ucu@inthe}{}
  \newcommand{\ucu@facultyname}{Факультет прикладних наук}
}

%% Печатка УКУ та лого факультету — поки лише в англомовному варіанті
%% (окремих українських версій файлів не надано), тому спільні для обох мов.
\newcommand{\ucu@sealfile}{assets/logo/uku_logo___seal_engl.png}
\newcommand{\ucu@facultylogofile}{assets/logo/Applied_Sciences_ENG.png}

%% Офіційне гасло УКУ. Формулювання не перекладається й не перефразовується
%% незалежно від мовної опції класу.
\newcommand{\ucu@slogantext}{%
  {\addfontfeature{LetterSpace=10.0}\color{ucured}\small СВІДЧИТИ. СЛУЖИТИ. СПІЛКУВАТИСЯ.}%
}

%% ------------------------------------------------------------------
%% Заголовки секцій: Fira Sans SemiBold 12pt, колір ucured, лише рівень 1
%% ------------------------------------------------------------------
\setcounter{secnumdepth}{1}
\setcounter{tocdepth}{1}

\titleformat{\section}
  {\normalfont\ucuheadfont\fontsize{12}{14}\selectfont\color{ucured}}
  {\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{1.1em}{0.5em}

\titleformat{\subsection}
  {\normalfont\ucuheadfont\fontsize{10.5}{13}\selectfont\color{ucured}}
  {}{0pt}{}
\titlespacing*{\subsection}{0pt}{0.9em}{0.4em}

%% ------------------------------------------------------------------
%% Списки: компактні, маркер — маленький квадрат ucured
%% ------------------------------------------------------------------
\setlist[itemize]{nosep,leftmargin=1.2em,label={\color{ucured}\rule{0.8ex}{0.8ex}}}
\setlist[enumerate]{nosep,leftmargin=1.4em}

%% ------------------------------------------------------------------
%% Блок "keyfacts" — врізка з ключовими цифрами/метриками.
%% Реалізовано на голому боксі замість mdframed: врізка ніколи не розривається
%% між сторінками, тому вся машинерія mdframed тут була зайвою.
%% ------------------------------------------------------------------
\newsavebox{\ucu@kfbox}
\newenvironment{keyfacts}{%
  \par\addvspace{8pt}%
  \begin{lrbox}{\ucu@kfbox}%
  \begin{minipage}{\dimexpr\linewidth-12pt\relax}%
  \setlength{\parskip}{0.4em}%
}{%
  \end{minipage}%
  \end{lrbox}%
  \noindent
  %% Лінійка виступає на 6pt над і під вмістом — той самий внутрішній
  %% відступ, що давав mdframed (innertopmargin/innerbottommargin).
  {\color{ucured}\rule[-\dimexpr\dp\ucu@kfbox+6pt\relax]{2pt}%
    {\dimexpr\ht\ucu@kfbox+\dp\ucu@kfbox+12pt\relax}}%
  \hspace{10pt}\usebox{\ucu@kfbox}%
  \par\addvspace{8pt}%
}

%% ------------------------------------------------------------------
%% hyperref (завантажується майже останнім)
%% bookmarks=false + pdfpagelabels=false навмисно: і закладки, і мітки
%% сторінок (через \setcounter{page}{1} після титулки) змушували hyperref
%% просити "Rerun", тобто цілий зайвий прохід XeLaTeX на кожній збірці.
%% Для тристорінкового пропоузалу панель закладок того не варта.
%% ------------------------------------------------------------------
\RequirePackage[bookmarks=false,pdfpagelabels=false]{hyperref}
\hypersetup{
  colorlinks=true,
  allcolors=ucured,
  breaklinks=true,
}
\iftoggle{ucu@english}{\hypersetup{pdflang=en}}{\hypersetup{pdflang=uk}}
\urlstyle{same}

%% ------------------------------------------------------------------
%% Метадані пропоузалу: командний інтерфейс main.tex
%% ------------------------------------------------------------------
\newcommand{\projecttitle}[1]{\def\ucu@title{#1}}
\let\ucu@title\@empty

\newcommand{\projecttagline}[1]{\def\ucu@tagline{#1}}
\let\ucu@tagline\@empty

%% \par (не \\) навмисно: у p-колонці tabular з \centering ручний перенос \\
%% псує спільну базову лінію рядка таблиці відносно сусідньої комірки.
%% #1 = ім'я, #2 = URL профілю (може бути порожнім — тоді без посилання).
\newcommand{\ucu@teammemberline}[2]{%
  \def\ucu@linktmp{#2}%
  \ifx\ucu@linktmp\@empty
    \ucu@sc{\textbf{\color{ucured}#1}}\par\smallskip
  \else
    \href{#2}{\ucu@sc{\textbf{\color{ucured}#1}}}\par\smallskip
  \fi
}

\newcounter{ucu@nmembers}
\newcommand{\ucu@memberlist}{}
\newcommand{\ucu@authorlist}{}
%% Використання: \teammember{Ім'я Прізвище} або \teammember[URL]{Ім'я Прізвище}
%% (опційний URL, напр. LinkedIn-профіль, робить ім'я клікабельним посиланням).
\newcommand{\teammember}[2][]{%
  \stepcounter{ucu@nmembers}%
  \g@addto@macro\ucu@memberlist{\ucu@teammemberline{#2}{#1}}%
  \ifnum\value{ucu@nmembers}=1
    \renewcommand{\ucu@authorlist}{#2}%
  \else
    \g@addto@macro\ucu@authorlist{, #2}%
  \fi
}

\newcommand{\mentor}[1]{\def\ucu@mentor{#1}}
\let\ucu@mentor\@empty

\newcommand{\track}[1]{\def\ucu@track{#1}}
\let\ucu@track\@empty

\newcommand{\repo}[1]{\def\ucu@repo{#1}}
\let\ucu@repo\@empty

\newcommand{\proposaldate}[1]{\def\ucu@date{#1}}
\let\ucu@date\@empty

\AtBeginDocument{%
  \hypersetup{pdftitle=\ucu@title,pdfauthor=\ucu@authorlist}%
}

%% ------------------------------------------------------------------
%% Титульна сторінка
%% ------------------------------------------------------------------
\fancypagestyle{ucutitle}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}

\fancypagestyle{ucubody}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0.6pt}
  \renewcommand{\headrule}{{\color{ucured}\hrule width\headwidth height\headrulewidth}}
  \renewcommand{\footrulewidth}{0pt}
  \fancyhead[L]{\sffamily\small\color{ucured}\ucu@courseline}%
  %% Без "з N": lastpage/\pageref{LastPage} змушували latexmk робити
  %% зайвий прохід XeLaTeX на кожній збірці.
  \fancyhead[R]{\sffamily\small\color{ucured}\ucu@pagelabel\ \thepage}%
}

\newcommand{\maketitlepage}{%
  \thispagestyle{ucutitle}
  \begin{center}
  %% Явні \vspace керують усім вертикальним ритмом титулки; \parskip з
  %% основного тексту тут лише заважає (накопичується на кожному \par і
  %% зсуває вміст, аж до переповнення сторінки), тож на титулці він вимкнений.
  \setlength{\parskip}{0pt}%
  \vspace*{18mm}
  {\ucu@sc{\fontsize{15}{18}\selectfont\color{ucured}\ucu@uniname}}\par
  \vspace{10mm}
  {\ucu@sc{\fontsize{10.5}{13}\selectfont\ucu@superheader}}\par
  \vspace{4mm}
  {\color{black}\rule{\linewidth}{0.5pt}}\par
  \vspace{7mm}
  {\bfseries\fontsize{20}{25}\selectfont\ucu@title}\par
  \ifdefvoid{\ucu@tagline}{}{%
    \vspace{4mm}
    {\itshape\fontsize{10.5}{13}\selectfont\color{ucugray}\ucu@tagline}\par
  }%
  \vspace{5mm}
  {\color{black}\rule{\linewidth}{0.5pt}}\par
  \vspace{10mm}
  \noindent
  \begin{tabular}[t]{@{}>{\centering\arraybackslash}p{0.44\linewidth}@{\hspace{0.06\linewidth}}>{\centering\arraybackslash}p{0.44\linewidth}@{}}
  {\itshape\ucu@labelteam:}\par\smallskip
  \ucu@memberlist
  &
  \ifdefvoid{\ucu@mentor}{}{%
    {\itshape\ucu@labelmentor:}\par\smallskip
    \ucu@sc{\textbf{\color{ucured}\ucu@mentor}}\par\smallskip
  }%
  \ifdefvoid{\ucu@repo}{}{%
    {\itshape\ucu@labelrepo:}\par\smallskip
    {\small\url{\ucu@repo}}\par
  }%
  \end{tabular}\par
  \vspace{10mm}
  {\itshape\ucu@submittedline\par\ucu@courseline}\par
  \ifdefvoid{\ucu@inthe}{}{%
    \vspace{3mm}
    {\itshape\ucu@inthe}\par
  }%
  \vspace{3mm}
  {\color{ucured}\ucu@facultyname}\par
  \vfill
  \includegraphics[height=22mm]{\ucu@sealfile}%
  \hspace{8mm}%
  \includegraphics[height=18mm]{\ucu@facultylogofile}\par
  \vspace{6mm}
  \iftoggle{ucu@slogan}{\ucu@slogantext\par\vspace{4mm}}{}
  \ifdefvoid{\ucu@date}{}{\ucu@date\par}
  \vspace*{14mm}
  \end{center}
  \clearpage
  \setcounter{page}{1}
  \pagestyle{ucubody}
}

%% ------------------------------------------------------------------
%% Список джерел.
%% Стиль unsrturl (пакет urlbst): нумерація за порядком цитування, як було
%% в biblatex numeric-comp, і — на відміну від звичайного unsrt — друкує
%% поле url клікабельним посиланням.
%% ------------------------------------------------------------------
\newcommand{\ucuprintbibliography}{%
  \clearpage
  \renewcommand{\refname}{\ucu@biblabel}%
  \bibliographystyle{unsrturl}%
  \bibliography{references}%
}

%% ------------------------------------------------------------------
%% Contribution statement
%% Відсотки НЕ обов'язкові й ніде не перевіряються — пишіть внесок словами
%% або додайте частку в необов'язковому аргументі, якщо так зручніше.
%% ------------------------------------------------------------------
%% Використовується під \section{Contribution statement} у main.tex —
%% власного заголовка не друкує, щоб уникнути дублювання з \section.
\newenvironment{contributions}{%
  \par\addvspace{0.4em}
  \begin{itemize}
}{%
  \end{itemize}
}

%% \contribution{Ім'я}{що робить}  або  \contribution[50\%]{Ім'я}{що робить}
\newcommand{\contribution}[3][]{%
  \def\ucu@ctmp{#1}%
  \ifx\ucu@ctmp\@empty
    \item \textbf{#2} --- #3%
  \else
    \item \textbf{#2} --- #1: #3%
  \fi
}

\endinput
