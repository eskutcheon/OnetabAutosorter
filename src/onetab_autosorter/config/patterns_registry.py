import re


# some default regex patterns for ignoring spam and meta phrases:
DEFAULT_IGNORE_PATTERNS = [
    re.compile(r"^javascript.*required$", re.IGNORECASE),
    re.compile(r"enable\s+javascript", re.IGNORECASE),
    #? NOTE: all commented out patterns below were recent additions that haven't been tested in practical use - may be irrelevant later
    # re.compile(r'sign\s+(?:in|up)', re.IGNORECASE),
    # re.compile(r'subscribe\s+(?:now|today)', re.IGNORECASE),
    # re.compile(r'share\s+this', re.IGNORECASE),
    # re.compile(r'copyright\s+[\d©]', re.IGNORECASE),
    # re.compile(r'all\s+rights\s+reserved', re.IGNORECASE),
    re.compile(r"cookies\s+.*enable", re.IGNORECASE),
    re.compile(r"(404|403)\s+error", re.IGNORECASE),
    re.compile(r"access\s+denied", re.IGNORECASE),
    re.compile(r"your\s+browser.*not\s+supported", re.IGNORECASE),
    re.compile(r"you\s+are\s+being\s+redirected", re.IGNORECASE),
    # common web phrases - adding these back in since they're useful
    re.compile(r'accept\s+(?:all\s+)?cookies', re.IGNORECASE),
    re.compile(r'we\s+use\s+cookies', re.IGNORECASE),
    re.compile(r'privacy\s+policy', re.IGNORECASE),
    re.compile(r'terms\s+(?:of\s+)?(?:use|service)', re.IGNORECASE),
    re.compile(r"copyright\s+[\d©]", re.IGNORECASE),
    re.compile(r"all\s+rights\s+reserved", re.IGNORECASE),
    # re.compile(r"\b[A-Fa-f0-9]{32,64}\b"),  # SHA hash or long hex
    # re.compile(r"\.py\b|\.(js|html|php|txt)\b"),  # Code files 
]

NAVIGATION_PATTERNS = [
    re.compile(r"\b(main menu|navigation|tools|actions|search)\b", re.IGNORECASE),
    re.compile(r"\b(jump to content|create account|sign up|log in|edit|view history)\b", re.IGNORECASE),
    re.compile(r"\b(move to sidebar|hide|toggle)\b", re.IGNORECASE),
    re.compile(r"\b(click here|read more|share\s+this)\b", re.IGNORECASE),
    # NEW: newsletter advertising patterns
    re.compile(r"\bnewsletter\b", re.IGNORECASE),
    re.compile(r"\brelated\s+(?:posts|articles)\b", re.IGNORECASE),
]

FORMATTING_PATTERNS = [
    re.compile(r"\[\s*\]"),          # Empty brackets
    re.compile(r"\(\s*\)"),          # Empty parentheses
    re.compile(r"\{\s*\}"),          # Empty curly braces
    re.compile(r"[_~`]{1,2}(?!\`)"), # Markdown formatting chars (except triple backticks)
]


LATEX_PATTERNS = [
    re.compile(r"\$[^$\n]{1,100}\$", re.MULTILINE),     # inline math (limited length)
    re.compile(r"\$\$[^$\n]{1,500}\$\$", re.MULTILINE), # block math (limited length)
    re.compile(r"\{[^}\n]{1,100}\}"),                   # bibtex citations (limited length)
]

# moved citation pattern out of LATEX_PATTERNS since it's needed more often
CITATION_PATTERNS = [
    re.compile(r"\[[\d,\s]{1,20}\]", re.MULTILINE)     # Citations like [1], [2,3], or [1, 2, 3] (limited length)
]

# LATEX_BIB_PATTERN = re.compile(r"\{[^}]*\}")

METADATA_PATTERNS = [
    re.compile(r'\b(?:\d{1,2}[-/th|st|nd|rd\s]*)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]{0,6}[-/\s]*\d{2,4}\b', re.IGNORECASE), # dates
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), # email addresses
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), # UUIDs
    re.compile(r"#[0-9a-fA-F]{6}"), # hex color codes
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), # IP addresses
    re.compile(r"\b(\w+/){2,}\w+\b"), # URL paths
]


SEQUENCE_PATTERNS = [
    # re.compile(r"\b(\d+[,\s]*){3,}\b"),     # numeric sequences (3 or more digits in a row) #! FIXME: design is pretty bad and can cause exponential backtracking with regex
    re.compile(r"\b\d+(?:[,\s]+\d+){2,}\b"), # numeric sequences (3 or more digits in a row)
    re.compile(r"[^\x00-\x7F]+"),           # unicode characters (non-ASCII)
    # re.compile(r"\s{2,}"),                  # multiple whitespaces (to collapse to a single space) #! FIXME: also bad - it indiscriminately removes all whitespace, including newlines
    re.compile(r" {2,}"),                   # multiple whitespaces (to collapse to a single space)
]

# NUMERIC_SEQ_PATTERN = re.compile(r"\b(\d+[,\s]*){3,}\b")
# UNICODE_PATTERN = re.compile(r"\b[^\x00-\x7F]+\b")                   # keep only the most common unicode characters (ASCII range))
# MULTIPLE_SPACES_PATTERN = re.compile(r'\s{2,}')

# new category for code blocks, including backticks
CODE_PATTERNS = [
    # code blocks with triple backticks (with a high length limit of 8000)
    re.compile(r"```[^`\n]{0,8000}```", re.DOTALL),
    # inline code blocks with single backticks
    re.compile(r"`[^`\n]{1,300}`"),
    # common file extensions to remove code references
    re.compile(r"\b\w+\.(py|js|html|css|php|java|cpp|h|cs|rb|go|rs|ts|jsx|tsx)\b"),
    # SHA hash or long hex strings
    re.compile(r"\b[A-Fa-f0-9]{32,64}\b"),
    # spacing around Python import keywords to deal with specific problems I kept seeing
    re.compile(r'\b(import|from)\b(?=\w)', re.IGNORECASE),  # "import" not followed by space
    re.compile(r'(?<=\w)(import|from|as)\b', re.IGNORECASE),  # "import" not preceded by space
    # TODO: remove later after refining filtering strategy
    # common Python syntax spacing issues
    re.compile(r'(?<=\w)(def|class|for|in|if|else|elif|while|return|with|try|except|finally)\b', re.IGNORECASE),
    re.compile(r'\b(def|class|for|in|if|else|elif|while|return|with|try|except|finally)(?=\w)', re.IGNORECASE)
]