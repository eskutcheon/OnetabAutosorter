import re


# Some default regex patterns for ignoring spam and meta phrases:
DEFAULT_IGNORE_PATTERNS = [
    re.compile(r"^javascript.*required$", re.IGNORECASE),
    re.compile(r"enable\s+javascript", re.IGNORECASE),
    re.compile(r"cookies\s+.*enable", re.IGNORECASE),
    re.compile(r"(404|403)\s+error", re.IGNORECASE),
    re.compile(r"access\s+denied", re.IGNORECASE),
    re.compile(r"your\s+browser.*not\s+supported", re.IGNORECASE),
    re.compile(r"you\s+are\s+being\s+redirected", re.IGNORECASE),
    re.compile(r"\b[A-Fa-f0-9]{32,64}\b"),  # SHA hash or long hex
    re.compile(r"\.py\b|\.(js|html|php|txt)\b"),  # Code files
]

NAVIGATION_PATTERNS = [
    re.compile(r"\b(main menu|navigation|tools|actions|search)\b", re.IGNORECASE),
    re.compile(r"\b(jump to content|create account|sign up|log in|edit|view history)\b", re.IGNORECASE),
    re.compile(r"\b(move to sidebar|hide|toggle)\b", re.IGNORECASE),
    re.compile(r"\b(click here|read more|terms of service|privacy policy)\b", re.IGNORECASE),
]

FORMATTING_PATTERNS = [
    re.compile(r"\[\s*\]"),          # Empty brackets
    re.compile(r"\(\s*\)"),          # Empty parentheses
    re.compile(r"\{\s*\}"),          # Empty curly braces
    re.compile(r"[_~`]"),            # Markdown or stray formatting characters
]


LATEX_PATTERNS = [
    re.compile(r"\$[^$]+\$", re.MULTILINE),     # inline math
    re.compile(r"\$\$[^$]+\$\$", re.MULTILINE), # block math
    #! might need to remove this one so as not to possibly lose metadata scraped from journal articles
    re.compile(r"\{[^}]*\}"),                   # bibtex citations
    re.compile(r"\[[^\]]*\]", re.MULTILINE)     # citations like [1], [2,3], or [1, 2, 3] etc.
]
#CITATION_PATTERN = re.compile(r"\[[^\]]*\]", re.MULTILINE)  # Matches citations like [1], [2,3], or [1, 2, 3] etc.

# LATEX_BIB_PATTERN = re.compile(r"\{[^}]*\}")

METADATA_PATTERNS = [
    re.compile(r'\b(?:\d{1,2}[-/th|st|nd|rd\s]*)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/\s]*\d{2,4}\b', re.IGNORECASE), # dates
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), # email addresses
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), # UUIDs
    re.compile(r"#[0-9a-fA-F]{6}"), # hex color codes
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), # IP addresses
    re.compile(r"\b(\w+/){2,}\w+\b"), # URL paths
]
#DATE_PATTERN = re.compile(r'\b(?:\d{1,2}[-/th|st|nd|rd\s]*)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/\s]*\d{2,4}\b', re.IGNORECASE)
#URL_PATH_PATTERN = re.compile(r"\b(\w+/){2,}\w+\b")

SEQUENCE_PATTERNS = [
    re.compile(r"\b(\d+[,\s]*){3,}\b"),     # numeric sequences (3 or more digits in a row)
    re.compile(r"[^\x00-\x7F]+"),       # unicode characters (non-ASCII)
    re.compile(r"\s{2,}"),                  # multiple whitespaces (to collapse to a single space)
]

# NUMERIC_SEQ_PATTERN = re.compile(r"\b(\d+[,\s]*){3,}\b")
# UNICODE_PATTERN = re.compile(r"\b[^\x00-\x7F]+\b")                   # keep only the most common unicode characters (ASCII range))
# MULTIPLE_SPACES_PATTERN = re.compile(r'\s{2,}')