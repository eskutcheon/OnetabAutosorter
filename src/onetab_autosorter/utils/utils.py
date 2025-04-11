import os
import json
import re
from thefuzz import fuzz
from itertools import permutations
from urllib.parse import urlparse
from typing import List, Dict, Set, Union, Set, Optional, Any
from collections import defaultdict



DEFAULT_IGNORE_FOLDER_NAMES = ["bookmark", "folder", "stuff", "link", "site", "website", "bar", "toolbar", "page", "menu", "list", "untitled", "other"]

def generate_ignored_regex(ignored_words: List[str]) -> str:
    """ Generate a regex pattern to match all plural forms and combinations of ignored words.
        Args:
            ignored_words (List[str]): List of single words to ignore.
        Returns:
            str: A regex pattern that matches all combinations and plural forms.
    """
    # add plural forms of each word manually since re.escape escapes the ? and destroys the regex pattern
    words_with_plural = [word + suffix for suffix in ["", "s"] for word in ignored_words]
    # generate all permutations of 1 or 2 words
    all_combinations = set()
    #! Careful with this because for n-grams with range (1,n) and M list elements, permutations grow exponentially as O(M^n)
    for i in range(1, 3):  # ngram-range of 1-2
        all_combinations.update([" ".join(p) for p in permutations(words_with_plural, i)])
    # Escape special characters and join combinations into a regex pattern
    escaped_combinations = [re.escape(comb) for comb in all_combinations]
    pattern = r"\b(" + "|".join(escaped_combinations) + r")\b"
    return pattern


def get_keywords_from_paths(folder_list: List[str], max_depth: int = 4) -> Set[str]:
    """ Extract keywords from folder paths, filtering out ignored words and their combinations.
        Args:
            folder_list (List[str]): List of folder paths.
            max_depth (int): Maximum depth of folder paths to consider.
            ignored_words (List[str]): List of single words to ignore.
        Returns:
            Set[str]: Set of filtered keywords.
    """
    # generate the regex pattern for all permutations of ignored words
    ignored_regex = generate_ignored_regex(DEFAULT_IGNORE_FOLDER_NAMES)
    print("ignored regex pattern: ", ignored_regex)
    print("number of ignored regex patterns: ", len(ignored_regex.split("|")))
    # helper function to extract the final folder name and filter out ignored words
    def filter_name(folder_name: str) -> str:
        print("initial folder path: ", folder_name)
        subfolders = folder_name.split("/")[:max_depth]
        filtered = subfolders[-1].strip().lower()
        # use regex to remove ignored words and their combinations
        filtered = re.sub(ignored_regex, "", filtered, flags=re.IGNORECASE).strip()
        return filtered
    folder_set = set([filter_name(folder) for folder in folder_list])
    folder_set.discard("")      # remove empty strings if present
    # TODO: should filter stopwords here too since they may be all that's left after filtering
    return folder_set


class FolderNode:
    def __init__(self, name):
        self.name = name
        self.children: List['FolderNode'] = []

    def add_child(self, child: 'FolderNode'):
        self.children.append(child)

    def to_dict(self):
        return {self.name: [child.to_dict() for child in self.children]}

    def to_list(self, prefix="") -> List[str]:
        full_path = os.path.join(prefix, self.name).strip(os.path.sep).replace("\\", "/")
        paths = [full_path]
        for child in self.children:
            paths.extend(child.to_list(full_path))
        return paths

    def extract_as_keywords(self, depth: int = 4, prefix="") -> List[str]:
        """ Extracts the folder names as keywords prepared for the extraction models """
        # convert to list with path-like strings then, extract the final folder name and filter it while removing duplicates
        folder_list = self.to_list(prefix)
        folder_list = list(map(lambda x: x.replace("ROOT/", "", 1), folder_list)) # remove the ROOT prefix from the paths before further filtering
        folder_set = get_keywords_from_paths(folder_list, depth)
        return list(folder_set)



def detect_bookmark_format(file_path: str) -> str:
    """ Returns 'netscape' for browser bookmarks, 'onetab' for OneTab export, or 'unsupported', which should raise an error in the caller """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.read(2048).lower()  # read first 2KB
    if "<!doctype netscape-bookmark-file-1" in header:
        return "netscape"
    elif "<!doctype html" in header and "<title>onetab" in header:
        return "onetab"
    else:
        return "unsupported"



def is_local_url(url: str) -> bool:
    return url.startswith("chrome-extension://") or "file:///" in url


# TODO: add to clean_utils.py
def deduplicate_entries(entries: List[Dict], max_length: int = 200) -> List[Dict]:
    """ post-process the entries to remove duplicates based on URL, keeping the first occurrence and adding the `group_id`s of dropped duplicates """
    seen = {}
    final_entries = []
    domain_groups = defaultdict(list)
    for entry in entries:
        domain = entry.get("domain", urlparse(entry["url"]).netloc)
        domain_groups[domain].append(entry)
    for domain, group in domain_groups.items():
        for entry in group:
            url = entry["url"]
            if len(url) > max_length:
                final_entries.append(entry)
                continue
            if url in seen:
                seen[url]["group_ids"].update(entry["group_ids"], set())
            else:
                entry["group_ids"] = set(entry.get("group_ids", set()))
                seen[url] = entry.copy()
                final_entries.append(entry)
    return final_entries




def is_internet_connected(host="8.8.8.8", port=53, timeout=3, vpn_host=None, vpn_port=80):
    """ Returns True if it can open a TCP connection to the DNS server at `host`. """
    import socket
    socket.setdefaulttimeout(timeout)

    def test_socket(h, p):
        """ Test if a socket can connect to the given host and port. """
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.connect((h, p))
        test_socket.close()
        return True
    # first try the default host and port
    try:
        return test_socket(host, port)
    except OSError:
        # If the default host fails, try the VPN-specific host if provided
        if vpn_host:
            try:
                return test_socket(vpn_host, vpn_port)
            except OSError:
                pass
        return False


def _remove_near_duplicates(lines: List[str], threshold: int = 85) -> list[str]:
    if not lines:
        return []
    # returns just the first line if it's only one
    output = [lines[0]]
    for line in lines[1:]:
        # TODO: look into using a partial ratio or token_sort_ratio for better near-duplicate detection
        if all(fuzz.ratio(line, seen) < threshold for seen in output):
            output.append(line)
    return output

def preprocess_html_text(raw: Union[str, List[str]]) -> str:
    # remove near-duplicate lines, then drop short or spammy lines
    lines = raw.splitlines() if isinstance(raw, str) else raw
    #!!! Consider removing in favor of TF-IDF approach
    lines = _remove_near_duplicates(lines)
    text = "\n".join(lines) # _clean_text(lines)
    return text