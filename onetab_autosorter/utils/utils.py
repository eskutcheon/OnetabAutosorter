import os
import json
import hashlib
import re
from thefuzz import fuzz
from itertools import permutations
from urllib.parse import urlparse
from typing import List, Dict, Set, Union, Set, Optional, Any
from collections import defaultdict



DEFAULT_IGNORE_FOLDER_NAMES = ["bookmark", "folder", "stuff", "link", "site", "website", "bar",
                               "toolbar", "page", "menu", "list", "untitled", "other"]

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
    #? NOTE: Careful with this because for n-grams with range (1,n) and M list elements, permutations grow exponentially as O(M^n)
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
    #! FIXME: still needs some work on the underlying logic since some expected folders seem to be missing and it's not tossing out `root`
    # generate the regex pattern for all permutations of ignored words
    ignored_regex = generate_ignored_regex(DEFAULT_IGNORE_FOLDER_NAMES)
    # helper function to extract the final folder name and filter out ignored words
    def filter_name(folder_name: str) -> str:
        subfolders = folder_name.split("/")[:max_depth]
        filtered = subfolders[-1].strip().lower()
        # use regex to remove ignored words and their combinations
        filtered = re.sub(ignored_regex, "", filtered, flags=re.IGNORECASE).strip()
        return filtered
    folder_set = set([filter_name(folder) for folder in folder_list])
    folder_set.discard("")      # remove empty strings if present
    # TODO: should filter stopwords here too since they may be all that's left after filtering
    return folder_set



def prompt_user_to_edit():
    import threading
    import time
    DEFAULT_TIMEOUT = 15 # seconds
    prompt = "Would you like to remove any of these labels?"
    dynamic_prompt = f"{prompt} [{DEFAULT_TIMEOUT}s remaining]"
    print(f"{prompt} (Will automatically keep all labels after {DEFAULT_TIMEOUT} seconds)")
    # variables to store user's response and action (using lists to allow access from the thread)
    user_wants_to_edit = [False]
    user_responded = [False]
    # local function for the thread to run
    def get_user_confirmation():
        answers = {'y': True, 'yes': True, 'n': False, 'no': False, '': False}
        try:
            # ! WARNING: walrus operator  only works in Python 3.8+
            while (response := input(f"{dynamic_prompt} [Y/n] : ").strip().lower()) not in answers:
                print("\tInvalid input. Please enter 'y' or 'n' (not case sensitive).")
            user_wants_to_edit[0] = answers[response]
            user_responded[0] = True
        except KeyboardInterrupt:
            print("\nInterrupted. Keeping all labels.")
            user_responded[0] = True
    # start input thread
    input_thread = threading.Thread(target=get_user_confirmation)
    input_thread.daemon = True
    input_thread.start()
    # wait for response with timeout
    start_time = time.time()
    # only start showing countdown after a small delay to avoid UI confusion
    time.sleep(0.5)
    while input_thread.is_alive() and (time.time() - start_time) < DEFAULT_TIMEOUT:
        if not user_responded[0]:
            remaining = int(DEFAULT_TIMEOUT - (time.time() - start_time))
            #print(f"\rWaiting for response... {remaining} seconds remaining", end="", flush=True)
            dynamic_prompt = f"{prompt} [{remaining}s remaining]"
            time.sleep(0.1)  # More responsive countdown
    # clear the countdown line
    print()
    # check if the thread timed out
    if input_thread.is_alive() and not user_responded[0]:
        print("\nTime's up! Keeping all labels.")
        return False
    return user_wants_to_edit[0]



def prompt_to_drop_labels(all_labels: List[str], req_confirmation: bool = True) -> bool:
    sorted_labels = sorted(all_labels)
    # first ask if the user wants to edit at all with timeout
    print("\nFound the following labels from the folder structure: \n", ", ".join(sorted_labels), end="\n\n")
    if req_confirmation and not prompt_user_to_edit():
        print("Keeping all labels...")
        return sorted_labels
    final_labels = []
    # display the extracted labels with indices
    print("\n===== EXTRACTED SEED LABELS =====")
    for i, label in enumerate(sorted_labels):
        print(f"[{i}] {label}")
    # interactive prompt to remove labels
    print("\nYou can remove unwanted labels by entering:")
    print(" - Index numbers (e.g., '0 3 5' to remove labels at positions 0, 3, and 5)")
    print(" - Exact label text (e.g., 'Python JavaScript' to remove those labels)")
    print(" - A mix of both (e.g., '0 JavaScript 5')")
    print(" - Or just press Enter to keep all labels")
    print("\nNote: Matching is case-insensitive")
    try:
        user_input = input("\nLabels to remove (or press Enter to keep all): ").strip()
        if not user_input:
            print("Confirmed: Keeping all labels...")
            final_labels = sorted_labels
            return
        # track labels to remove (by index and by name)
        to_remove = set()
        # split input by whitespace
        tokens = user_input.split()
        # process each token
        for token in tokens:
            # try to interpret as an index
            try:
                idx = int(token)
                if 0 <= idx < len(sorted_labels):
                    to_remove.add(idx)
                else:
                    print(f"WARNING: Index {idx} is out of range (0-{len(sorted_labels)-1}), ignoring.")
            except ValueError:
                # interpret as a label name (case-insensitive)
                found = False
                for i, label in enumerate(sorted_labels):
                    if token.lower() == label.lower():
                        to_remove.add(i)
                        found = True
                if not found:
                    print(f"WARNING: No exact match found for '{token}', ignoring.")
        # create a new list while excluding the removed indices
        filtered_labels = [label for i, label in enumerate(sorted_labels) if i not in to_remove]
        # provide feedback on what was removed
        removed_labels = [label for i, label in enumerate(sorted_labels) if i in to_remove]
        if removed_labels:
            print(f"\nRemoved {len(removed_labels)} labels: \n\t{', '.join(removed_labels)}")
        # update seed labels
        final_labels = filtered_labels
        print(f"Keeping {len(final_labels)} candidate labels for keyword extraction or zero-shot labeling.")
    except KeyboardInterrupt:
        print("\nInterrupted. Keeping all labels.")
        final_labels = sorted_labels
    except Exception as e:
        print(f"\nError during label selection: {e}. Keeping all labels.")
        final_labels = sorted_labels
    return final_labels


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




class PythonSetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def compute_hash(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()

# def cache_path(base_dir: str, name: str, ext="json"):
#     return os.path.join(base_dir, f"{name}.{ext}")


def get_hashed_path_from_string(input_path: str, stage_name: str, cache_dir: str) -> str:
    hash_str = compute_hash(input_path)
    return get_hashed_path_from_hash(hash_str, stage_name, cache_dir)

def get_hashed_path_from_hash(hash_str: str, stage_name: str, cache_dir: str) -> str:
    file_prefix = "" if not stage_name else f"{stage_name}_"
    return os.path.join(cache_dir, f"{file_prefix}{hash_str}.json")


def load_json(file_path: str) -> Any:
    #? NOTE: might just import and use this within the JSON parser I made for no reason
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as fptr:
        return json.load(fptr)


def save_json(file_path: str, data: Dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fptr:
        json.dump(data, fptr, indent=2, cls=PythonSetEncoder)  # Use the custom encoder for sets