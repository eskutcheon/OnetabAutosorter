import os
import json
from urllib.parse import urlparse
from typing import List, Dict
from collections import defaultdict


class PythonSetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


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
                # new addition:
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