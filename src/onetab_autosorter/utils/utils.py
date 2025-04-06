
import json
from typing import List, Dict
from collections import defaultdict


class PythonSetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)



def is_local_url(url: str) -> bool:
    return url.startswith("chrome-extension://") or "file:///" in url

def deduplicate_entries(entries: List[Dict], max_length: int = 200) -> List[Dict]:
    """ post-process the entries to remove duplicates based on URL, keeping the first occurrence and adding the `group_id`s of dropped duplicates """
    seen = {}
    domain_groups = defaultdict(list)
    for entry in entries:
        domain_groups[entry["domain"]].append(entry)
    final_entries = []
    for domain, group in domain_groups.items():
        for entry in group:
            url = entry["url"]
            if len(url) > max_length:
                final_entries.append(entry)
                continue
            if url in seen:
                seen[url]["group_ids"].update(entry["group_ids"])
            else:
                new_entry = entry.copy()
                seen[url] = new_entry
                final_entries.append(new_entry)
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