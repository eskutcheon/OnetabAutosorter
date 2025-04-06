import re
import json
import argparse
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
# local imports
from onetab_autosorter.utils.utils import is_local_url


class BaseParser:
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

# TODO: I'm considering rewriting this in a lower-level language since it's a lot more straightforward, except for the BeautifulSoup parsing

class OneTabParser(BaseParser):
    def _find_date_text(self, group_div: BeautifulSoup) -> str:
        """ Looks for a <div> that contains only the "Created ..." text without commands in the containing div """
        for div in group_div.find_all('div'):
            text = div.get_text(strip=True)
            if re.match(r"^Created \d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2} [AP]M$", text):
                return text
        return None #""

    def _parse_date(self, text: str) -> str:
        """ Converts 'Created MM/DD/YYYY, hh:mm:ss AM' to a datetime object """
        try:
            #? NOTE: isoformat is new in 3.7, so this will not work in 3.6 or earlier
            return datetime.strptime(text[len("Created "):], '%m/%d/%Y, %I:%M:%S %p').isoformat()
        except:
            return None

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        tab_groups = soup.select('div#tabGroupsDiv > div.tabGroup')
        parsed = []
        for idx, group in tqdm(enumerate(tab_groups), desc="Parsing Tab Groups", total=len(tab_groups), unit="group"):
            # 1. Extract creation date from the tab group div
            date_text = self._find_date_text(group)
            creation_date = self._parse_date(date_text) if date_text else None
            # 2. Extract all tabs (links) within the tab group div
            tabs = group.select('div.tabList div.tab a.tabLink')
            for a in tabs:
                url = a["href"]
                if is_local_url(url):
                    continue
                parsed.append({
                    "url": url,
                    "title": a.get_text(strip=True),
                    "domain": urlparse(url).netloc,
                    "group_date": creation_date,
                    # store group_ids as a set from the get-go to allow multiple, disjoint IDs for duplicates in different tab groups
                    "group_ids": {idx},
                    "raw_date_text": date_text,
                })
        return parsed


class JSONParser(BaseParser):
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

