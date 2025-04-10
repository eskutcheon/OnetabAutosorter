import os
import re
import json
from tqdm import tqdm
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup, Tag
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Sequence
from urllib.parse import urlparse
# local imports
from onetab_autosorter.utils.utils import is_local_url, FolderNode


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        raise NotImplementedError


class JSONParser(BaseParser):
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

# TODO: I'm considering rewriting some of this in a lower-level language since it's pretty straightforward and they all have some "soup" library for parsing HTML

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
                    "date": creation_date,
                    # store group_ids as a set from the get-go to allow multiple, disjoint IDs for duplicates in different tab groups
                    "group_ids": {idx},
                    "raw_date": date_text,
                })
        return parsed



class NetscapeBookmarkParser(BaseParser):
    def __init__(self):
        self.bookmarks = []
        self.group_ids = set()

    def parse(self, bookmark_file: str) -> List[Dict[str, Any]]:
        """ Parse a Netscape bookmark file and extract bookmark metadata.
            Args:
                bookmark_file (str): path to the bookmark HTML file
            Returns:
                List[Dict[str, Any]]: list of bookmark metadata dictionaries
        """
        with open(bookmark_file, "r", encoding="utf-8") as file:
            #! The parser `lxml` argument is really important here - I debugged things forever before finding out that bs4 does some default conversions
            soup = BeautifulSoup(file, "lxml")
        # to reuse parser for subsequent files if needed
        self._reset_parser()
        # get the root <DL> tag (encloses all other <DL> and <DT> tags) and start parsing the folder structure
        main_dl = soup.find('dl')
        if not main_dl:
            raise ValueError("Expected a primary <dl> tag found in the file")
        self._parse_folder(main_dl, current_folder="")
        return self.bookmarks

    def _parse_folder(self, dl: Tag, current_folder: str):
        """ Recursively parse a <DL> tag to extract bookmarks and folder structure
            Args:
                dl (Tag): The <DL> tag representing a folder or root.
                current_folder (str): The current folder path being processed.
        """
        # generate unique group ID for the current folder
        local_group_id = id(dl) # TODO: may want this to be integers starting at 0, but it works fine for making unique IDs based on memory address
        self.group_ids.add(local_group_id)
        # find all <DT> tags directly under the current <DL>
        dt_tags: Sequence[Tag] = dl.find_all("dt", recursive=False)
        for dt in dt_tags:
            if dt.h3: # if folder, parse it recursively
                folder_name = dt.h3.get_text(strip=True)
                sub_dl = dt.find_next_sibling("dl")
                if sub_dl:
                    new_folder = os.path.join(current_folder, folder_name)
                    self._parse_folder(sub_dl, new_folder)
            elif dt.a: # if bookmark, extract its metadata
                url = dt.a.get('href', '')
                date = self._parse_timestamp(dt.a.get("add_date")) or self._parse_timestamp(dt.a.get("last_modified"))
                bookmark_entry = {
                    "url": url,
                    "title": dt.a.get_text(strip=True),
                    "domain": urlparse(url).netloc,
                    "date": date,
                    "group_ids": {local_group_id},
                    "raw_date": dt.a.get("add_date"),
                    "folder": current_folder.strip(os.path.sep).replace("\\", "/"),
                }
                self.bookmarks.append(bookmark_entry)
            # TODO: still considering else blocks for error catching but need to think over other possible structures

    def _reset_parser(self):
        self.bookmarks.clear()
        self.group_ids.clear()

    @staticmethod
    def _parse_timestamp(timestamp_str: str) -> Optional[str]:
        """ parse a POSIX timestamp string into a formatted datetime string
            Args:
                timestamp_str (str): timestamp string to parse
            Returns:
                str or None: formatted datetime string or None if invalid
        """
        #! FIXME: remove second condition - forgot I added that in a sample HTML at one point
        if not (timestamp_str and timestamp_str.isdigit()):
            return None
        try:
            return datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None

    @staticmethod
    def _extract_folder(dl: Tag, parent: FolderNode):
        """ Recursively extract the folder structure from a <DL> tag.
            Args:
                dl (Tag): a <DL> tag representing a folder or root
                parent (FolderNode): the parent FolderNode to attach child folders nodes to
        """
        # iterate over all <DT> tags directly under the current <DL>
        for dt in dl.find_all("dt", recursive=False):
            h3 = dt.find("h3")
            if h3: # if folder, parse recursively
                folder_name = h3.get_text(strip=True)
                folder_node = FolderNode(folder_name)
                parent.add_child(folder_node)
                # find the next <DL> tag representing the subfolder's contents
                sub_dl = dt.find_next_sibling("dl")
                if sub_dl:
                    # recursively extract subfolder structure
                    NetscapeBookmarkParser._extract_folder(sub_dl, folder_node)

    @staticmethod
    def extract_folder_structure_tree(file_path: str) -> FolderNode:
        """ Extract the folder structure as a tree from a Netscape bookmark file.
            Args:
                file_path (str): path to the bookmark HTML file
            Returns:
                FolderNode: root of the folder structure tree
        """
        with open(file_path, "r", encoding="utf-8") as f:
            # Parse the HTML file with BeautifulSoup
            soup = BeautifulSoup(f.read(), "lxml")
        root_dl = soup.find("dl")
        # naming the primary root node for the folder structure "ROOT"
        root = FolderNode("ROOT")
        if root_dl:
            # extract folder structure starting from the root <DL>
            NetscapeBookmarkParser._extract_folder(root_dl, root)
        return root