import json
import urllib.parse
from typing import List, Optional
from dotenv import load_dotenv
import httpx
from langchain.tools.base import StructuredTool
import os
load_dotenv()

class Searxng:
    def __init__(
        self,
        host: str = os.getenv("SEARXNG_HOST"),
        engines: List[str] = [],
        fixed_max_results: Optional[int] = None,
        images: bool = False,
        it: bool = False,
        map: bool = False,
        music: bool = False,
        news: bool = False,
        science: bool = False,
        videos: bool = False,
    ):
        "Initialize parameters for ..."
        self.host = host
        self.engines = engines
        self.fixed_max_results = fixed_max_results
        # Initialize toolkit with base search
        self.toolkit = [StructuredTool.from_function(self.webpage_search)]
        # Add optional tools based on flags
        tool_mapping = [
            (images, StructuredTool.from_function(self.image_search)),
            (it, StructuredTool.from_function(self.it_search)),
            (map, StructuredTool.from_function(self.map_search)),
            (music, StructuredTool.from_function(self.music_search)),
            (news, StructuredTool.from_function(self.news_search)),
            (science, StructuredTool.from_function(self.science_search)),
            (videos, StructuredTool.from_function(self.video_search)),
        ]
        for search_content in tool_mapping:
            if search_content[0]:
                self.toolkit.append(search_content[1])

        # print(self.toolkit)

    def webpage_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search the web.

        Args:
            query (str): The query to search the web with.
            max_results (int, optional): The maximum number of results to return.

        Returns:
            The results of the search.
        """
        return self.search(query, max_results=max_results)

    def image_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search for images.

        Args:
            query (str): The query to search images with.
            max_results (int, optional): The maximum number of results to return.

        Returns:
            The results of the search.
        """
        return self.search(query, "images", max_results)

    def it_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search for IT related information.

        Args:
            query (str): The query to search for IT related information.
            max_results (int, optional): The maximum number of results to return. Defaults to 5.

        Returns:
            The results of the search.
        """
        return self.search(query, "it", max_results)

    def map_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search maps

        Args:
            query (str): The query to search maps with.
            max_results (int, optional): The maximum number of results to return. Defaults to 5.

        Returns:
            The results of the search.
        """
        return self.search(query, "map", max_results)

    def music_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search for information related to music.

        Args:
            query (str): The query to search music with.
            max_results (int, optional): The maximum number of results to return. Defaults to 5.

        Returns:
            The results of the search.
        """
        return self.search(query, "music", max_results)

    def news_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search for news.

        Args:
            query (str): The query to search news with.
            max_results (int, optional): The maximum number of results to return. Defaults to 5.

        Returns:
            The results of the search.
        """
        return self.search(query, "news", max_results)

    def science_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search for information related to science.

        Args:
            query (str): The query to search science with.
            max_results (int, optional): The maximum number of results to return. Defaults to 5.

        Returns:
            The results of the search.
        """
        return self.search(query, "science", max_results)

    def video_search(self, query: str, max_results: int = 5) -> str:
        """Use this function to search for videos.

        Args:
            query (str): The query to search videos with.
            max_results (int, optional): The maximum number of results to return. Defaults to 5.

        Returns:
            The results of the search.
        """
        return self.search(query, "videos", max_results)

    def search(
        self, query: str, category: Optional[str] = None, max_results: int = 5
    ) -> str:
        encoded_query = urllib.parse.quote(query)
        url = f"{self.host}/search?format=json&q={encoded_query}"

        if self.engines:
            url += f"&engines={','.join(self.engines)}"
        if category:
            url += f"&categories={category}"

        try:
            resp = httpx.get(url).json()
            results = self.fixed_max_results or max_results
            resp["results"] = resp["results"][:results]
            return json.dumps(resp)
        except Exception as e:
            return f"Error fetching results from searxng: {e}"