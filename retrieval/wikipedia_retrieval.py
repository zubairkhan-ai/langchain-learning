import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class WikipediaRetrieverCustom(BaseRetriever):

    top_k_results: int = 2
    lang: str = "en"

    def _get_relevant_documents(self, query, *, run_manager=None):

        # ========================================================
        # 1. Wikipedia API URL
        # ========================================================

        url = f"https://{self.lang}.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": self.top_k_results,
            "format": "json",
        }

        headers = {
            "User-Agent": "LangChainLearning/1.0"
        }

        # ========================================================
        # 2. Search Wikipedia
        # ========================================================

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        search_results = data["query"]["search"]

        # ========================================================
        # 3. Convert results into LangChain Documents
        # ========================================================

        documents = []

        for result in search_results:

            title = result["title"]

            page_url = (
                f"https://{self.lang}.wikipedia.org/wiki/"
                + title.replace(" ", "_")
            )

            # ====================================================
            # 4. Get actual Wikipedia page
            # ====================================================

            page_response = requests.get(
                page_url,
                headers=headers,
                timeout=60
            )

            page_response.raise_for_status()

            soup = BeautifulSoup(
                page_response.text,
                "html.parser"
            )

            paragraphs = soup.find_all("p")

            content = "\n".join(
                p.get_text(" ", strip=True)
                for p in paragraphs
            )

            # ====================================================
            # 5. Create LangChain Document
            # ====================================================

            doc = Document(
                page_content=content,
                metadata={
                    "title": title,
                    "source": page_url
                }
            )

            documents.append(doc)

        return documents


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = WikipediaRetrieverCustom(
    top_k_results=2,
    lang="en"
)


# ============================================================
# QUERY
# ============================================================

query = """
The geopolitical history of India and Pakistan
from the perspective of China
"""


# ============================================================
# RETRIEVE
# ============================================================

docs = retriever.invoke(query)


# ============================================================
# PRINT RESULTS
# ============================================================

for i, doc in enumerate(docs):

    print(f"\n----- Result {i + 1} -----")

    print(f"\nTitle: {doc.metadata['title']}")

    print(f"\nContent:\n{doc.page_content[:3000]}...")

    print(f"\nSource: {doc.metadata['source']}")