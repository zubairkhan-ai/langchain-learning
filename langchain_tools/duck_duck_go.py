from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool


# ============================================================
# 1. DUCKDUCKGO SEARCH TOOL
# ============================================================

search_tool = DuckDuckGoSearchRun()

query = "poverty in Pakistan"

results = search_tool.invoke(query)

print("\n================ DUCKDUCKGO RESULTS ================\n")
print(results)


# ============================================================
# 2. SHELL TOOL
# ============================================================

shell_tool = ShellTool()

command = "dir"

results = shell_tool.invoke(command)

print("\n================ SHELL RESULTS ================\n")
print(results)