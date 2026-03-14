from ddgs import DDGS

def web_search_tool(query=""):

    results = []

    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=5))

            for r in search_results:
                title = r.get("title","")
                snippet = r.get("body","")
                link = r.get("href","")

                results.append(
                    f"Title: {title}\nSummary: {snippet}\nLink: {link}"
                )

    except Exception as e:
        return f"Web search error: {str(e)}"

    if not results:
        return "Web search returned no results."

    return "External research findings:\n\n" + "\n\n".join(results)