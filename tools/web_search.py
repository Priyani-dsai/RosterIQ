from ddgs import DDGS


def web_search_tool(query=""):

    # Ensure query is valid
    if not isinstance(query, str) or query.strip() == "":
        query = "ETL pipeline debugging techniques"

    results = []

    try:
        with DDGS() as ddgs:

            search_results = list(ddgs.text(query, max_results=5))

            for r in search_results:

                title = r.get("title", "No title available")
                snippet = r.get("body", "No summary available")
                link = r.get("href", "")

                formatted = (
                    f"Title: {title}\n"
                    f"Summary: {snippet}\n"
                    f"Link: {link}"
                )

                results.append(formatted)

    except Exception as e:
        return f"Web search error: {str(e)}"

    if not results:
        return "No external information was found for this query."

    return (
        "External research findings related to the query:\n\n"
        + "\n\n".join(results)
    )