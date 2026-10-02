"""Format a results table as GitHub-flavoured Markdown for the README."""


def markdown_table(rows, columns):
    """Turn a list of dict rows into a Markdown table with the given column order."""
    if not rows:
        raise ValueError("no rows to format")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join([header, divider] + body)
