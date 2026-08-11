import json

def format_tender_list(index_file: str = "tenders_index.json", limit: int = 10) -> str:
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            tenders = json.load(f)

        tenders = sorted(tenders, key=lambda t: t.get("deadline") or "", reverse=False)
        total = len(tenders)
        shown = tenders[:limit]

        result = ""
        for t in shown:
            result += (
                f"### {t.get('title', 'Unknown')}\n"
                f"- **Stadt:** {t.get('city', '—')}\n"
                f"- **Frist:** {t.get('deadline', '—')}\n"
                f"- **Link:** [{t.get('link')}]({t.get('link')})\n"
                f"- **Bekanntmachung:** "
                f"{('[' + t.get('announcement_url') + '](' + t.get('announcement_url') + ')') if t.get('announcement_url') else '—'}\n\n"
            )

        if not result:
            return "Keine aktuellen Ausschreibungen gefunden."

        if total > len(shown):
            result += f"*Showing {len(shown)} of {total} tenders. Ask for more to see the rest.*\n"

        return result
    except FileNotFoundError:
        return "Tender index not found. Run fetch_latest_tenders first."
    except Exception as e:
        return f"Error loading tender list: {e}"