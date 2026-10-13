from playwright.sync_api import sync_playwright

def scrape(params):
    url = params.get("url")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            content = page.content()
            browser.close()
            return f"HTML length: {len(content)}\nPreview: {content[:500]}"
    except Exception as e:
        return str(e)
