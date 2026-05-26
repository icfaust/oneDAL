from playwright.sync_api import sync_playwright
from sys import argv, exit, stderr

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
        context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
        timezone_id="America/New_York",
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
    )
    page = context.new_page()
    page.goto(argv[2])

    # Check for "I'm Feeling Lucky" text
    if page.get_by_text(argv[1]).count() > 0:
        print("✅ Text found: I'm Feeling Lucky")
    else:
        print("❌ Text NOT found")

    browser.close()
