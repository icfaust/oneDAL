from playwright.sync_api import sync_playwright



with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html")

    # Check for "I'm Feeling Lucky" text
    if page.get_by_text("I'm Feeling Lucky").count() > 0:
        print("✅ Text found: I'm Feeling Lucky")
    else:
        print("❌ Text NOT found")

    browser.close()
