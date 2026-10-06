import sys
from playwright.sync_api import sync_playwright, expect

def run_tests():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        print("Testing app loads...")
        page.goto("http://localhost:8501")
        expect(page.locator("text=Evidence Atlas")).to_be_visible(timeout=60000)
        expect(page.locator("text=Find evidence, not just documents.")).to_be_visible(timeout=15000)
        
        print("Testing search interaction...")
        input_box = page.locator("input[aria-label='Ask a question about the corpus']")
        input_box.wait_for()
        input_box.fill("What attendance percentage is mandatory?")
        
        submit_button = page.locator("button:has-text('Ask the corpus')")
        submit_button.click()
        
        expect(page.locator("h1.question-headline")).to_contain_text("What attendance percentage is mandatory?", timeout=15000)
        print("Tests passed successfully.")
        browser.close()

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
