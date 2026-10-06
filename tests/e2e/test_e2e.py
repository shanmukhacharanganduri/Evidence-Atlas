"""Opt-in browser tests: `python -m pytest -m e2e` (needs playwright browsers and cached models)."""
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.e2e


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = free_port()
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port),
                                "--server.headless", "true", "--server.fileWatcherType", "none"], cwd=ROOT)
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.5)
    yield f"http://localhost:{port}"
    process.terminate()
    process.wait(timeout=15)


def test_answer_contains_the_value_and_offers_the_cited_page(server):
    from playwright.sync_api import expect, sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(server)
        page.get_by_label("Institution scope").click()
        page.get_by_role("option", name="KMEC (synthetic demo)").click()
        page.get_by_label("Ask a question about the corpus").fill("What attendance percentage is mandatory?")
        page.get_by_role("button", name="Ask the corpus").click()
        expect(page.locator(".answer-copy")).to_contain_text("75%", timeout=180_000)  # the wrong sentence would fail here
        page.get_by_role("button", name="Open cited page", exact=False).first.click()
        expect(page.get_by_role("dialog")).to_be_visible()
        expect(page.get_by_role("dialog")).to_contain_text("Academic_Regulations_2024.pdf")
        browser.close()


def test_unknown_entity_is_not_answered(server):
    from playwright.sync_api import expect, sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(server)
        page.get_by_label("Ask a question about the corpus").fill("Which department owns the Novalux asteroid laboratory?")
        page.get_by_role("button", name="Ask the corpus").click()
        expect(page.get_by_text("Unable to verify")).to_be_visible(timeout=180_000)
        browser.close()
