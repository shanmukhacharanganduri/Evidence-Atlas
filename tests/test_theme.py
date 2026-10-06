import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def css_tokens():
    css = (ROOT / "theme.css").read_text("utf-8")
    found = re.findall(r"--(\w+):\s*light-dark\((#[0-9A-Fa-f]{6}),\s*(#[0-9A-Fa-f]{6})\)", css)
    assert found, "no light-dark() tokens in theme.css"
    return {name: {"light": light, "dark": dark} for name, light, dark in found}


def luminance(hex_colour):
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_css_tokens_match_streamlit_config_palette():
    theme = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text("utf-8"))["theme"]
    tokens = css_tokens()
    for mode in ("light", "dark"):
        assert theme[mode]["backgroundColor"].lower() == tokens["paper"][mode].lower()
        assert theme[mode]["textColor"].lower() == tokens["ink"][mode].lower()
        assert theme[mode]["primaryColor"].lower() == tokens["accent"][mode].lower()


def test_text_tokens_meet_wcag_aa_in_both_modes():
    tokens = css_tokens()
    for mode in ("light", "dark"):
        for name in ("ink", "muted", "accent", "warning", "danger", "focus"):
            assert contrast(tokens[name][mode], tokens["paper"][mode]) >= 4.5, (mode, name)
        assert contrast(tokens["ink"][mode], tokens["notice"][mode]) >= 4.5, (mode, "notice")


def test_no_forced_base_theme_or_generated_class_overrides():
    config = (ROOT / ".streamlit" / "config.toml").read_text("utf-8")
    assert 'base = "dark"' not in config and 'base = "light"' not in config
    assert '[class*="css"]' not in (ROOT / "theme.css").read_text("utf-8")
