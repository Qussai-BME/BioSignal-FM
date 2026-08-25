"""Execution-based tests for the Streamlit dashboard.

The previously-existing "UI" tests (see test_v2_1_loaders_and_ui.py's
TestUIHonestyFixes) only grep the raw source text of UI files for expected
strings -- they never actually import or execute the app. That is precisely
why app.py's relative import (`from .theme import ...`, which crashes every
time Streamlit runs the file as a top-level script) went completely
undetected: no test in the whole suite ever ran the dashboard.

These tests use Streamlit's own `AppTest` harness (streamlit.testing.v1),
which actually executes a page/app the way `streamlit run` would, and lets
us assert `at.exception` is empty -- the same check used to diagnose and
verify the original fix. This is what should have existed from the start.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = REPO_ROOT / "biosignal_fm" / "ui"

PAGE_FILES = [
    "app.py",
    "pages/1_Overview.py",
    "pages/2_Pretrain.py",
    "pages/3_Finetune.py",
    "pages/4_Evaluate.py",
    "pages/5_Deploy.py",
]


@pytest.mark.parametrize("page_file", PAGE_FILES)
def test_page_runs_without_exception(page_file: str) -> None:
    """Every dashboard entry point must execute cleanly end-to-end.

    This is a regression test for the app.py relative-import crash
    (ImportError: attempted relative import with no known parent package)
    that broke the dashboard 100% of the time before the fix.
    """
    at = AppTest.from_file(str(UI_DIR / page_file))
    at.run(timeout=60)
    assert not at.exception, (
        f"{page_file} raised on run: {[str(e) for e in at.exception] if at.exception else None}"
    )


def test_main_app_renders_title() -> None:
    """Sanity check beyond 'no exception': the app actually renders content."""
    at = AppTest.from_file(str(UI_DIR / "app.py"))
    at.run(timeout=60)
    assert not at.exception
    assert any("BioSignal-FM" in t.value for t in at.title)


class TestLiveDemoButtons:
    """Regression tests for the live-demo buttons on each page.

    These pages used to show hardcoded/random placeholder numbers with no
    real computation behind them. They now run genuine (if small/capped)
    versions of the real pipelines. Clicking through each button here
    guards against that regressing back into decoration, and against the
    kind of import/type errors that only show up once a button is actually
    pressed (test_page_runs_without_exception above only covers the
    initial page load, not these deeper code paths).
    """

    def test_pretrain_live_demo_runs_real_training(self) -> None:
        at = AppTest.from_file(str(UI_DIR / "pages" / "2_Pretrain.py"))
        at.run(timeout=90)
        at.button[0].click()
        at.run(timeout=90)
        assert not at.exception
        assert any("real training steps" in s.value for s in at.success)

    def test_pretrain_config_generation(self) -> None:
        at = AppTest.from_file(str(UI_DIR / "pages" / "2_Pretrain.py"))
        at.run(timeout=90)
        at.button[1].click()
        at.run(timeout=90)
        assert not at.exception
        assert len(at.code) >= 1
        assert "d_model" in at.code[0].value

    def test_finetune_live_demo_runs_real_loso(self) -> None:
        at = AppTest.from_file(str(UI_DIR / "pages" / "3_Finetune.py"))
        at.run(timeout=90)
        at.button[0].click()
        at.run(timeout=90)
        assert not at.exception
        # 2 dataframes: per-fold results table + confusion matrix
        assert len(at.dataframe) == 2
        assert any(m.label == "Mean Accuracy" for m in at.metric)

    def test_evaluate_live_stats_demo(self) -> None:
        at = AppTest.from_file(str(UI_DIR / "pages" / "4_Evaluate.py"))
        at.run(timeout=90)
        at.button[0].click()
        at.run(timeout=90)
        assert not at.exception
        assert any("Critical difference" in m.value for m in at.markdown)
        assert any("Hedges" in m.value for m in at.markdown)

    def test_deploy_live_benchmark(self) -> None:
        at = AppTest.from_file(str(UI_DIR / "pages" / "5_Deploy.py"))
        at.run(timeout=90)
        at.button[0].click()
        at.run(timeout=90)
        assert not at.exception
        assert any("numerical parity" in m.value for m in at.markdown)
        # Reports the real outcome either way (speedup, slowdown, or
        # fallback) rather than assuming quantization always wins.
        assert (
            any("faster" in s.value for s in at.success)
            or any("slower" in i.value for i in at.info)
            or any("fell back" in w.value for w in at.warning)
        )

    def test_deploy_live_rest_playground(self) -> None:
        at = AppTest.from_file(str(UI_DIR / "pages" / "5_Deploy.py"))
        at.run(timeout=90)
        at.button[1].click()
        at.run(timeout=90)
        assert not at.exception
        # 3 real HTTP round-trips through the actual FastAPI app: register,
        # predict, list models — each rendered as its own code block.
        codes = [c.value for c in at.code if "HTTP 200" in c.value]
        assert len(codes) == 3
        assert any("model_id" in c and "sha256" in c for c in codes)
        assert any("cls_token" in c for c in codes)


class TestStageIndicatorConsistency:
    """Regression test for the pipeline wayfinding indicator (theme.py's
    stage_indicator()) on every dashboard page.

    Note for anyone extending this: don't match on a generic substring like
    "bsfm-stage-track" to find the right markdown block — inject_css()'s
    injected <style> block literally contains ".bsfm-stage-track { ... }"
    as a CSS selector, runs before the indicator is rendered, and will
    false-positive-match first. Match the actual opening tag instead.
    """

    PAGES = [
        ("Overview", "1_Overview.py"),
        ("Pretrain", "2_Pretrain.py"),
        ("Finetune", "3_Finetune.py"),
        ("Evaluate", "4_Evaluate.py"),
        ("Deploy", "5_Deploy.py"),
    ]

    @pytest.mark.parametrize("name,filename", PAGES)
    def test_exactly_current_page_marked_active(self, name: str, filename: str) -> None:
        at = AppTest.from_file(str(UI_DIR / "pages" / filename))
        at.run(timeout=60)
        assert not at.exception

        stage_html = next(
            (m.value for m in at.markdown if m.value.startswith('<div class="bsfm-stage-track">')),
            None,
        )
        assert stage_html is not None, f"stage indicator not rendered on {filename}"
        assert stage_html.count('class="bsfm-stage active"') == 1
        active_pos = stage_html.find("active")
        assert f">{name}<" in stage_html[active_pos : active_pos + 60]
