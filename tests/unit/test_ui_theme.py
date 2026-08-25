"""Tests for biosignal_fm.ui.theme — the design system.

The previous theme.py claimed WCAG AA compliance in a docstring comment
("verified manually via the WebAIM contrast checker") with nothing in the
codebase actually checking it — exactly the kind of claim this project's
own audit found to be unreliable elsewhere. This computes real contrast
ratios (the standard WCAG relative-luminance formula) so the claim is
enforced, not just asserted.
"""

from __future__ import annotations

from biosignal_fm.ui.theme import COLORS, MODALITIES, modality_badge, stage_indicator


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    def channel(c: int) -> float:
        c_norm = c / 255.0
        return c_norm / 12.92 if c_norm <= 0.03928 else ((c_norm + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_ratio(hex1: str, hex2: str) -> float:
    """WCAG 2.x contrast ratio between two hex colors, in [1, 21].

    This is a *readability* metric (would text of one color be legible on
    a background of the other), not a perceptual color-difference metric —
    two different hues at matched luminance can have a low ratio here while
    still looking completely different to the eye. Use delta_e76 (below)
    for "are these two colors distinguishable from each other".
    """
    l1 = _relative_luminance(_hex_to_rgb(hex1))
    l2 = _relative_luminance(_hex_to_rgb(hex2))
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _srgb_to_linear(c: int) -> float:
    c_norm = c / 255.0
    return c_norm / 12.92 if c_norm <= 0.04045 else ((c_norm + 0.055) / 1.055) ** 2.4


def _rgb_to_xyz(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(c) for c in rgb)
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    return x, y, z


def _xyz_to_lab(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[0] / xn, xyz[1] / yn, xyz[2] / zn

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    ell = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return ell, a, b


def delta_e76(hex1: str, hex2: str) -> float:
    """CIE76 perceptual color difference. ~2.3 = just noticeable, ~10 =
    clearly different, ~20+ = very different. This is the right metric for
    "can a person tell these two colors apart", unlike contrast_ratio above.
    """
    lab1 = _xyz_to_lab(_rgb_to_xyz(_hex_to_rgb(hex1)))
    lab2 = _xyz_to_lab(_rgb_to_xyz(_hex_to_rgb(hex2)))
    distance: float = sum((a - b) ** 2 for a, b in zip(lab1, lab2, strict=True)) ** 0.5
    return distance


class TestWCAGCompliance:
    """Every foreground/background pairing actually used in the UI."""

    AA_NORMAL_TEXT = 4.5
    # COLORS mixes str values (hex colors) and list[str] (the "chart" series),
    # so mypy can't narrow COLORS[key] to str for an arbitrary key. This
    # sub-dict is scoped to the actual str-valued keys these tests use.
    _hex: dict[str, str] = {k: v for k, v in COLORS.items() if isinstance(v, str)}

    def test_text_colors_meet_aa_on_backgrounds(self) -> None:
        for bg_name in ("bg", "surface"):
            bg = self._hex[bg_name]
            for fg_name in ("text", "text_muted"):
                fg = self._hex[fg_name]
                ratio = contrast_ratio(fg, bg)
                assert ratio >= self.AA_NORMAL_TEXT, (
                    f"{fg_name} ({fg}) on {bg_name} ({bg}) = {ratio:.2f}:1, "
                    f"below AA's {self.AA_NORMAL_TEXT}:1"
                )

    def test_primary_accent_meets_aa_on_backgrounds(self) -> None:
        for bg_name in ("bg", "surface"):
            bg = self._hex[bg_name]
            ratio = contrast_ratio(self._hex["primary"], bg)
            assert ratio >= self.AA_NORMAL_TEXT, f"primary on {bg_name} = {ratio:.2f}:1, below AA"

    def test_status_colors_meet_aa_on_backgrounds(self) -> None:
        for bg_name in ("bg", "surface"):
            bg = self._hex[bg_name]
            for status in ("success", "warning", "danger"):
                ratio = contrast_ratio(self._hex[status], bg)
                assert ratio >= self.AA_NORMAL_TEXT, (
                    f"{status} on {bg_name} = {ratio:.2f}:1, below AA"
                )

    def test_modality_colors_meet_aa_on_backgrounds(self) -> None:
        """The per-modality wayfinding colors are used as text/badge
        foregrounds throughout the dashboard, so they need the same bar as
        body text, not just decorative-use-exempt large text.
        """
        for bg_name in ("bg", "surface"):
            bg = self._hex[bg_name]
            for key, spec in MODALITIES.items():
                ratio = contrast_ratio(spec["color"], bg)
                assert ratio >= self.AA_NORMAL_TEXT, (
                    f"modality '{key}' ({spec['color']}) on {bg_name} = {ratio:.2f}:1, below AA"
                )

    def test_modality_colors_are_perceptually_distinguishable(self) -> None:
        """The four modality colors must be pairwise distinct enough that
        color alone is a usable (if not sole) wayfinding signal. Uses
        CIE76 Delta-E (perceptual distance), not WCAG contrast ratio —
        contrast ratio measures readability, not "can you tell these
        apart", and gives false positives for e.g. orange vs blue at
        matched luminance.
        """
        colors = [spec["color"] for spec in MODALITIES.values()]
        for i, c1 in enumerate(colors):
            for c2 in colors[i + 1 :]:
                de = delta_e76(c1, c2)
                assert de >= 15.0, f"{c1} and {c2} are too visually similar (dE={de:.1f})"

    def test_text_inverse_meets_aa_on_primary(self) -> None:
        """White stage-indicator numerals on the primary-colored active dot."""
        ratio = contrast_ratio(self._hex["text_inverse"], self._hex["primary"])
        assert ratio >= self.AA_NORMAL_TEXT


class TestModalityBadge:
    def test_all_four_modalities_render(self) -> None:
        for key in ("emg", "ecg", "eeg", "fnirs"):
            html = modality_badge(key)
            assert MODALITIES[key]["label"] in html
            assert MODALITIES[key]["color"] in html

    def test_case_insensitive(self) -> None:
        assert modality_badge("EMG") == modality_badge("emg")

    def test_unknown_modality_raises(self) -> None:
        import pytest

        with pytest.raises(KeyError):
            modality_badge("not-a-modality")


class TestStageIndicator:
    def test_all_five_stages_present(self) -> None:
        html = stage_indicator("Evaluate")
        for stage in ["Overview", "Pretrain", "Finetune", "Evaluate", "Deploy"]:
            assert stage in html

    def test_current_stage_marked_active(self) -> None:
        html = stage_indicator("Finetune")
        assert 'class="bsfm-stage active"' in html
        # Exactly one stage should be marked active
        assert html.count('class="bsfm-stage active"') == 1
