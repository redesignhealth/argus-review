"""Unit tests for the image-tag-bump blast-radius exemption.

Covers _is_image_tag_bump_only() and the _edge_preflight_decision gate:

  (a) pure image-tag bump diff in a .tfvars file        → True
  (b) mixed diff (one non-bump line)                    → False
  (c) empty string                                      → False
  (d) header-only diff (no +/- content lines)           → False
  (e) bump in a non-.tfvars file (.tf module)           → False
  (f) multiple .tfvars files, all bumps                 → True
  (g) _edge_preflight_decision: blast-radius + pure bump → does NOT return "plan"
  (h) _edge_preflight_decision: blast-radius + mixed    → returns "plan"
"""

from __future__ import annotations

from argus.graph import _is_image_tag_bump_only, _edge_preflight_decision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bump_diff(
    filename: str = "infrastructure/environments/prod/terraform.tfvars",
    old_sha: str = "abc1234",
    new_sha: str = "def5678",
    a_filename: str | None = None,
    b_filename: str | None = None,
) -> str:
    a = a_filename if a_filename is not None else filename
    b = b_filename if b_filename is not None else filename
    return (
        f"diff --git a/{a} b/{b}\n"
        "--- a/terraform.tfvars\n"
        "+++ b/terraform.tfvars\n"
        "@@ -1 +1 @@\n"
        f'-my_service_image_tag = "sha-{old_sha}"\n'
        f'+my_service_image_tag = "sha-{new_sha}"\n'
    )


# ---------------------------------------------------------------------------
# _is_image_tag_bump_only
# ---------------------------------------------------------------------------


def test_pure_tfvars_bump_returns_true() -> None:
    assert _is_image_tag_bump_only(_bump_diff()) is True


def test_mixed_diff_returns_false() -> None:
    diff = _bump_diff() + "-some_other_var = true\n+some_other_var = false\n"
    assert _is_image_tag_bump_only(diff) is False


def test_empty_string_returns_false() -> None:
    assert _is_image_tag_bump_only("") is False


def test_header_only_diff_returns_false() -> None:
    diff = (
        "diff --git a/infrastructure/environments/prod/terraform.tfvars "
        "b/infrastructure/environments/prod/terraform.tfvars\n"
        "--- a/terraform.tfvars\n"
        "+++ b/terraform.tfvars\n"
    )
    assert _is_image_tag_bump_only(diff) is False


def test_bump_in_tf_module_returns_false() -> None:
    assert (
        _is_image_tag_bump_only(_bump_diff(filename="infrastructure/modules/ecs/main.tf")) is False
    )


def test_multiple_tfvars_all_bumps_returns_true() -> None:
    diff = _bump_diff("infrastructure/environments/prod/terraform.tfvars") + _bump_diff(
        "infrastructure/environments/dev/terraform.tfvars", "aaa1111", "bbb2222"
    )
    assert _is_image_tag_bump_only(diff) is True


def test_multiple_files_one_non_tfvars_returns_false() -> None:
    diff = _bump_diff("infrastructure/environments/prod/terraform.tfvars") + _bump_diff(
        "infrastructure/modules/ecs/main.tf"
    )
    assert _is_image_tag_bump_only(diff) is False


def test_variable_name_leading_char_constraint() -> None:
    # Uppercase prefix — rejected by [a-z] leading char constraint
    diff_upper = _bump_diff().replace("my_service_image_tag", "My_service_image_tag")
    assert _is_image_tag_bump_only(diff_upper) is False
    # Digit prefix — rejected
    diff_digit = _bump_diff().replace("my_service_image_tag", "1service_image_tag")
    assert _is_image_tag_bump_only(diff_digit) is False
    # Underscore prefix — rejected
    diff_underscore = _bump_diff().replace("my_service_image_tag", "_service_image_tag")
    assert _is_image_tag_bump_only(diff_underscore) is False


def test_asymmetric_a_b_extensions_returns_false() -> None:
    # .tf source renamed to .tfvars destination — both sides must be .tfvars
    diff = _bump_diff(
        a_filename="infrastructure/modules/ecs/main.tf",
        b_filename="infrastructure/environments/prod/terraform.tfvars",
    )
    assert _is_image_tag_bump_only(diff) is False


def test_addition_only_returns_false() -> None:
    # Only + lines (no - lines): new service onboarding, not a bump — must route to full review.
    diff = (
        "diff --git a/infrastructure/environments/prod/terraform.tfvars "
        "b/infrastructure/environments/prod/terraform.tfvars\n"
        "--- a/terraform.tfvars\n"
        "+++ b/terraform.tfvars\n"
        "@@ -0,0 +1 @@\n"
        '+new_service_image_tag = "sha-abc1234"\n'
    )
    assert _is_image_tag_bump_only(diff) is False


def test_sha_length_boundaries() -> None:
    # 7-char SHA (short) — valid
    assert _is_image_tag_bump_only(_bump_diff(old_sha="abc1234", new_sha="def5678")) is True
    # 40-char SHA (full) — valid
    assert _is_image_tag_bump_only(_bump_diff(old_sha="a" * 40, new_sha="b" * 40)) is True
    # 6-char SHA — too short, invalid
    assert _is_image_tag_bump_only(_bump_diff(old_sha="abc123", new_sha="def456")) is False
    # 41-char SHA — too long, invalid
    assert _is_image_tag_bump_only(_bump_diff(old_sha="a" * 41, new_sha="b" * 41)) is False


# ---------------------------------------------------------------------------
# _edge_preflight_decision integration
# ---------------------------------------------------------------------------


def _make_state(diff: str, is_lite_preflight: bool = True) -> dict:
    return {
        "diff": diff,
        "is_lite": is_lite_preflight,
        "preflight_result": {"route": "lite" if is_lite_preflight else "full", "reason": "test"},
        "verification": {},
    }


def test_blast_radius_with_pure_bump_does_not_force_full() -> None:
    state = _make_state(_bump_diff())
    result = _edge_preflight_decision(state)
    # Should honor the LLM lite decision, not override to "plan"
    assert result == "lite_review"


def test_blast_radius_with_mixed_diff_forces_full() -> None:
    mixed = _bump_diff() + "-some_other_var = true\n+some_other_var = false\n"
    state = _make_state(mixed)
    result = _edge_preflight_decision(state)
    assert result == "plan"


def test_auto_tfvars_json_not_exempt() -> None:
    diff = (
        "diff --git a/infrastructure/environments/prod/external-apps.auto.tfvars.json "
        "b/infrastructure/environments/prod/external-apps.auto.tfvars.json\n"
        "--- a/external-apps.auto.tfvars.json\n"
        "+++ b/external-apps.auto.tfvars.json\n"
        "@@ -1 +1 @@\n"
        '-  "other_service_image_tag": "sha-abc1234"\n'
        '+  "other_service_image_tag": "sha-def5678"\n'
    )
    assert _is_image_tag_bump_only(diff) is False


def test_deletion_only_returns_false() -> None:
    # Only - lines (no + lines): variable being removed, not bumped — must route to full review.
    diff = (
        "diff --git a/infrastructure/environments/prod/terraform.tfvars "
        "b/infrastructure/environments/prod/terraform.tfvars\n"
        "--- a/terraform.tfvars\n"
        "+++ b/terraform.tfvars\n"
        "@@ -1 +0,0 @@\n"
        '-my_service_image_tag = "sha-abc1234"\n'
    )
    assert _is_image_tag_bump_only(diff) is False


def test_blast_radius_with_pure_bump_and_is_lite_false_routes_to_plan() -> None:
    # When preflight already routed to full (is_lite=False), the bump exemption
    # does not override it — result must be "plan".
    state = _make_state(_bump_diff(), is_lite_preflight=False)
    result = _edge_preflight_decision(state)
    assert result == "plan"


def test_context_line_not_counted_as_content() -> None:
    # Context lines (space-prefixed) must be ignored; only +/- lines matter.
    diff = (
        "diff --git a/infrastructure/environments/prod/terraform.tfvars "
        "b/infrastructure/environments/prod/terraform.tfvars\n"
        "--- a/terraform.tfvars\n"
        "+++ b/terraform.tfvars\n"
        "@@ -1,2 +1,2 @@\n"
        ' my_service_image_tag = "sha-abc1234"\n'
        '-my_service_image_tag = "sha-abc1234"\n'
        '+my_service_image_tag = "sha-def5678"\n'
    )
    assert _is_image_tag_bump_only(diff) is True


# ---------------------------------------------------------------------------
# Catch-up merge exemption in _edge_preflight_decision
# ---------------------------------------------------------------------------

# A diff that touches a high-blast-radius infra file but is NOT an image-tag bump.
# This triggers _is_high_blast_radius but not _is_image_tag_bump_only.
_INFRA_NON_BUMP_DIFF = (
    "diff --git a/infrastructure/environments/prod/terraform.tfvars "
    "b/infrastructure/environments/prod/terraform.tfvars\n"
    "--- a/terraform.tfvars\n"
    "+++ b/terraform.tfvars\n"
    "@@ -1 +1 @@\n"
    '-some_infra_setting = "old-value"\n'
    '+some_infra_setting = "new-value"\n'
)


def test_catchup_merge_exemption_fires_with_prior_review() -> None:
    # Exemption case: high-blast-radius diff + is_catchup_merge=True + prior_review present.
    # The blast-radius gate must be bypassed; result defers to is_lite.
    state = {
        "diff": _INFRA_NON_BUMP_DIFF,
        "is_catchup_merge": True,
        "prior_review": {"round": 1, "verdict": "APPROVE"},
        "is_lite": True,
        "preflight_result": {"route": "lite", "reason": "test"},
        "verification": {},
    }
    result = _edge_preflight_decision(state)
    assert result != "plan", "catch-up merge exemption should bypass the blast-radius hard gate"
    assert result == "lite_review"


def test_catchup_merge_exemption_blocked_without_prior_review() -> None:
    # Round-2+ guard: is_catchup_merge=True but no prior_review → exemption must NOT fire.
    # A round-1 PR whose HEAD happens to be a merge commit must still go through full review.
    state = {
        "diff": _INFRA_NON_BUMP_DIFF,
        "is_catchup_merge": True,
        "prior_review": {},
        "is_lite": True,
        "preflight_result": {"route": "lite", "reason": "test"},
        "verification": {},
    }
    result = _edge_preflight_decision(state)
    assert result == "plan"


def test_non_catchup_high_blast_radius_still_forces_full() -> None:
    # is_catchup_merge=False: the exemption is off; blast-radius gate fires unconditionally.
    state = {
        "diff": _INFRA_NON_BUMP_DIFF,
        "is_catchup_merge": False,
        "prior_review": {"round": 1, "verdict": "APPROVE"},
        "is_lite": True,
        "preflight_result": {"route": "lite", "reason": "test"},
        "verification": {},
    }
    result = _edge_preflight_decision(state)
    assert result == "plan"
