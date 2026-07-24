from __future__ import annotations

from hephaestus.ai.identity import SYSTEM_INSTRUCTION, SYSTEM_INSTRUCTION_VERSION


def test_system_instruction_version_is_stable_and_embedded() -> None:
    assert SYSTEM_INSTRUCTION_VERSION == "hephaestus.identity.v3"
    assert "Identity version: hephaestus.identity.v3" in SYSTEM_INSTRUCTION
    assert "hephaestus.identity.v1" not in SYSTEM_INSTRUCTION
    assert SYSTEM_INSTRUCTION.endswith("\n")


def test_system_instruction_contains_response_quality_contract() -> None:
    required_contract_fragments = (
        "answer simple questions briefly",
        "avoid unnecessary tables and boilerplate",
        "Do not repeat the user's prompt",
        "empty project summaries",
        "adding a requirement",
        "Do not expose internal tool names",
        "Do not mention nonexistent GUIs",
        "Ground claims strictly in tool results",
        "Acknowledge accepted or declined project mutations briefly",
    )

    for fragment in required_contract_fragments:
        assert fragment in SYSTEM_INSTRUCTION
