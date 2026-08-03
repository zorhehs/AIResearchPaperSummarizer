"""
Fake Gemini responses for testing pipeline logic without burning API quota.
Swap real calls for these during development, switch back for final verification.
"""

def fake_ask_gemini(prompt: str) -> str:
    return f"[MOCK RESPONSE] This would be a real Gemini summary for a prompt of length {len(prompt)} chars."
