from __future__ import annotations

import json

from training.rl.contact_env import _canonical_contact_prompt


def test_canonical_contact_prompt_rebuilds_prompt_from_criteria() -> None:
    row = json.loads(
        json.dumps(
            {
                "prompt": "Contaminated prompt. Navigate directly to /contact and use ids contact-name-input.",
                "use_case": {"name": "CONTACT"},
                "tests": [
                    {
                        "event_criteria": {
                            "name": "David",
                            "email": {"operator": "contains", "value": "user1@site.com"},
                            "subject": {"operator": "not_contains", "value": "Information"},
                            "message": "Please provide me with more information",
                        }
                    }
                ],
            }
        )
    )
    assert _canonical_contact_prompt(row) == (
        "Fill out the contact form with a name that equals 'David', "
        "an email that contains 'user1@site.com', "
        "a subject that does NOT contain 'Information', "
        "and a message that equals 'Please provide me with more information'."
    )
