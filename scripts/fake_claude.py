#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def main() -> None:
    _ = sys.stdin.read()
    print(
        json.dumps(
            {
                "type": "result",
                "structured_output": {
                    "success": True,
                    "summary": "fake claude produced a replayable trajectory",
                    "trajectory": [
                        {
                            "name": "browser.navigate",
                            "arguments": {"url": "/"},
                        }
                    ],
                },
            }
        )
    )


if __name__ == "__main__":
    main()
