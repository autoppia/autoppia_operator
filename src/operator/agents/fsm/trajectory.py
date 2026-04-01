from __future__ import annotations

import re
from typing import Any, Dict, List

TRAJECTORIES: List[Dict[str, Any]] = [
    {
        "project_id": "p01_autocinema",
        "trajectories": [
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Add a comment to the movie_name that is NOT 'The Godfather' with a content that is NOT 'couldn't look away'.",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "featured-movie-view-details-btn-2",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "featured-movie-view-details-btn-2",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "comment-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "comment-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Agent",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "comment-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Agent",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "comment-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "comment-message-textarea",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "comment-message-textarea",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "good movie",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "comment-message-textarea",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "good movie",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "comment-message-textarea",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "share-feedback-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "share-feedback-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "ADD_COMMENT",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Login with username equals 'user<web_agent_id>' and password equals 'Passw0rd!'. Insert a new film with genres equals 'Thriller'.",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[6]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[6]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "user<web_agent_id>",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "user<web_agent_id>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Passw0rd!",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Passw0rd!",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-sign-in-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-sign-in-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//button[normalize-space()='Add Movies']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//button[normalize-space()='Add Movies']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//button[normalize-space()='Adventure']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//button[normalize-space()='Adventure']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "save-changes-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "save-changes-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "ADD_FILM",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Login with the username equals 'user<web_agent_id>' and password equals 'Passw0rd!' and then add to watchlist a film with rating less equal 5.0 and duration less than 124 minutes long",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[6]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[6]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id=\"login-username-input\"]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id=\"login-username-input\"]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "user<web_agent_id>",
                        "type": "TypeAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id=\"login-username-input\"]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "user<web_agent_id>",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id=\"login-username-input\"]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id=\"login-password-input\"]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id=\"login-password-input\"]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Passw0rd!",
                        "type": "TypeAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id=\"login-password-input\"]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Passw0rd!",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id=\"login-password-input\"]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-sign-in-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-sign-in-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "featured-movie-view-details-btn",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "featured-movie-view-details-btn",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "watchlist-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "watchlist-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "ADD_TO_WATCHLIST",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/contact?seed=1",
                "prompt": "Fill out the contact form with a name NOT 'Lisa', an email that contains 'in@d', and a subject that does NOT contain 'mwg'.",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[4]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[4]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Javier Test",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Javier Test",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-email-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-email-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "javier.test@example.com",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-email-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "javier.test@example.com",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-email-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-subject-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-subject-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Hello from trajectory",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-subject-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Hello from trajectory",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-subject-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-message-textarea",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-message-textarea",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Please help with movie recommendations.",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-message-textarea",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Please help with movie recommendations.",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-message-textarea",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "send-message-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "send-message-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "CONTACT",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Login with username equals 'user<web_agent_id>' and password equals 'Passw0rd!'. Then, delete your movie.",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[6]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[6]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "user<web_agent_id>",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "user<web_agent_id>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Passw0rd!",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Passw0rd!",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-sign-in-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-sign-in-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//button[normalize-space()='Edit Movies']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//button[normalize-space()='Edit Movies']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "delete-movie-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "delete-movie-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "DELETE_FILM",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Login with username equals 'user<web_agent_id>' and password equals 'Passw0rd!'. Edit your movie by setting year to '1966'.",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[6]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[6]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "user<web_agent_id>",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "user<web_agent_id>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Passw0rd!",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Passw0rd!",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-sign-in-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-sign-in-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//button[normalize-space()='Edit Movies']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//button[normalize-space()='Edit Movies']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//label[contains(normalize-space(), 'Year')]/input)[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//label[contains(normalize-space(), 'Year')]/input)[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "1966",
                        "type": "TypeAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//label[contains(normalize-space(), 'Year')]/input)[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "1966",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//label[contains(normalize-space(), 'Year')]/input)[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "save-changes-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "save-changes-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "EDIT_FILM",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Login with username equals 'user<web_agent_id>' and password equals 'Passw0rd!'. Edit your profile: ensure your first_name contains 'mes', your website does NOT contain 'nhl', and your location does NOT contain 'evc'.",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[6]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[6]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "user<web_agent_id>",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "user<web_agent_id>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Passw0rd!",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Passw0rd!",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-sign-in-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-sign-in-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "profile-last-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "profile-last-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Alexander",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "profile-last-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Alexander",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "profile-last-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "profile-bio-textarea",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "profile-bio-textarea",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "cinema lover",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "profile-bio-textarea",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "cinema lover",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "profile-bio-textarea",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "profile-website-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "profile-website-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "https://example.org",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "profile-website-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "https://example.org",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "profile-website-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "save-profile-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "save-profile-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "EDIT_USER",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Take me directly to the interstellar film details page",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "spotlight-view-details-btn-2",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "FILM_DETAIL",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Filter for Action movies",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[2]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[2]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//button[normalize-space()='Action']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//button[normalize-space()='Action']",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "FILTER_FILM",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Please login using username equals 'user<web_agent_id>' and password equals 'Passw0rd!'.",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[6]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[6]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "javier",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "javier",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "123456",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "123456",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-sign-in-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-sign-in-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "LOGIN",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Please login using username equals 'user<web_agent_id>' and password equals 'Passw0rd!' and then logout.",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[6]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[6]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "AGENTE",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "AGENTE",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Passw0rd!",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Passw0rd!",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-sign-in-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-sign-in-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/button",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/button",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "LOGOUT",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Please register using username equals 'newuser<web_agent_id>', email equals 'newuser<web_agent_id>@gmail.com' and password equals 'Passw0rd!'",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[5]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[5]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "register-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "register-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "javier",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "register-username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "javier",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "register-username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "register-email-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "register-email-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "javi@gmail.com",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "register-email-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "javi@gmail.com",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "register-email-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "register-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "register-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "123456",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "register-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "123456",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "register-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "register-confirm-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "register-confirm-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "123456",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "register-confirm-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "123456",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "register-confirm-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "create-account-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "create-account-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "REGISTRATION",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Login with username equals 'user<web_agent_id>' and password equals 'Passw0rd!' and remove a movie from watchlist",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[6]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[6]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id=\"login-username-input\"]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id=\"login-username-input\"]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "user<web_agent_id>",
                        "type": "TypeAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id=\"login-username-input\"]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "user<web_agent_id>",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id=\"login-username-input\"]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id=\"login-password-input\"]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id=\"login-password-input\"]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "Passw0rd!",
                        "type": "TypeAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id=\"login-password-input\"]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Passw0rd!",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id=\"login-password-input\"]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-sign-in-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-sign-in-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//html/body/header/div/nav/a[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//html/body/header/div/nav/a[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "featured-movie-view-details-btn",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "featured-movie-view-details-btn",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "watchlist-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "watchlist-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "watchlist-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "watchlist-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "REMOVE_FROM_WATCHLIST",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Search for the movie 'La La Land'",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "La La Land",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "La La Land",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "SendKeysAction",
                        "keys": ["Enter"],
                        "attributes": {
                            "keys": ["Enter"]
                        },
                    },
                ],
                "use_case": "SEARCH_FILM",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Share a movie directed by one of 'Ethan Coen', 'Lana Wachowski', 'Fernando Meirelles' that is NOT named 'Schindler's List'",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "text": "ethan Coen",
                        "type": "TypeAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "ethan Coen",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "search-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "view-details-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "view-details-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "share-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "share-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "SHARE_MOVIE",
                "has_success": True,
            },
            {
                "url": "http://localhost:8000/?seed=1",
                "prompt": "Watch the trailer for a movie with a duration NOT EQUALS '118' minutes that has a rating GREATER EQUAL '5.0'",
                "actions": [
                    {
                        "url": "http://localhost:8000/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8000/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "featured-movie-view-details-btn",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "featured-movie-view-details-btn",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "watch-trailer-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "watch-trailer-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id=\"movie_player\"]/div[1]/video",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id=\"movie_player\"]/div[1]/video",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "WATCH_TRAILER",
                "has_success": True,
            },
        ],
    },
    {
        "project_id": "p02_autobooks",
        "trajectories": [
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Register with username, email and password placeholders.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Register']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Register']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SIGNUP_USERNAME__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__SIGNUP_USERNAME__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "signup-email-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "signup-email-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SIGNUP_EMAIL__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "signup-email-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__SIGNUP_EMAIL__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "signup-email-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SIGNUP_PASSWORD__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__SIGNUP_PASSWORD__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "confirm-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "confirm-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SIGNUP_PASSWORD__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "confirm-password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__SIGNUP_PASSWORD__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "confirm-password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "signup-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "signup-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "REGISTRATION_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Look for the book 'Lolita'",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/search')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/search')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-field' or @id='search-input']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__SEARCH_QUERY__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='search-field' or @id='search-input']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "SEARCH_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Filter books released in the year 2021",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/search')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/search')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "__year_select__",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "__year_select__",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "SendKeysAction",
                        "keys": [
                            "__YEAR_KEY_1__",
                        ],
                        "attributes": {
                            "keys": [
                                "__YEAR_KEY_1__",
                            ],
                        },
                    },
                    {
                        "type": "SendKeysAction",
                        "keys": [
                            "__YEAR_KEY_2__",
                        ],
                        "attributes": {
                            "keys": [
                                "__YEAR_KEY_2__",
                            ],
                        },
                    },
                    {
                        "type": "SendKeysAction",
                        "keys": [
                            "__YEAR_KEY_3__",
                        ],
                        "attributes": {
                            "keys": [
                                "__YEAR_KEY_3__",
                            ],
                        },
                    },
                    {
                        "type": "SendKeysAction",
                        "keys": [
                            "__YEAR_KEY_4__",
                        ],
                        "attributes": {
                            "keys": [
                                "__YEAR_KEY_4__",
                            ],
                        },
                    },
                    {
                        "type": "SendKeysAction",
                        "keys": [
                            "Enter",
                        ],
                        "attributes": {
                            "keys": [
                                "Enter",
                            ],
                        },
                    },
                ],
                "use_case": "FILTER_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Fill and submit the contact form.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Contact']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Contact']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__CONTACT_NAME__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__CONTACT_NAME__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-email-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-email-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__CONTACT_EMAIL__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-email-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__CONTACT_EMAIL__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-email-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-subject-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-subject-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__CONTACT_SUBJECT__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-subject-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__CONTACT_SUBJECT__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-subject-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-message-textarea",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-message-textarea",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__CONTACT_MESSAGE__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "contact-message-textarea",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__CONTACT_MESSAGE__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "contact-message-textarea",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "send-message-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "send-message-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "CONTACT_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login with username and password.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "LOGIN_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login and logout.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//button[normalize-space()='Logout']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//button[normalize-space()='Logout']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "LOGOUT_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login and delete one assigned book.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/profile')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/profile')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "profile-tab-books",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "profile-tab-books",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[starts-with(@id,'delete-book-button')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[starts-with(@id,'delete-book-button')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "DELETE_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login and add a new book.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "profile-tab-add-books",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "profile-tab-add-books",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__AUTHOR__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//label[contains(normalize-space(),'Author')]/input)[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__AUTHOR__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//label[contains(normalize-space(),'Author')]/input)[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__YEAR__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//label[contains(normalize-space(),'Year')]/input)[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__YEAR__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//label[contains(normalize-space(),'Year')]/input)[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__PAGES__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//label[contains(normalize-space(),'Pages')]/input)[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__PAGES__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//label[contains(normalize-space(),'Pages')]/input)[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__RATING__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//label[contains(normalize-space(),'Rating')]/input)[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__RATING__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//label[contains(normalize-space(),'Rating')]/input)[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__GENRE__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//input[@placeholder='Custom genre list'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__GENRE__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//input[@placeholder='Custom genre list'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[normalize-space()='Add Book'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//button[normalize-space()='Add Book'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "ADD_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Open a book and add a comment.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/search')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/search')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-field' or @id='search-input']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__SEARCH_QUERY__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='search-field' or @id='search-input']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//a[contains(@href,'/books/')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//a[contains(@href,'/books/')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "comment-author-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "comment-author-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__COMMENTER_NAME__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "comment-author-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__COMMENTER_NAME__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "comment-author-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "comment-message-textarea",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "comment-message-textarea",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__COMMENT_MESSAGE__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "comment-message-textarea",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__COMMENT_MESSAGE__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "comment-message-textarea",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "share-feedback-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "share-feedback-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "ADD_COMMENT_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login and edit user profile fields.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "first-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "first-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__FIRST_NAME__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "first-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__FIRST_NAME__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "first-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "last-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "last-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__LAST_NAME__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "last-name-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__LAST_NAME__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "last-name-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "website-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "website-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__WEBSITE__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "website-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__WEBSITE__",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "website-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "save-profile-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "save-profile-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "EDIT_USER_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Navigate to '1984' book page",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/search')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/search')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-field' or @id='search-input']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__SEARCH_QUERY__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='search-field' or @id='search-input']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//a[contains(@href,'/books/')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//a[contains(@href,'/books/')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='share-detail-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='share-detail-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "BOOK_DETAIL",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login with username: <username> and password: <password>. Edit a book by changing the rating to 4.8.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='books-view-tab' or @id='profile-tab-books']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='books-view-tab' or @id='profile-tab-books']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__RATING__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//label[contains(normalize-space(),'Rating')]/input)[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__RATING__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//label[contains(normalize-space(),'Rating')]/input)[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//label[contains(normalize-space(),'Pages')]/input)[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//label[contains(normalize-space(),'Pages')]/input)[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__PAGES__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//label[contains(normalize-space(),'Pages')]/input)[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__PAGES__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//label[contains(normalize-space(),'Pages')]/input)[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//input[@placeholder='Custom genre list'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//input[@placeholder='Custom genre list'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__GENRE__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//input[@placeholder='Custom genre list'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__GENRE__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//input[@placeholder='Custom genre list'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//form[.//label[contains(normalize-space(),'Rating')]]//button[@type='submit'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//form[.//label[contains(normalize-space(),'Rating')]]//button[@type='submit'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "EDIT_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login with username: <username> and password: <password>. After logging in, purchase the book 'The Silent Patient'.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/search')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/search')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-field' or @id='search-input']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__SEARCH_QUERY__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='search-field' or @id='search-input']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//a[contains(@href,'/books/')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//a[contains(@href,'/books/')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='add-to-cart-detail-button' or @id='add-cart-button' or @id='cart-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='add-to-cart-detail-button' or @id='add-cart-button' or @id='cart-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/cart')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/cart')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='purchase-button' or @id='buy-now-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='purchase-button' or @id='buy-now-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "PURCHASE_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Open detail and share the book.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//a[contains(@href,'/books/')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//a[contains(@href,'/books/')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "share-detail-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "share-detail-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "SHARE_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Open detail and start preview.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//a[contains(@href,'/books/')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//a[contains(@href,'/books/')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "read-book-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "read-book-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "OPEN_PREVIEW",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login with username: <username> and password: <password>. Add 'The Iliad' to your reading list.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/search')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/search')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__READING_LIST_QUERY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-field' or @id='search-input']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__READING_LIST_QUERY__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='search-field' or @id='search-input']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//a[contains(@href,'/books/')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//a[contains(@href,'/books/')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='reading-list-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='reading-list-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='reading-list-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='reading-list-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "ADD_TO_READING_LIST",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login with username: <username> and password: <password>. Remove 'The Iliad' from your reading list.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/search')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/search')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__READING_LIST_QUERY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-field' or @id='search-input']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__READING_LIST_QUERY__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='search-field' or @id='search-input']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//a[contains(@href,'/books/')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//a[contains(@href,'/books/')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='reading-list-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='reading-list-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='reading-list-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='reading-list-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "REMOVE_FROM_READING_LIST",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login with username: <username> and password: <password>. After logging in, view your shopping cart.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/cart')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/cart')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/search')][1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/search')][1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "VIEW_CART_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login with username: <username> and password: <password>. After logging in, add 'Romeo and Juliet' to your shopping cart.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/search')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/search')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__CART_QUERY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-field' or @id='search-input']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__CART_QUERY__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='search-field' or @id='search-input']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//a[contains(@href,'/books/')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//a[contains(@href,'/books/')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='add-to-cart-detail-button' or @id='add-cart-button' or @id='cart-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='add-to-cart-detail-button' or @id='add-cart-button' or @id='cart-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "ADD_TO_CART_BOOK",
                "has_success": True,
            },
            {
                "url": "http://localhost:8001/?seed=1",
                "prompt": "Login with username: <username> and password: <password>. After logging in, remove 'Romeo and Juliet' from your shopping cart.",
                "actions": [
                    {
                        "url": "http://localhost:8001/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8001/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[normalize-space()='Login']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[normalize-space()='Login']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<username>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "username-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<username>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "username-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "<password>",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "password-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "<password>",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "password-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "login-submit-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "login-submit-button",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/search')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/search')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__CART_QUERY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-field' or @id='search-input']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "__CART_QUERY__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='search-field' or @id='search-input']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//a[contains(@href,'/books/')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//a[contains(@href,'/books/')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='add-to-cart-detail-button' or @id='add-cart-button' or @id='cart-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='add-to-cart-detail-button' or @id='add-cart-button' or @id='cart-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//a[contains(@href,'/cart')]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//a[contains(@href,'/cart')]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[contains(@id,'remove-from-cart-button') or contains(@id,'delete-cart-item-button') or contains(@id,'remove-cart')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[contains(@id,'remove-from-cart-button') or contains(@id,'delete-cart-item-button') or contains(@id,'remove-cart')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "REMOVE_FROM_CART_BOOK",
                "has_success": True,
            },
        ],
    },
    {
        "project_id": "p03_autozone",
        "trajectories": [
            {
                "url": "http://localhost:8002/?seed=18",
                "prompt": "Show me details for the Premium Drone",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=18",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=18",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Premium Drone",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Premium Drone",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "VIEW_DETAIL",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=18",
                "prompt": "Expand the Explore further section for the Premium Drone page.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=18",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=18",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Premium Drone",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Premium Drone",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='toggle-btn' or @id='toggle-button' or @id='switch-btn' or @id='toggle-control' or @id='toggle-action' or @id='switch-control' or @id='toggle-state' or @id='toggle-option' or @id='toggle-choice' or @id='toggle']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='toggle-btn' or @id='toggle-button' or @id='switch-btn' or @id='toggle-control' or @id='toggle-action' or @id='switch-control' or @id='toggle-state' or @id='toggle-option' or @id='toggle-choice' or @id='toggle']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "DETAILS_TOGGLE",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "Search for products that contain 'Premium Drone'",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Premium Drone",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Premium Drone",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "SEARCH_PRODUCT",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "Filter results to Technology products.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='browse-all-button' or @id='browse-all-btn' or @id='browse-btn' or @id='browse-button' or @id='browse-all-items-btn' or @id='browse-items-btn' or @id='browse-catalog-btn' or @id='browse-list-btn' or @id='open-browse-btn' or @id='browse-more-btn']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='browse-all-button' or @id='browse-all-btn' or @id='browse-btn' or @id='browse-button' or @id='browse-all-items-btn' or @id='browse-items-btn' or @id='browse-catalog-btn' or @id='browse-list-btn' or @id='open-browse-btn' or @id='browse-more-btn']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='category-link' or @id='cat-link' or @id='category-btn' or @id='cat-btn' or @id='category-action' or @id='cat-action' or @id='browse-category' or @id='view-category' or @id='goto-category' or @id='category-nav'][contains(translate(normalize-space(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'technology')])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='category-link' or @id='cat-link' or @id='category-btn' or @id='cat-btn' or @id='category-action' or @id='cat-action' or @id='browse-category' or @id='view-category' or @id='goto-category' or @id='category-nav'][contains(translate(normalize-space(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'technology')])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "CATEGORY_FILTER",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "Add the Premium Drone to my cart.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Premium Drone",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Premium Drone",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='add-cart-btn' or @id='cart-add' or @id='add-basket' or @id='add-to-basket' or @id='cart-action' or @id='basket-action' or @id='add-item' or @id='cart-item-add' or @id='basket-add-item' or @id='add-product'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='add-cart-btn' or @id='cart-add' or @id='add-basket' or @id='add-to-basket' or @id='cart-action' or @id='basket-action' or @id='add-item' or @id='cart-item-add' or @id='basket-add-item' or @id='add-product'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "ADD_TO_CART",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "Add the Premium Drone to my wishlist.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Premium Drone",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Premium Drone",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "SendKeysAction",
                        "keys": [
                            "Enter",
                        ],
                        "attributes": {
                            "keys": [
                                "Enter",
                            ],
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='wishlist-btn' or @id='add-wishlist' or @id='save-later' or @id='wishlist-add' or @id='favorite-btn' or @id='save-item' or @id='add-favorite' or @id='wishlist-action' or @id='save-product' or @id='favorite-action']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='wishlist-btn' or @id='add-wishlist' or @id='save-later' or @id='wishlist-add' or @id='favorite-btn' or @id='save-item' or @id='add-favorite' or @id='wishlist-action' or @id='save-product' or @id='favorite-action']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "ADD_TO_WISHLIST",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "Open my wishlist page.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='view-wishlist-button' or @id='wishlist-btn' or @id='wishlist-link' or @id='go-wishlist' or @id='view-wishlist-btn' or @id='wishlist-button' or @id='show-wishlist-btn' or @id='open-wishlist-btn' or @id='wishlist-view-btn' or @id='all-wishlist-btn' or @id='save-later']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='view-wishlist-button' or @id='wishlist-btn' or @id='wishlist-link' or @id='go-wishlist' or @id='view-wishlist-btn' or @id='wishlist-button' or @id='show-wishlist-btn' or @id='open-wishlist-btn' or @id='wishlist-view-btn' or @id='all-wishlist-btn' or @id='save-later']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "VIEW_WISHLIST",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "Open my cart page.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='cart-btn' or @id='shopping-cart' or @id='basket-btn' or @id='cart-action' or @id='view-cart' or @id='goto-cart' or @id='cart-link' or @id='basket-link' or @id='cart-icon' or @id='shopping-basket']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='cart-btn' or @id='shopping-cart' or @id='basket-btn' or @id='cart-action' or @id='view-cart' or @id='goto-cart' or @id='cart-link' or @id='basket-link' or @id='cart-icon' or @id='shopping-basket']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "VIEW_CART",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "On Premium Drone details, change quantity from 1 to 2.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Premium Drone",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Premium Drone",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='qty-input' or @id='quantity-field' or @id='qty-field' or @id='amount-input' or @id='qty-box' or @id='quantity-box' or @id='qty-select' or @id='quantity-select' or @id='item-qty' or @id='product-qty']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='qty-input' or @id='quantity-field' or @id='qty-field' or @id='amount-input' or @id='qty-box' or @id='quantity-box' or @id='qty-select' or @id='quantity-select' or @id='item-qty' or @id='product-qty']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "SendKeysAction",
                        "keys": [
                            "ArrowDown",
                        ],
                        "attributes": {
                            "keys": [
                                "ArrowDown",
                            ],
                        },
                    },
                    {
                        "type": "SendKeysAction",
                        "keys": [
                            "Enter",
                        ],
                        "attributes": {
                            "keys": [
                                "Enter",
                            ],
                        },
                    },
                ],
                "use_case": "QUANTITY_CHANGED",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "From cart, proceed to checkout with the Premium Drone.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Premium Drone",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Premium Drone",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='add-cart-btn' or @id='cart-add' or @id='add-basket' or @id='add-to-basket' or @id='cart-action' or @id='basket-action' or @id='add-item' or @id='cart-item-add' or @id='basket-add-item' or @id='add-product'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='add-cart-btn' or @id='cart-add' or @id='add-basket' or @id='add-to-basket' or @id='cart-action' or @id='basket-action' or @id='add-item' or @id='cart-item-add' or @id='basket-add-item' or @id='add-product'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='checkout-btn' or @id='proceed-checkout' or @id='goto-checkout' or @id='checkout-action' or @id='checkout-now' or @id='proceed-btn' or @id='finalize-order' or @id='complete-order' or @id='checkout-link' or @id='order-btn']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='checkout-btn' or @id='proceed-checkout' or @id='goto-checkout' or @id='checkout-action' or @id='checkout-now' or @id='proceed-btn' or @id='finalize-order' or @id='complete-order' or @id='checkout-link' or @id='order-btn']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "PROCEED_TO_CHECKOUT",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "Start checkout from the Premium Drone detail page.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Premium Drone",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Premium Drone",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='order-now' or @id='checkout-btn' or @id='proceed-checkout' or @id='goto-checkout' or @id='checkout-action' or @id='checkout-now' or @id='proceed-btn' or @id='finalize-order' or @id='complete-order' or @id='checkout-link' or @id='order-btn']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='order-now' or @id='checkout-btn' or @id='proceed-checkout' or @id='goto-checkout' or @id='checkout-action' or @id='checkout-now' or @id='proceed-btn' or @id='finalize-order' or @id='complete-order' or @id='checkout-link' or @id='order-btn']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "CHECKOUT_STARTED",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "Share the Premium Drone product page.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Premium Drone",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Premium Drone",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='share-btn' or @id='share-button' or @id='share-action' or @id='share-link' or @id='share-control' or @id='share-item' or @id='share-product' or @id='share-page' or @id='share-trigger' or @id='share']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='share-btn' or @id='share-button' or @id='share-action' or @id='share-link' or @id='share-control' or @id='share-item' or @id='share-product' or @id='share-page' or @id='share-trigger' or @id='share']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "SHARE_PRODUCT",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "Scroll right in the Featured Products carousel.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='carousel-right-btn' or @id='carousel-next' or @id='carousel-forward' or @id='carousel-right' or @id='carousel-next-btn' or @id='carousel-arrow-right' or @id='carousel-control-right' or @id='carousel-nav-right' or @id='carousel-right-control' or @id='carousel-right-arrow']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='carousel-right-btn' or @id='carousel-next' or @id='carousel-forward' or @id='carousel-right' or @id='carousel-next-btn' or @id='carousel-arrow-right' or @id='carousel-control-right' or @id='carousel-nav-right' or @id='carousel-right-control' or @id='carousel-right-arrow']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "CAROUSEL_SCROLL",
                "has_success": True,
            },
            {
                "url": "http://localhost:8002/?seed=15",
                "prompt": "Complete an order for the Premium Drone.",
                "actions": [
                    {
                        "url": "http://localhost:8002/?seed=15",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8002/?seed=15",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Premium Drone",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Premium Drone",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' or @id='find-input' or @id='search-box']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='execute-search' or @id='search-btn' or @id='submit-search' or @id='go-search' or @id='search-action' or @id='find-btn' or @id='query-btn' or @id='search-submit' or @id='do-search' or @id='run-search']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='view-details-btn' or @id='details-btn' or @id='view-btn' or @id='open-details' or @id='view-details' or @id='details-action' or @id='product-details-btn' or @id='item-details-btn' or @id='more-details-btn' or @id='details-link'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='add-cart-btn' or @id='cart-add' or @id='add-basket' or @id='add-to-basket' or @id='cart-action' or @id='basket-action' or @id='add-item' or @id='cart-item-add' or @id='basket-add-item' or @id='add-product'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='add-cart-btn' or @id='cart-add' or @id='add-basket' or @id='add-to-basket' or @id='cart-action' or @id='basket-action' or @id='add-item' or @id='cart-item-add' or @id='basket-add-item' or @id='add-product'])[1]",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='checkout-btn' or @id='proceed-checkout' or @id='goto-checkout' or @id='checkout-action' or @id='checkout-now' or @id='proceed-btn' or @id='finalize-order' or @id='complete-order' or @id='checkout-link' or @id='order-btn']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='checkout-btn' or @id='proceed-checkout' or @id='goto-checkout' or @id='checkout-action' or @id='checkout-now' or @id='proceed-btn' or @id='finalize-order' or @id='complete-order' or @id='checkout-link' or @id='order-btn']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='finalize-order-button' or @id='place-order-btn' or @id='complete-order-button' or @id='confirm-order-btn' or @id='submit-order-button' or @id='finish-order-btn' or @id='order-now-button' or @id='checkout-button' or @id='place-order-button' or @id='confirm-purchase-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='finalize-order-button' or @id='place-order-btn' or @id='complete-order-button' or @id='confirm-order-btn' or @id='submit-order-button' or @id='finish-order-btn' or @id='order-now-button' or @id='checkout-button' or @id='place-order-button' or @id='confirm-purchase-button']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "ORDER_COMPLETED",
                "has_success": True,
            },
        ],
    },
    {
        "project_id": "p04_autodining",
        "trajectories": [
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Show me details for 'Thai Garden'",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Thai Garden",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Thai Garden",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//input[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view_details_button' or @id='dining-view-details-button' or @id='resto-view-details-button' or @id='view-details-btn' or @id='dining-view-details-btn' or @id='resto-view-details-btn' or @id='view-details-action'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='view_details_button' or @id='dining-view-details-button' or @id='resto-view-details-button' or @id='view-details-btn' or @id='dining-view-details-btn' or @id='resto-view-details-btn' or @id='view-details-action'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "VIEW_RESTAURANT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/restaurant/7?seed=1",
                "prompt": "Show the full menu for 'Thai Garden' for 2 people for dinner on July 18.",
                "actions": [
                    {
                        "url": "http://localhost:8003/restaurant/7?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/restaurant/7?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='menu-toggle-button' or @id='dining-menu-toggle' or @id='resto-menu-toggle' or @id='menu-expand-button' or @id='dining-menu-expand' or @id='resto-menu-expand' or @id='menu-collapse-button' or @id='dining-menu-collapse' or @id='resto-menu-collapse' or @id='menu-view-toggle'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='menu-toggle-button' or @id='dining-menu-toggle' or @id='resto-menu-toggle' or @id='menu-expand-button' or @id='dining-menu-expand' or @id='resto-menu-expand' or @id='menu-collapse-button' or @id='dining-menu-collapse' or @id='resto-menu-collapse' or @id='menu-view-toggle'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "VIEW_FULL_MENU",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/restaurant/7?seed=1",
                "prompt": "Hide the menu for 'Thai Garden'.",
                "actions": [
                    {
                        "url": "http://localhost:8003/restaurant/7?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/restaurant/7?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='menu-toggle-button' or @id='dining-menu-toggle' or @id='resto-menu-toggle' or @id='menu-expand-button' or @id='dining-menu-expand' or @id='resto-menu-expand' or @id='menu-collapse-button' or @id='dining-menu-collapse' or @id='resto-menu-collapse' or @id='menu-view-toggle'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='menu-toggle-button' or @id='dining-menu-toggle' or @id='resto-menu-toggle' or @id='menu-expand-button' or @id='dining-menu-expand' or @id='resto-menu-expand' or @id='menu-collapse-button' or @id='dining-menu-collapse' or @id='resto-menu-collapse' or @id='menu-view-toggle'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='menu-toggle-button' or @id='dining-menu-toggle' or @id='resto-menu-toggle' or @id='menu-expand-button' or @id='dining-menu-expand' or @id='resto-menu-expand' or @id='menu-collapse-button' or @id='dining-menu-collapse' or @id='resto-menu-collapse' or @id='menu-view-toggle'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='menu-toggle-button' or @id='dining-menu-toggle' or @id='resto-menu-toggle' or @id='menu-expand-button' or @id='dining-menu-expand' or @id='resto-menu-expand' or @id='menu-collapse-button' or @id='dining-menu-collapse' or @id='resto-menu-collapse' or @id='menu-view-toggle'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "COLLAPSE_MENU",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Open the date selector and select the date '2026-02-23'.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='date_picker' or @id='date-picker' or @id='date-selector' or @id='date-input' or @id='booking-date' or @id='reservation-date' or @id='calendar-trigger' or @id='date-trigger' or @id='checkin-date' or @id='date-field']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='date_picker' or @id='date-picker' or @id='date-selector' or @id='date-input' or @id='booking-date' or @id='reservation-date' or @id='calendar-trigger' or @id='date-trigger' or @id='checkin-date' or @id='date-field']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[@aria-label='Go to previous month' or @aria-label='Previous month'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//button[@aria-label='Go to previous month' or @aria-label='Previous month'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[@aria-label='Go to previous month' or @aria-label='Previous month'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//button[@aria-label='Go to previous month' or @aria-label='Previous month'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[normalize-space()='23' and not(@disabled)])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//button[normalize-space()='23' and not(@disabled)])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "DATE_DROPDOWN_OPENED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Open the time dropdown and select the time equals '2:30 PM'.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='time_picker' or @id='time-picker' or @id='time-selector' or @id='time-input' or @id='booking-time' or @id='reservation-time' or @id='time-trigger' or @id='checkin-time' or @id='time-field']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='time_picker' or @id='time-picker' or @id='time-selector' or @id='time-input' or @id='booking-time' or @id='reservation-time' or @id='time-trigger' or @id='checkin-time' or @id='time-field']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[normalize-space()='2:30 PM'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//button[normalize-space()='2:30 PM'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "TIME_DROPDOWN_OPENED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Open the guest selector dropdown and select people equals 4.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='people_picker' or @id='people-picker' or @id='guest-picker' or @id='guests-picker' or @id='people-selector' or @id='guest-selector' or @id='booking-people' or @id='reservation-people' or @id='people-input' or @id='guests-input']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='people_picker' or @id='people-picker' or @id='guest-picker' or @id='guests-picker' or @id='people-selector' or @id='guest-selector' or @id='booking-people' or @id='reservation-people' or @id='people-input' or @id='guests-input']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[contains(normalize-space(), '4') and (contains(normalize-space(), 'Guest') or contains(normalize-space(), 'Guests'))])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//button[contains(normalize-space(), '4') and (contains(normalize-space(), 'Guest') or contains(normalize-space(), 'Guests'))])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "PEOPLE_DROPDOWN_OPENED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Search for 'Thai Garden'",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Thai Garden",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Thai Garden",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//input[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "SEARCH_RESTAURANT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Scroll in the direction 'right' where section equals 'Featured Products'.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@data-testid='scroll-right-1' or @id='scroll-right-button' or @id='scroll-right' or @id='scroll-right-btn' or @id='carousel-right' or @id='carousel-right-button' or @id='carousel-right-btn' or @id='next-button' or @id='next-slide' or @id='right-arrow'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@data-testid='scroll-right-1' or @id='scroll-right-button' or @id='scroll-right' or @id='scroll-right-btn' or @id='carousel-right' or @id='carousel-right-button' or @id='carousel-right-btn' or @id='next-button' or @id='next-slide' or @id='right-arrow'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "SCROLL_VIEW",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "I'd like to book a table at the restaurant which name 'Thai Garden' for 2 people on 2026-04-03 at 12:00 PM.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Thai Garden",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Thai Garden",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//input[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[normalize-space()='Thai Garden'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[normalize-space()='Thai Garden'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='people_picker' or @id='dining-people-picker' or @id='resto-people-picker' or @id='guests-picker' or @id='dining-guests-picker' or @id='resto-guests-picker' or @id='party-size-picker' or @id='dining-party-size-picker' or @id='resto-party-size-picker' or @id='people-selector']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='people_picker' or @id='dining-people-picker' or @id='resto-people-picker' or @id='guests-picker' or @id='dining-guests-picker' or @id='resto-guests-picker' or @id='party-size-picker' or @id='dining-party-size-picker' or @id='resto-party-size-picker' or @id='people-selector']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[contains(normalize-space(), '2') and (contains(normalize-space(), 'Guest') or contains(normalize-space(), 'Guests'))])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//button[contains(normalize-space(), '2') and (contains(normalize-space(), 'Guest') or contains(normalize-space(), 'Guests'))])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='date_picker' or @id='dining-date-picker' or @id='resto-date-picker' or @id='booking-date-picker' or @id='dining-booking-date-picker' or @id='resto-booking-date-picker' or @id='date-selector' or @id='dining-date-selector' or @id='resto-date-selector' or @id='date-field']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='date_picker' or @id='dining-date-picker' or @id='resto-date-picker' or @id='booking-date-picker' or @id='dining-booking-date-picker' or @id='resto-booking-date-picker' or @id='date-selector' or @id='dining-date-selector' or @id='resto-date-selector' or @id='date-field']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[normalize-space()='3' and not(@disabled)])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//button[normalize-space()='3' and not(@disabled)])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='time_picker' or @id='dining-time-picker' or @id='resto-time-picker' or @id='booking-time-picker' or @id='dining-booking-time-picker' or @id='resto-booking-time-picker' or @id='time-selector' or @id='dining-time-selector' or @id='resto-time-selector' or @id='time-field']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='time_picker' or @id='dining-time-picker' or @id='resto-time-picker' or @id='booking-time-picker' or @id='dining-booking-time-picker' or @id='resto-booking-time-picker' or @id='time-selector' or @id='dining-time-selector' or @id='resto-time-selector' or @id='time-field']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[normalize-space()='12:00 PM'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//button[normalize-space()='12:00 PM'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='book_button' or @id='dining-book-button' or @id='resto-book-button' or @id='booking-button' or @id='dining-booking-button' or @id='resto-booking-button' or @id='reserve-button' or @id='dining-reserve-button' or @id='resto-reserve-button' or @id='book-action-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='book_button' or @id='dining-book-button' or @id='resto-book-button' or @id='booking-button' or @id='dining-booking-button' or @id='resto-booking-button' or @id='reserve-button' or @id='dining-reserve-button' or @id='resto-reserve-button' or @id='book-action-button']",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "BOOK_RESTAURANT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/booking/7/12%3A00%20PM?seed=1&people=2&date=2026-04-03",
                "prompt": "Select a country where code equals 'IN'.",
                "actions": [
                    {
                        "url": "http://localhost:8003/booking/7/12%3A00%20PM?seed=1&people=2&date=2026-04-03",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/booking/7/12%3A00%20PM?seed=1&people=2&date=2026-04-03",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='country-select' or @id='country-dropdown' or @id='country-picker' or @id='country-selector' or @id='country-choice' or @id='country-option' or @id='country-field' or @id='country-input' or @id='country-selection' or @id='country-picker-dropdown']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='country-select' or @id='country-dropdown' or @id='country-picker' or @id='country-selector' or @id='country-choice' or @id='country-option' or @id='country-field' or @id='country-input' or @id='country-selection' or @id='country-picker-dropdown']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='country-select' or @id='country-dropdown' or @id='country-picker' or @id='country-selector' or @id='country-choice' or @id='country-option' or @id='country-field' or @id='country-input' or @id='country-selection' or @id='country-picker-dropdown']/option[@value='IN'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='country-select' or @id='country-dropdown' or @id='country-picker' or @id='country-selector' or @id='country-choice' or @id='country-option' or @id='country-field' or @id='country-input' or @id='country-selection' or @id='country-picker-dropdown']/option[@value='IN'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "COUNTRY_SELECTED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/booking/7/12%3A00%20PM?seed=1&people=2&date=2026-04-03",
                "prompt": "This reservation is for a 'birthday'.",
                "actions": [
                    {
                        "url": "http://localhost:8003/booking/7/12%3A00%20PM?seed=1&people=2&date=2026-04-03",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/booking/7/12%3A00%20PM?seed=1&people=2&date=2026-04-03",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='occasion-select' or @id='occasion-dropdown' or @id='occasion-picker' or @id='occasion-selector' or @id='occasion-choice' or @id='occasion-option' or @id='occasion-field' or @id='occasion-input' or @id='occasion-selection' or @id='occasion-picker-dropdown']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='occasion-select' or @id='occasion-dropdown' or @id='occasion-picker' or @id='occasion-selector' or @id='occasion-choice' or @id='occasion-option' or @id='occasion-field' or @id='occasion-input' or @id='occasion-selection' or @id='occasion-picker-dropdown']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='occasion-select' or @id='occasion-dropdown' or @id='occasion-picker' or @id='occasion-selector' or @id='occasion-choice' or @id='occasion-option' or @id='occasion-field' or @id='occasion-input' or @id='occasion-selection' or @id='occasion-picker-dropdown']/option[@value='birthday'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='occasion-select' or @id='occasion-dropdown' or @id='occasion-picker' or @id='occasion-selector' or @id='occasion-choice' or @id='occasion-option' or @id='occasion-field' or @id='occasion-input' or @id='occasion-selection' or @id='occasion-picker-dropdown']/option[@value='birthday'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "OCCASION_SELECTED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/booking/7/12%3A00%20PM?seed=1&people=2&date=2026-04-03",
                "prompt": "Complete my reservation for 'Thai Garden' on 2026-04-03 at 12:00 PM for 2 people. My phone is 666777888, it's for an anniversary, and special request is 'delicious'.",
                "actions": [
                    {
                        "url": "http://localhost:8003/booking/7/12%3A00%20PM?seed=1&people=2&date=2026-04-03",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/booking/7/12%3A00%20PM?seed=1&people=2&date=2026-04-03",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='full-name-input' or @id='name-input' or @id='full-name' or @id='booking-name' or @id='reservation-name' or @id='customer-name' or @id='fullname-input' or @id='name-field' or @id='guest-name' or @id='full-name-field']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='full-name-input' or @id='name-input' or @id='full-name' or @id='booking-name' or @id='reservation-name' or @id='customer-name' or @id='fullname-input' or @id='name-field' or @id='guest-name' or @id='full-name-field']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "agent1",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='full-name-input' or @id='name-input' or @id='full-name' or @id='booking-name' or @id='reservation-name' or @id='customer-name' or @id='fullname-input' or @id='name-field' or @id='guest-name' or @id='full-name-field']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "agent1",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//input[@id='full-name-input' or @id='name-input' or @id='full-name' or @id='booking-name' or @id='reservation-name' or @id='customer-name' or @id='fullname-input' or @id='name-field' or @id='guest-name' or @id='full-name-field']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='phone-number-input' or @id='phone-input' or @id='booking-phone' or @id='reservation-phone' or @id='customer-phone' or @id='phone-field' or @id='mobile-input' or @id='contact-phone' or @id='phone-number' or @id='phone']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='phone-number-input' or @id='phone-input' or @id='booking-phone' or @id='reservation-phone' or @id='customer-phone' or @id='phone-field' or @id='mobile-input' or @id='contact-phone' or @id='phone-number' or @id='phone']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "666777888",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='phone-number-input' or @id='phone-input' or @id='booking-phone' or @id='reservation-phone' or @id='customer-phone' or @id='phone-field' or @id='mobile-input' or @id='contact-phone' or @id='phone-number' or @id='phone']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "666777888",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//input[@id='phone-number-input' or @id='phone-input' or @id='booking-phone' or @id='reservation-phone' or @id='customer-phone' or @id='phone-field' or @id='mobile-input' or @id='contact-phone' or @id='phone-number' or @id='phone']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='email-input' or @id='booking-email' or @id='reservation-email' or @id='customer-email' or @id='email-field' or @id='contact-email' or @id='email-address-input' or @id='email-address' or @id='email']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='email-input' or @id='booking-email' or @id='reservation-email' or @id='customer-email' or @id='email-field' or @id='contact-email' or @id='email-address-input' or @id='email-address' or @id='email']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "user_name@gmail.com",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='email-input' or @id='booking-email' or @id='reservation-email' or @id='customer-email' or @id='email-field' or @id='contact-email' or @id='email-address-input' or @id='email-address' or @id='email']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "user_name@gmail.com",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//input[@id='email-input' or @id='booking-email' or @id='reservation-email' or @id='customer-email' or @id='email-field' or @id='contact-email' or @id='email-address-input' or @id='email-address' or @id='email']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='occasion-select' or @id='occasion-dropdown' or @id='occasion-picker' or @id='occasion-selector' or @id='occasion-choice' or @id='occasion-option' or @id='occasion-field' or @id='occasion-input' or @id='occasion-selection' or @id='occasion-picker-dropdown']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='occasion-select' or @id='occasion-dropdown' or @id='occasion-picker' or @id='occasion-selector' or @id='occasion-choice' or @id='occasion-option' or @id='occasion-field' or @id='occasion-input' or @id='occasion-selection' or @id='occasion-picker-dropdown']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='occasion-select' or @id='occasion-dropdown' or @id='occasion-picker' or @id='occasion-selector' or @id='occasion-choice' or @id='occasion-option' or @id='occasion-field' or @id='occasion-input' or @id='occasion-selection' or @id='occasion-picker-dropdown']/option[@value='anniversary'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='occasion-select' or @id='occasion-dropdown' or @id='occasion-picker' or @id='occasion-selector' or @id='occasion-choice' or @id='occasion-option' or @id='occasion-field' or @id='occasion-input' or @id='occasion-selection' or @id='occasion-picker-dropdown']/option[@value='anniversary'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='special-requests-textarea' or @id='special-request-textarea' or @id='special-request' or @id='special-requests' or @id='request-textarea' or @id='booking-request' or @id='reservation-request' or @id='notes-textarea' or @id='comments-textarea' or @id='special-notes']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='special-requests-textarea' or @id='special-request-textarea' or @id='special-request' or @id='special-requests' or @id='request-textarea' or @id='booking-request' or @id='reservation-request' or @id='notes-textarea' or @id='comments-textarea' or @id='special-notes']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "delicious",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//textarea[@id='special-requests-textarea' or @id='special-request-textarea' or @id='special-request' or @id='special-requests' or @id='request-textarea' or @id='booking-request' or @id='reservation-request' or @id='notes-textarea' or @id='comments-textarea' or @id='special-notes']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "delicious",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//textarea[@id='special-requests-textarea' or @id='special-request-textarea' or @id='special-request' or @id='special-requests' or @id='request-textarea' or @id='booking-request' or @id='reservation-request' or @id='notes-textarea' or @id='comments-textarea' or @id='special-notes']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='confirm_button' or @id='confirm-booking-button' or @id='complete-reservation-button' or @id='finalize-reservation-button' or @id='submit-reservation-button' or @id='reservation-confirm-button' or @id='finish-booking-button' or @id='complete-booking-button' or @id='confirm-reservation-button' or @id='reservation-submit-button']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='confirm_button' or @id='confirm-booking-button' or @id='complete-reservation-button' or @id='finalize-reservation-button' or @id='submit-reservation-button' or @id='reservation-confirm-button' or @id='finish-booking-button' or @id='complete-booking-button' or @id='confirm-reservation-button' or @id='reservation-submit-button']",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "RESERVATION_COMPLETE",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Contact where name equals 'James'.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='nav-contact' or @id='contact-link' or @id='contact-us-link' or @id='contact-nav' or @id='contact-us-nav' or @id='contact-button' or @id='contact-us-button' or @id='contact-menu-item' or @id='contact-us-menu-item' or @id='contact-navigation']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='nav-contact' or @id='contact-link' or @id='contact-us-link' or @id='contact-nav' or @id='contact-us-nav' or @id='contact-button' or @id='contact-us-button' or @id='contact-menu-item' or @id='contact-us-menu-item' or @id='contact-navigation']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='contact-name-input' or @id='name-input-contact' or @id='contact-name-field' or @id='name-field-contact' or @id='contact-name-text-input' or @id='name-text-input-contact' or @id='contact-name-entry' or @id='name-entry-contact' or @id='contact-name-textbox' or @id='name-textbox-contact']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='contact-name-input' or @id='name-input-contact' or @id='contact-name-field' or @id='name-field-contact' or @id='contact-name-text-input' or @id='name-text-input-contact' or @id='contact-name-entry' or @id='name-entry-contact' or @id='contact-name-textbox' or @id='name-textbox-contact']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "agent1",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='contact-name-input' or @id='name-input-contact' or @id='contact-name-field' or @id='name-field-contact' or @id='contact-name-text-input' or @id='name-text-input-contact' or @id='contact-name-entry' or @id='name-entry-contact' or @id='contact-name-textbox' or @id='name-textbox-contact']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "agent1",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//input[@id='contact-name-input' or @id='name-input-contact' or @id='contact-name-field' or @id='name-field-contact' or @id='contact-name-text-input' or @id='name-text-input-contact' or @id='contact-name-entry' or @id='name-entry-contact' or @id='contact-name-textbox' or @id='name-textbox-contact']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='contact-email-input' or @id='email-input-contact' or @id='contact-email-field' or @id='email-field-contact' or @id='contact-email-text-input' or @id='email-text-input-contact' or @id='contact-email-entry' or @id='email-entry-contact' or @id='contact-email-textbox' or @id='email-textbox-contact']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='contact-email-input' or @id='email-input-contact' or @id='contact-email-field' or @id='email-field-contact' or @id='contact-email-text-input' or @id='email-text-input-contact' or @id='contact-email-entry' or @id='email-entry-contact' or @id='contact-email-textbox' or @id='email-textbox-contact']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "gg@hairmail.com",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='contact-email-input' or @id='email-input-contact' or @id='contact-email-field' or @id='email-field-contact' or @id='contact-email-text-input' or @id='email-text-input-contact' or @id='contact-email-entry' or @id='email-entry-contact' or @id='contact-email-textbox' or @id='email-textbox-contact']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "gg@hairmail.com",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//input[@id='contact-email-input' or @id='email-input-contact' or @id='contact-email-field' or @id='email-field-contact' or @id='contact-email-text-input' or @id='email-text-input-contact' or @id='contact-email-entry' or @id='email-entry-contact' or @id='contact-email-textbox' or @id='email-textbox-contact']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='contact-subject-input' or @id='subject-input-contact' or @id='contact-subject-field' or @id='subject-field-contact' or @id='contact-subject-text-input' or @id='subject-text-input-contact' or @id='contact-subject-entry' or @id='subject-entry-contact' or @id='contact-subject-textbox' or @id='subject-textbox-contact']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='contact-subject-input' or @id='subject-input-contact' or @id='contact-subject-field' or @id='subject-field-contact' or @id='contact-subject-text-input' or @id='subject-text-input-contact' or @id='contact-subject-entry' or @id='subject-entry-contact' or @id='contact-subject-textbox' or @id='subject-textbox-contact']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "hesitations",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='contact-subject-input' or @id='subject-input-contact' or @id='contact-subject-field' or @id='subject-field-contact' or @id='contact-subject-text-input' or @id='subject-text-input-contact' or @id='contact-subject-entry' or @id='subject-entry-contact' or @id='contact-subject-textbox' or @id='subject-textbox-contact']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "hesitations",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//input[@id='contact-subject-input' or @id='subject-input-contact' or @id='contact-subject-field' or @id='subject-field-contact' or @id='contact-subject-text-input' or @id='subject-text-input-contact' or @id='contact-subject-entry' or @id='subject-entry-contact' or @id='contact-subject-textbox' or @id='subject-textbox-contact']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='contact-message-textarea' or @id='message-textarea-contact' or @id='contact-message-field' or @id='message-field-contact' or @id='contact-message-text-area' or @id='message-text-area-contact' or @id='contact-message-entry' or @id='message-entry-contact' or @id='contact-message-textbox' or @id='message-textbox-contact']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='contact-message-textarea' or @id='message-textarea-contact' or @id='contact-message-field' or @id='message-field-contact' or @id='contact-message-text-area' or @id='message-text-area-contact' or @id='contact-message-entry' or @id='message-entry-contact' or @id='contact-message-textbox' or @id='message-textbox-contact']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "idk where is the cheeckout button",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//textarea[@id='contact-message-textarea' or @id='message-textarea-contact' or @id='contact-message-field' or @id='message-field-contact' or @id='contact-message-text-area' or @id='message-text-area-contact' or @id='contact-message-entry' or @id='message-entry-contact' or @id='contact-message-textbox' or @id='message-textbox-contact']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "idk where is the cheeckout button",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//textarea[@id='contact-message-textarea' or @id='message-textarea-contact' or @id='contact-message-field' or @id='message-field-contact' or @id='contact-message-text-area' or @id='message-text-area-contact' or @id='contact-message-entry' or @id='message-entry-contact' or @id='contact-message-textbox' or @id='message-textbox-contact']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='send-message-button' or @id='send-message-btn' or @id='submit-contact-form' or @id='contact-submit-button' or @id='message-submit-button' or @id='send-contact-button' or @id='contact-send-button' or @id='submit-message-button' or @id='contact-form-submit' or @id='send-btn']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='send-message-button' or @id='send-message-btn' or @id='submit-contact-form' or @id='contact-submit-button' or @id='message-submit-button' or @id='send-contact-button' or @id='contact-send-button' or @id='submit-message-button' or @id='contact-form-submit' or @id='send-btn']",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "CONTACT_FORM_SUBMIT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Navigate to the About page to read about the company's mission and values.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='nav-about' or @id='about-link' or @id='about-us-link' or @id='about-nav' or @id='about-us-nav' or @id='about-button' or @id='about-us-button' or @id='about-menu-item' or @id='about-us-menu-item' or @id='about-navigation']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='nav-about' or @id='about-link' or @id='about-us-link' or @id='about-nav' or @id='about-us-nav' or @id='about-button' or @id='about-us-button' or @id='about-menu-item' or @id='about-us-menu-item' or @id='about-navigation']",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "ABOUT_PAGE_VIEW",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Navigate to the Help page to view frequently asked questions and support guides.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='nav-help' or @id='help-link' or @id='support-link' or @id='help-nav' or @id='support-nav' or @id='help-button' or @id='support-button' or @id='help-menu-item' or @id='support-menu-item' or @id='help-navigation']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='nav-help' or @id='help-link' or @id='support-link' or @id='help-nav' or @id='support-nav' or @id='help-button' or @id='support-button' or @id='help-menu-item' or @id='support-menu-item' or @id='help-navigation']",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "HELP_PAGE_VIEW",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Click the Trending Spots feature on the About page.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='nav-about' or @id='about-link' or @id='about-us-link' or @id='about-nav' or @id='about-us-nav' or @id='about-button' or @id='about-us-button' or @id='about-menu-item' or @id='about-us-menu-item' or @id='about-navigation']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='nav-about' or @id='about-link' or @id='about-us-link' or @id='about-nav' or @id='about-us-nav' or @id='about-button' or @id='about-us-button' or @id='about-menu-item' or @id='about-us-menu-item' or @id='about-navigation']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[normalize-space()='Trending Spots' or normalize-space()='Easy Reservations' or normalize-space()='Curated Restaurants'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[normalize-space()='Trending Spots' or normalize-space()='Easy Reservations' or normalize-space()='Curated Restaurants'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "ABOUT_FEATURE_CLICK",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Open the contact page.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='nav-contact' or @id='contact-link' or @id='contact-us-link' or @id='contact-nav' or @id='contact-us-nav' or @id='contact-button' or @id='contact-us-button' or @id='contact-menu-item' or @id='contact-us-menu-item' or @id='contact-navigation']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='nav-contact' or @id='contact-link' or @id='contact-us-link' or @id='contact-nav' or @id='contact-us-nav' or @id='contact-button' or @id='contact-us-button' or @id='contact-menu-item' or @id='contact-us-menu-item' or @id='contact-navigation']",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "CONTACT_PAGE_VIEW",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Click the phone contact card on the contact page.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='nav-contact' or @id='contact-link' or @id='contact-us-link' or @id='contact-nav' or @id='contact-us-nav' or @id='contact-button' or @id='contact-us-button' or @id='contact-menu-item' or @id='contact-us-menu-item' or @id='contact-navigation']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='nav-contact' or @id='contact-link' or @id='contact-us-link' or @id='contact-nav' or @id='contact-us-nav' or @id='contact-button' or @id='contact-us-button' or @id='contact-menu-item' or @id='contact-us-menu-item' or @id='contact-navigation']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//a[starts-with(@href,'tel:') or .//*[normalize-space()='Call Us']])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//a[starts-with(@href,'tel:') or .//*[normalize-space()='Call Us']])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "CONTACT_CARD_CLICK",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Select the Payments category in Help.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='nav-help' or @id='help-link' or @id='support-link' or @id='help-nav' or @id='support-nav' or @id='help-button' or @id='support-button' or @id='help-menu-item' or @id='support-menu-item' or @id='help-navigation']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='nav-help' or @id='help-link' or @id='support-link' or @id='help-nav' or @id='support-nav' or @id='help-button' or @id='support-button' or @id='help-menu-item' or @id='support-menu-item' or @id='help-navigation']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='help-category-payments' or @id='category-payments' or @id='help-payments-category' or @id='payments-category' or @id='help-category-payments-btn' or @id='category-payments-btn' or @id='help-payments-filter' or @id='payments-filter' or @id='help-payments-category-button' or @id='payments-category-button' or normalize-space()='Payments']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='help-category-payments' or @id='category-payments' or @id='help-payments-category' or @id='payments-category' or @id='help-category-payments-btn' or @id='category-payments-btn' or @id='help-payments-filter' or @id='payments-filter' or @id='help-payments-category-button' or @id='payments-category-button' or normalize-space()='Payments']",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "HELP_CATEGORY_SELECTED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8003/?seed=1",
                "prompt": "Expand the refund FAQ.",
                "actions": [
                    {
                        "url": "http://localhost:8003/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8003/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='nav-help' or @id='help-link' or @id='support-link' or @id='help-nav' or @id='support-nav' or @id='help-button' or @id='support-button' or @id='help-menu-item' or @id='support-menu-item' or @id='help-navigation']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='nav-help' or @id='help-link' or @id='support-link' or @id='help-nav' or @id='support-nav' or @id='help-button' or @id='support-button' or @id='help-menu-item' or @id='support-menu-item' or @id='help-navigation']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='faq-item-4' or @id='faq-tile-4' or @id='faq-card-4' or @id='faq-item-0' or @id='faq-tile-0' or @id='faq-card-0']//button | //button[.//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'refund')] or .//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'cancellation policy')]])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//*[@id='faq-item-4' or @id='faq-tile-4' or @id='faq-card-4' or @id='faq-item-0' or @id='faq-tile-0' or @id='faq-card-0']//button | //button[.//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'refund')] or .//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'cancellation policy')]])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "HELP_FAQ_TOGGLED",
                "has_success": False,
            },
        ],
    },
]


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _project_keys(project_id: str) -> set[str]:
    raw = str(project_id or "").strip().lower()
    key = _norm_key(raw)
    out = {key} if key else set()

    # Accept forms like:
    # - p01_autocinema
    # - p01-autocinema
    # - p01 autocinema
    match = re.match(r"^p\d+[\s_-]+(.+)$", raw)
    if match:
        tail_key = _norm_key(match.group(1))
        if tail_key:
            out.add(tail_key)

    return {item for item in out if item}


def _compact_trajectory(trajectory: Dict[str, Any], *, max_actions: int = 8) -> Dict[str, Any]:
    actions = trajectory.get("actions") if isinstance(trajectory.get("actions"), list) else []
    compact_actions: List[Dict[str, Any]] = []
    for action in actions[: max(1, int(max_actions))]:
        if not isinstance(action, dict):
            continue
        selector = action.get("selector") if isinstance(action.get("selector"), dict) else {}
        compact_actions.append(
            {
                "type": str(action.get("type") or ""),
                "selector": {
                    "type": str(selector.get("type") or ""),
                    "attribute": str(selector.get("attribute") or ""),
                    "value": str(selector.get("value") or "")[:120],
                },
                "text": str(action.get("text") or "")[:120],
            }
        )
    return {
        "prompt": str(trajectory.get("prompt") or "")[:320],
        "use_case": str(trajectory.get("use_case") or "")[:120],
        "has_success": bool(trajectory.get("has_success")),
        "actions": compact_actions,
    }


def get_trajectory_examples(
    *,
    web_project_id: str = "",
    use_case: str = "",
    prompt: str = "",
    limit: int = 2,
) -> List[Dict[str, Any]]:
    max_items = max(1, int(limit))
    wanted_project_keys = _project_keys(web_project_id)
    wanted_use_case = str(use_case or "").strip().lower()
    prompt_terms = {t for t in re.findall(r"[a-zA-Z0-9_]{4,}", str(prompt or "").lower())}

    ranked: List[tuple[int, Dict[str, Any]]] = []
    for project in TRAJECTORIES:
        if not isinstance(project, dict):
            continue
        project_id = str(project.get("project_id") or "")
        project_keys = _project_keys(project_id)
        project_match = bool(wanted_project_keys and wanted_project_keys.intersection(project_keys))
        if wanted_project_keys and not project_match:
            continue
        for trajectory in project.get("trajectories") if isinstance(project.get("trajectories"), list) else []:
            if not isinstance(trajectory, dict):
                continue
            score = 0
            if project_match:
                score += 5
            tr_use_case = str(trajectory.get("use_case") or "").strip().lower()
            if wanted_use_case and tr_use_case == wanted_use_case:
                score += 3
            tr_prompt = str(trajectory.get("prompt") or "").lower()
            if prompt_terms and tr_prompt:
                score += len(prompt_terms.intersection(set(re.findall(r"[a-zA-Z0-9_]{4,}", tr_prompt))))
            ranked.append((score, _compact_trajectory(trajectory)))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        fallback: List[Dict[str, Any]] = []
        for project in TRAJECTORIES:
            for trajectory in project.get("trajectories") if isinstance(project.get("trajectories"), list) else []:
                if isinstance(trajectory, dict):
                    fallback.append(_compact_trajectory(trajectory))
                if len(fallback) >= max_items:
                    return fallback
        return fallback
    return [row[1] for row in ranked[:max_items]]


def _find_best_trajectory(
    *,
    web_project_id: str = "",
    use_case: str = "",
    prompt: str = "",
) -> Dict[str, Any]:
    wanted_project_keys = _project_keys(web_project_id)
    wanted_use_case = str(use_case or "").strip().lower()
    prompt_terms = {t for t in re.findall(r"[a-zA-Z0-9_]{4,}", str(prompt or "").lower())}
    best_score = -1
    best_out: Dict[str, Any] = {}

    for project in TRAJECTORIES:
        if not isinstance(project, dict):
            continue
        project_id = str(project.get("project_id") or "")
        project_keys = _project_keys(project_id)
        project_match = bool(wanted_project_keys and wanted_project_keys.intersection(project_keys))
        if wanted_project_keys and not project_match:
            continue
        for trajectory in project.get("trajectories") if isinstance(project.get("trajectories"), list) else []:
            if not isinstance(trajectory, dict):
                continue
            score = 0
            tr_use_case = str(trajectory.get("use_case") or "").strip().lower()
            tr_prompt = str(trajectory.get("prompt") or "").lower()
            if project_match:
                score += 5
            if wanted_use_case and tr_use_case == wanted_use_case:
                score += 4
            if prompt_terms and tr_prompt:
                score += len(prompt_terms.intersection(set(re.findall(r"[a-zA-Z0-9_]{4,}", tr_prompt))))
            actions = trajectory.get("actions") if isinstance(trajectory.get("actions"), list) else []
            if score > best_score and actions:
                best_score = score
                best_out = {
                    "project_id": project_id,
                    "url": str(trajectory.get("url") or ""),
                    "prompt": str(trajectory.get("prompt") or ""),
                    "use_case": str(trajectory.get("use_case") or ""),
                    "has_success": bool(trajectory.get("has_success")),
                    "actions": [dict(a) for a in actions if isinstance(a, dict)],
                }

    return best_out


def _extract_prompt_value(prompt: str, patterns: List[str]) -> str:
    text = str(prompt or "")
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = str(match.group(1) or "").strip()
            if value:
                return value[:120]
    return ""


def _default_comment_text(forbidden_terms: List[str]) -> str:
    candidates = ["Great movie and solid pacing.", "Really enjoyed this film.", "Excellent direction and cast."]
    blocked = [str(x or "").strip().lower() for x in forbidden_terms if str(x or "").strip()]
    for candidate in candidates:
        lowered = candidate.lower()
        if all(term not in lowered for term in blocked):
            return candidate
    return "Nice movie."


def _is_placeholder_token(value: str) -> bool:
    return bool(re.fullmatch(r"\s*<[^>]+>\s*", str(value or "")))


def _is_unusable_query_candidate(value: str) -> bool:
    cleaned = str(value or "").strip()
    if not cleaned:
        return True
    if len(cleaned) < 3:
        return True
    if _is_placeholder_token(cleaned):
        return True
    if len(cleaned) > 80:
        return True
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", cleaned):
        return True
    return False


def _extract_search_query(prompt: str) -> str:
    forbidden_values = [
        str(v).strip()
        for v in re.findall(
            r"(?:\b[a-z_]+\b)[^'\"]{0,80}(?:not_equals|!=|does\s+not\s+contain|not_contains|not\s+contain|\bis\s+not\b)\s*['\"]([^'\"]+)['\"]",
            str(prompt or ""),
            flags=re.I,
        )
        if str(v).strip()
    ]
    forbidden_norm = {_norm_key(v) for v in forbidden_values}

    forbidden_query = _extract_prompt_value(
        prompt,
        [
            r"query\s*(?:not_equals|!=)\s*['\"]([^'\"]+)['\"]",
            r"(?:title|name|book_name)\s*(?:not_equals|!=)\s*['\"]([^'\"]+)['\"]",
            r"(?:title|name|book_name)\s*(?:is\s+)?not\s*['\"]([^'\"]+)['\"]",
            r"(?:title|name|book_name)\s*(?:that\s+is\s+)?not\s*['\"]([^'\"]+)['\"]",
            r"(?:title|name|book_name)[^'\"]{0,60}(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"author\s*(?:not_equals|!=)\s*['\"]([^'\"]+)['\"]",
        ],
    )

    patterns = [
        r"(?:search\s+for|find)\s+(?:the\s+)?(?:movie|film|book)\s*['\"]([^'\"]+)['\"]",
        r"(?:movie|film|book)(?:_name)?[^'\"]{0,40}(?:equals|contains|is)\s*['\"]([^'\"]+)['\"]",
        r"(?:title|name)\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
        r"query\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
    ]
    extracted = _extract_prompt_value(prompt, patterns)
    if extracted and not _is_unusable_query_candidate(extracted) and (
        not forbidden_query or _norm_key(extracted) != _norm_key(forbidden_query)
    ) and _norm_key(extracted) not in forbidden_norm:
        return extracted

    generic = re.findall(r"['\"]([^'\"]+)['\"]", str(prompt or ""))
    if generic:
        for generic_value in generic:
            normalized = str(generic_value).strip()[:120]
            if _is_unusable_query_candidate(normalized):
                continue
            if forbidden_query and _norm_key(normalized) == _norm_key(forbidden_query):
                continue
            if _norm_key(normalized) in forbidden_norm:
                continue
            return normalized
    return ""


def _extract_prompt_year(prompt: str) -> str:
    extracted = _extract_prompt_value(
        prompt,
        [
            r"year\s*(?:equals|=|is|:|greater_equal|greater_than|less_than|less_equal|not_equals)\s*['\"]?([12][0-9]{3})['\"]?",
            r"from\s+(?:the\s+)?([12][0-9]{3})",
        ],
    )
    if extracted and re.fullmatch(r"[12][0-9]{3}", extracted):
        return extracted
    inline_match = re.search(r"year[^0-9]{0,24}([12][0-9]{3})", str(prompt or ""), flags=re.I)
    if inline_match:
        return str(inline_match.group(1))
    return ""


def _extract_prompt_author(prompt: str) -> str:
    return _extract_prompt_value(
        prompt,
        [
            r"(?:author|director)\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"directed\s+by\s*['\"]([^'\"]+)['\"]",
            r"written\s+by\s*['\"]([^'\"]+)['\"]",
        ],
    )


def _extract_prompt_genre(prompt: str) -> str:
    return _extract_prompt_value(
        prompt,
        [
            r"(?:genre|genres)\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"(?:a|an)\s+['\"]([^'\"]+)['\"]\s+(?:movie|film|book)",
        ],
    )


def _extract_prompt_rating(prompt: str) -> str:
    return _extract_prompt_value(
        prompt,
        [
            r"rating\s*(?:equals|=|is|:|greater_equal|greater_than|less_than|less_equal)\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?",
        ],
    )


def _extract_prompt_pages(prompt: str) -> str:
    return _extract_prompt_value(
        prompt,
        [
            r"(?:page_count|pages|duration)\s*(?:equals|=|is|:|greater_equal|greater_than|less_than|less_equal)\s*['\"]?([0-9]+)['\"]?",
        ],
    )


def _extract_prompt_total_amount(prompt: str) -> float | None:
    extracted = _extract_prompt_value(
        prompt,
        [
            r"total[_ ]amount\s*(?:equals|=|is|:|of|greater_equal|greater_than|less_than|less_equal)\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?",
            r"amount\s*(?:equals|=|is|:|of|greater_equal|greater_than|less_than|less_equal)\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?",
            r"total\s*(?:of|is)\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?",
        ],
    )
    if not extracted:
        return None
    try:
        return float(extracted)
    except ValueError:
        return None


def _extract_prompt_category(prompt: str) -> str:
    return _extract_prompt_value(
        prompt,
        [
            r"category\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"in\s+the\s+category\s+that\s+equals\s*['\"]([^'\"]+)['\"]",
            r"category\s+that\s+contains\s*['\"]([^'\"]+)['\"]",
        ],
    )


def _extract_prompt_forbidden_category(prompt: str) -> str:
    return _extract_prompt_value(
        prompt,
        [
            r"category\s*(?:not_equals|!=)\s*['\"]([^'\"]+)['\"]",
            r"category\s*(?:is\s+)?not\s*['\"]([^'\"]+)['\"]",
            r"category[^'\"]{0,60}(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )


def _extract_prompt_brand_hint(prompt: str) -> str:
    return _extract_prompt_value(
        prompt,
        [
            r"brand\s*(?:contains|equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
        ],
    )


def _extract_prompt_forbidden_brand(prompt: str) -> str:
    return _extract_prompt_value(
        prompt,
        [
            r"brand\s*(?:not_equals|!=)\s*['\"]([^'\"]+)['\"]",
            r"brand\s*(?:is\s+)?not\s*['\"]([^'\"]+)['\"]",
            r"brand[^'\"]{0,60}(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )


def _extract_prompt_forbidden_carousel_title(prompt: str) -> str:
    return _extract_prompt_value(
        prompt,
        [
            r"title\s*(?:not_equals|!=)\s*['\"]([^'\"]+)['\"]",
            r"title\s*(?:is\s+)?not\s*['\"]([^'\"]+)['\"]",
        ],
    )


def _extract_carousel_direction(prompt: str) -> str:
    text = str(prompt or "")
    not_match = re.search(
        r"direction\s*(?:is\s*)?(?:not_equals|!=|not)\s*['\"](left|right)['\"]",
        text,
        flags=re.I,
    )
    if not_match:
        blocked = str(not_match.group(1) or "").strip().upper()
        return "LEFT" if blocked == "RIGHT" else "RIGHT"

    eq_match = re.search(
        r"direction\s*(?:equals|=|is|:)\s*['\"](left|right)['\"]",
        text,
        flags=re.I,
    )
    if eq_match:
        return str(eq_match.group(1) or "RIGHT").strip().upper()

    if re.search(r"\b(scroll|navigate)\s+left\b", text, flags=re.I):
        return "LEFT"
    return "RIGHT"


def _apply_prompt_overrides(actions: List[Dict[str, Any]], prompt: str, use_case: str = "") -> List[Dict[str, Any]]:
    def _resolved_value(raw_value: str, fallback: str) -> str:
        cleaned = str(raw_value or "").strip()
        if not cleaned or _is_placeholder_token(cleaned):
            return fallback
        return cleaned

    def _pick_text(
        *,
        equals_value: str,
        contains_value: str,
        forbidden_value: str,
        default_value: str,
        alternatives: List[str],
    ) -> str:
        candidate = equals_value or contains_value or default_value
        blocked = str(forbidden_value or "").strip().lower()
        if blocked and blocked in str(candidate).lower():
            candidate = ""
        if candidate:
            return candidate
        for item in alternatives:
            if not blocked or blocked not in item.lower():
                return item
        return default_value

    commenter_name = _extract_prompt_value(
        prompt,
        [
            r"(?:by\s+(?:the\s+)?)?commenter[_ ]name\s*(?:equals|=|is|:)?\s*['\"]([^'\"]+)['\"]",
            r"name\s+['\"]([^'\"]+)['\"]",
        ],
    )
    username_raw = _extract_prompt_value(prompt, [r"username\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
    email_raw = _extract_prompt_value(prompt, [r"email\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
    password_raw = _extract_prompt_value(prompt, [r"password\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
    username = _resolved_value(username_raw, "user1")
    password = _resolved_value(password_raw, "Passw0rd!")
    email = _resolved_value(email_raw, f"{username}@gmail.com")
    if "@" not in email:
        email = f"{username}@gmail.com"

    signup_username = _resolved_value(username_raw, "newuser1")
    signup_email = _resolved_value(email_raw, f"{signup_username}@gmail.com")
    if signup_email.startswith("@"):
        signup_email = f"{signup_username}{signup_email}"
    if "@" not in signup_email:
        signup_email = f"{signup_username}@gmail.com"
    signup_password = _resolved_value(password_raw, "Passw0rd!")

    contact_name_equals = _extract_prompt_value(
        prompt,
        [
            r"\bname\s*(?:that\s+)?(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"\bname\s+that\s+equals\s*['\"]([^'\"]+)['\"]",
        ],
    )
    contact_name_forbidden = _extract_prompt_value(
        prompt,
        [
            r"\bname\s*(?:not_equals|!=|not_contains|not\s+contain|does\s+not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"\bname\s*(?:that\s+is\s+|is\s+)?not\s*['\"]([^'\"]+)['\"]",
            r"\bname\s+different\s+to\s*['\"]([^'\"]+)['\"]",
        ],
    )
    contact_message_equals = _extract_prompt_value(
        prompt,
        [
            r"\bmessage\s*(?:that\s+)?(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"\bmessage\s+that\s+equals\s*['\"]([^'\"]+)['\"]",
        ],
    )
    contact_subject_equals = _extract_prompt_value(prompt, [r"\bsubject\s*(?:that\s+)?(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
    contact_email_equals = _extract_prompt_value(
        prompt,
        [
            r"\bemail\s*(?:that\s+)?(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"\bemail\s+(?:different\s+to|different\s+from)\s*['\"]([^'\"]+)['\"]",
        ],
    )

    contact_name = _pick_text(
        equals_value=contact_name_equals,
        contains_value="",
        forbidden_value=contact_name_forbidden,
        default_value="John Smith",
        alternatives=["Marta", "Lucas", "Robin", "Noah"],
    )
    contact_message = contact_message_equals or "Great website, thanks!"
    contact_subject = contact_subject_equals or "Feedback"
    contact_email = contact_email_equals or "john@example.com"
    if "@" not in contact_email:
        contact_email = f"{signup_username}@gmail.com"

    explicit_content = _extract_prompt_value(
        prompt,
        [r"content\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]", r"comment(?:_message)?\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"],
    )
    forbidden_content_terms = re.findall(
        r"(?:content|comment(?:_message)?|message)[^'\"]{0,60}(?:does\s+not\s+contain|not\s+contain|without)[^'\"]*['\"]([^'\"]+)['\"]",
        str(prompt or ""),
        flags=re.I,
    )
    content_text = explicit_content or _default_comment_text([str(x) for x in forbidden_content_terms if str(x).strip()])
    search_query = _extract_search_query(prompt)
    prompt_year = _extract_prompt_year(prompt)
    prompt_author = _extract_prompt_author(prompt)
    prompt_genre = _extract_prompt_genre(prompt)
    prompt_rating = _extract_prompt_rating(prompt)
    prompt_pages = _extract_prompt_pages(prompt)
    prompt_category = _extract_prompt_category(prompt)
    prompt_brand_hint = _extract_prompt_brand_hint(prompt)
    prompt_forbidden_category = _extract_prompt_forbidden_category(prompt)
    prompt_forbidden_brand = _extract_prompt_forbidden_brand(prompt)
    prompt_total_amount = _extract_prompt_total_amount(prompt)
    prompt_forbidden_carousel_title = _extract_prompt_forbidden_carousel_title(prompt)
    carousel_direction = _extract_carousel_direction(prompt)
    normalized_use_case = str(use_case or "").strip().upper()
    autozone_query_hint = search_query or ""

    def _norm_text(value: str) -> str:
        return str(value or "").strip().lower()

    def _query_for_total_amount(total_amount: float | None) -> str:
        if total_amount is None:
            return ""
        if abs(total_amount - 149.95) < 0.01:
            return "Compact Ice Maker"
        if total_amount >= 500:
            return "Premium Drone"
        if total_amount >= 349:
            return "Compact Digital Camera"
        if total_amount >= 229.99:
            return "Portable Projector"
        if total_amount >= 199.99:
            return "Premium Coffee Machine"
        if total_amount >= 149.95:
            return "Compact Ice Maker"
        return ""

    known_product_queries = [
        ("Compact Ice Maker", "kitchen", "frostflow"),
        ("Portable Projector", "electronics", "beamlite"),
        ("Premium Coffee Machine", "kitchen", "brewtech"),
        ("Smart Jump Rope", "fitness", "cardiopulse"),
        ("Premium Drone", "technology", "skycam"),
        ("Air Purifier", "home", "purebreeze"),
        ("Memory Foam Seat Cushion", "home", "comfortease"),
    ]

    def _pick_safe_product_query() -> str:
        blocked_category = _norm_text(prompt_forbidden_category)
        blocked_brand = _norm_text(prompt_forbidden_brand)
        for query, category, brand in known_product_queries:
            if blocked_category and blocked_category == category:
                continue
            if blocked_brand and blocked_brand == brand:
                continue
            return query
        return "Compact Ice Maker"

    if normalized_use_case in {"PROCEED_TO_CHECKOUT", "CHECKOUT_STARTED"} and prompt_total_amount is not None:
        amount_query = _query_for_total_amount(prompt_total_amount)
        if amount_query:
            autozone_query_hint = amount_query

    if normalized_use_case in {"SHARE_PRODUCT", "VIEW_DETAIL"} and not autozone_query_hint:
        autozone_query_hint = _pick_safe_product_query()

    if normalized_use_case == "ADD_TO_WISHLIST":
        token = _norm_text(prompt_brand_hint)
        brand_candidates = [
            "ComfortEase",
            "TechWear",
            "PowerTech",
            "StoragePro",
            "TypeWave",
            "ChefMaster",
        ]
        if token:
            for candidate in brand_candidates:
                if token in candidate.lower():
                    autozone_query_hint = candidate
                    break
        if not autozone_query_hint:
            autozone_query_hint = "ComfortEase"

    if normalized_use_case == "DETAILS_TOGGLE" and not autozone_query_hint and prompt_category:
        category_token = _norm_text(prompt_category)
        category_candidates = ["home", "kitchen", "technology", "electronics", "fitness"]
        for category in category_candidates:
            if category_token and category_token in category:
                autozone_query_hint = category.title()
                break

    if not autozone_query_hint:
        if prompt_category and not _is_unusable_query_candidate(prompt_category):
            autozone_query_hint = prompt_category
        elif prompt_brand_hint and not _is_unusable_query_candidate(prompt_brand_hint):
            autozone_query_hint = prompt_brand_hint

    autozone_search_input_xpath = (
        "//input[@id='type-to-search' or @id='search-input' or @id='query-box' or @id='filter-input' "
        "or @id='product-search' or @id='item-search' or @id='search-field' or @id='lookup-input' "
        "or @id='find-input' or @id='search-box']"
    )
    autozone_carousel_left_xpath = (
        "//button[@id='carousel-left-arrow' and count(preceding::button[@id='carousel-left-arrow'])=1]/*[name()='svg']"
    )
    autozone_carousel_right_xpath = (
        "//button[@id='carousel-right-control' and count(preceding::button[@id='carousel-right-control'])=1]/*[name()='svg']"
    )
    autozone_add_cart_xpath = (
        "(//*[@id='add-cart-btn' or @id='cart-add' or @id='add-basket' or @id='add-to-basket' "
        "or @id='add-to-cart' or @id='add-to-cart-button' or @id='add-cart-button' "
        "or @id='cart-action' or @id='basket-action' or @id='add-item' or @id='cart-item-add' "
        "or @id='basket-add-item' or @id='add-product'])[1]"
    )
    autozone_quantity_xpath = (
        "//*[@id='qty-input' or @id='quantity-field' or @id='qty-field' or @id='amount-input' "
        "or @id='qty-box' or @id='quantity-box' or @id='qty-select' or @id='quantity-select' "
        "or @id='quantity-input' or @id='quantity-selector' or @id='item-qty' or @id='product-qty']"
    )
    autozone_buy_now_xpath = (
        "(//button[@id='buy-now' or @id='buy-now-button' or @id='buy-now-btn' or @id='order-now' "
        "or contains(translate(normalize-space(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'buy now')])[1]"
    )
    autozone_detail_wishlist_xpath = (
        "(//main//button[@id='wishlist-btn' or @id='add-wishlist' or @id='save-later' or @id='wishlist-add' "
        "or @id='favorite-btn' or @id='save-item' or @id='add-favorite' or @id='wishlist-action' "
        "or @id='save-product' or @id='favorite-action'])[1]"
    )
    autozone_details_toggle_xpath = (
        "(//main//button[@id='toggle-btn' or @id='toggle-button' or @id='switch-btn' or @id='toggle-control' "
        "or @id='toggle-action' or @id='switch-control' or @id='toggle-state' or @id='toggle-option' "
        "or @id='toggle-choice' or @id='toggle'])[1]"
    )

    def _set_xpath_selector(action_obj: Dict[str, Any], xpath_value: str) -> None:
        selector_payload = {
            "type": "xpathSelector",
            "value": str(xpath_value),
            "case_sensitive": False,
        }
        action_obj["selector"] = selector_payload
        attrs = action_obj.get("attributes")
        if isinstance(attrs, dict):
            attrs = dict(attrs)
            attrs["selector"] = selector_payload
            action_obj["attributes"] = attrs

    normalized_year_token = re.sub(r"[^0-9-]", "", str(prompt_year or "").strip())
    if len(normalized_year_token) < 4:
        normalized_year_token = "2020"
    year_key_values = list(normalized_year_token[:4])
    first_name_equals = _extract_prompt_value(prompt, [r"first[_ ]name\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
    first_name_contains = _extract_prompt_value(prompt, [r"first[_ ]name\s*(?:contains)\s*['\"]([^'\"]+)['\"]"])
    first_name_forbidden = _extract_prompt_value(
        prompt,
        [r"first[_ ]name\s*(?:not_contains|not\s+contain|does\s+not\s+contain|not_equals|!=)\s*['\"]([^'\"]+)['\"]"],
    )
    last_name_equals = _extract_prompt_value(prompt, [r"last[_ ]name\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
    last_name_contains = _extract_prompt_value(prompt, [r"last[_ ]name\s*(?:contains)\s*['\"]([^'\"]+)['\"]"])
    last_name_forbidden = _extract_prompt_value(
        prompt,
        [r"last[_ ]name\s*(?:not_contains|not\s+contain|does\s+not\s+contain|not_equals|!=)\s*['\"]([^'\"]+)['\"]"],
    )
    website_equals = _extract_prompt_value(prompt, [r"website\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
    website_contains = _extract_prompt_value(prompt, [r"website\s*(?:contains)\s*['\"]([^'\"]+)['\"]"])
    website_forbidden = _extract_prompt_value(
        prompt,
        [r"website\s*(?:not_contains|not\s+contain|does\s+not\s+contain|not_equals|!=)\s*['\"]([^'\"]+)['\"]"],
    )
    first_name_value = _pick_text(
        equals_value=first_name_equals,
        contains_value=first_name_contains,
        forbidden_value=first_name_forbidden,
        default_value="Alex",
        alternatives=["Noah", "Mia", "Jules", "Reader"],
    )
    last_name_value = _pick_text(
        equals_value=last_name_equals,
        contains_value=last_name_contains,
        forbidden_value=last_name_forbidden,
        default_value="Stone",
        alternatives=["Miller", "Brooks", "Carter", "Rivers"],
    )
    website_value = _pick_text(
        equals_value=website_equals,
        contains_value=(f"{website_contains}.example.com" if website_contains else ""),
        forbidden_value=website_forbidden,
        default_value="bookshelf.example.com",
        alternatives=["readerhub.com", "librarylane.com", "bookspace.io"],
    )
    year_select_xpath = "//*[@id='library']//select[.//option[contains(translate(normalize-space(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'all years')]]"
    genre_select_xpath = "//*[@id='library']//select[.//option[contains(translate(normalize-space(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'all genres')]]"
    year_option_xpath = (
        f"({year_select_xpath}/option[@value='{prompt_year}'] | {year_select_xpath}/option[2])[1]"
        if prompt_year
        else f"{year_select_xpath}/option[2]"
    )
    genre_option_xpath = (
        f"({genre_select_xpath}/option[@value='{prompt_genre}' or normalize-space()='{prompt_genre}'] | {genre_select_xpath}/option[2])[1]"
        if prompt_genre
        else f"{genre_select_xpath}/option[2]"
    )

    def _xpath_literal(value: str) -> str:
        text = str(value or "")
        if "'" not in text:
            return f"'{text}'"
        if '"' not in text:
            return f'"{text}"'
        parts = text.split("'")
        pieces: List[str] = []
        for idx, part in enumerate(parts):
            if part:
                pieces.append(f"'{part}'")
            if idx < len(parts) - 1:
                pieces.append('"\'"')
        return "concat(" + ", ".join(pieces) + ")"

    def _make_nav(url: str) -> Dict[str, Any]:
        return {"type": "NavigateAction", "url": str(url)}

    def _make_click(xpath: str) -> Dict[str, Any]:
        return {
            "type": "ClickAction",
            "selector": {
                "type": "xpathSelector",
                "value": str(xpath),
                "case_sensitive": False,
            },
        }

    def _make_type(xpath: str, text: str) -> Dict[str, Any]:
        return {
            "type": "TypeAction",
            "text": str(text),
            "selector": {
                "type": "xpathSelector",
                "value": str(xpath),
                "case_sensitive": False,
            },
        }

    def _make_select(xpath: str, value: str) -> Dict[str, Any]:
        return {
            "type": "SelectAction",
            "value": str(value),
            "selector": {
                "type": "xpathSelector",
                "value": str(xpath),
                "case_sensitive": False,
            },
        }

    def _make_send_keys(key: str) -> Dict[str, Any]:
        return {
            "type": "SendKeysAction",
            "keys": [str(key)],
        }

    prompt_time_equals = _extract_prompt_value(
        prompt,
        [
            r"time\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"\bat\s*['\"]([0-9]{1,2}:[0-9]{2}\s*[AP]M)['\"]",
        ],
    )
    prompt_time_not_equals = _extract_prompt_value(
        prompt,
        [
            r"time\s*(?:is\s+)?(?:not_equals|!=|not\s+equal(?:s)?(?:\s+to)?)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    prompt_people_equals = _extract_prompt_value(
        prompt,
        [
            r"people\s*(?:equals|=|is)\s*['\"]?([0-9]+)['\"]?",
            r"for\s*['\"]?([0-9]+)['\"]?\s*people?",
        ],
    )
    prompt_people_greater_equal = _extract_prompt_value(
        prompt,
        [
            r"people\s*(?:greater than or equal to|greater_equal|greater equal|>=)\s*['\"]?([0-9]+)['\"]?",
        ],
    )
    prompt_date_iso = _extract_prompt_value(
        prompt,
        [
            r"(?:date\s*(?:equals|=|is|:|greater_equal|greater than or equal to|less_equal|less than or equal to)[^0-9]{0,20})([0-9]{4}-[0-9]{2}-[0-9]{2})",
            r"on\s*['\"]([0-9]{4}-[0-9]{2}-[0-9]{2})",
            r"([0-9]{4}-[0-9]{2}-[0-9]{2})T",
        ],
    )
    prompt_country_code = _extract_prompt_value(
        prompt,
        [
            r"country\s+with\s+code\s*['\"]([A-Za-z]{2})['\"]",
            r"code\s*(?:equals|=|is|:)\s*['\"]([A-Za-z]{2})['\"]",
        ],
    )
    prompt_occasion_equals = _extract_prompt_value(
        prompt,
        [
            r"occasion\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"for\s+a[n]?\s*['\"]([^'\"]+)['\"]",
        ],
    )
    prompt_occasion_not_equals = _extract_prompt_value(
        prompt,
        [
            r"occasion\s*(?:is\s+)?(?:not_equals|!=|not\s+equal(?:s)?(?:\s+to)?)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    prompt_restaurant_name = _extract_prompt_value(
        prompt,
        [
            r"name\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"details\s+for\s+['\"]([^'\"]+)['\"]",
        ],
    )
    prompt_help_category_equals = _extract_prompt_value(
        prompt,
        [
            r"category\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"select\s+the\s+([^'\"]+?)\s+category",
        ],
    )
    prompt_help_category_not_equals = _extract_prompt_value(
        prompt,
        [
            r"category\s*(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
            r"category[^'\"]{0,120}\bis\s+not\s*['\"]([^'\"]+)['\"]",
            r"category[^'\"]{0,120}\bnot\s*['\"]([^'\"]+)['\"]",
        ],
    )
    prompt_help_category_not_contains = _extract_prompt_value(
        prompt,
        [
            r"category\s*(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    prompt_faq_question = _extract_prompt_value(
        prompt,
        [
            r"question\s*(?:contains|equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"faq\s+item\s+where\s+the\s+question\s+contains\s*['\"]([^'\"]+)['\"]",
        ],
    )
    prompt_scroll_section = _extract_prompt_value(
        prompt,
        [
            r"section\s*(?:contains|equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    prompt_scroll_section_not_contains = _extract_prompt_value(
        prompt,
        [
            r"section\s*(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    prompt_feature_value = _extract_prompt_value(
        prompt,
        [
            r"feature\s*(?:contains|equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"click\s+the\s+([^'\"]+?)\s+feature",
        ],
    )
    prompt_feature_not_value = _extract_prompt_value(
        prompt,
        [
            r"feature\s*(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
            r"feature[^'\"]{0,120}\bis\s+not\s*['\"]([^'\"]+)['\"]",
            r"feature[^'\"]{0,120}\bnot\s*['\"]([^'\"]+)['\"]",
        ],
    )
    prompt_card_type = _extract_prompt_value(
        prompt,
        [
            r"card_type\s*(?:contains|equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"click\s+the\s+([^'\"]+?)\s+contact\s+card",
        ],
    )
    prompt_card_type_not = _extract_prompt_value(
        prompt,
        [
            r"card_type\s*(?:not_equals|!=|does\s+not\s+contain|not_contains)\s*['\"]([^'\"]+)['\"]",
            r"contact\s+card[^'\"]{0,120}\bnot\b[^'\"]*['\"]([^'\"]+)['\"]",
            r"card[^'\"]{0,120}\bis\s+not\s*['\"]([^'\"]+)['\"]",
        ],
    )
    contact_email_contains = _extract_prompt_value(prompt, [r"email\s*(?:contains)\s*['\"]([^'\"]+)['\"]"])
    contact_subject_contains = _extract_prompt_value(prompt, [r"subject\s*(?:contains)\s*['\"]([^'\"]+)['\"]"])
    prompt_country_code_not_equals = _extract_prompt_value(
        prompt,
        [
            r"code\s*(?:not_equals|!=|is\s+not)\s*['\"]([A-Za-z]{2})['\"]",
        ],
    )

    if contact_email_contains:
        email_hint = str(contact_email_contains).strip()
        if "@" in email_hint and "." in email_hint:
            contact_email = email_hint
        elif "@" in email_hint:
            contact_email = f"{email_hint}e.com"
        else:
            contact_email = f"{email_hint}@example.com"
    if contact_subject_contains:
        subject_hint = str(contact_subject_contains).strip()
        if subject_hint and subject_hint.lower() not in contact_subject.lower():
            contact_subject = f"Re: {subject_hint}"

    autodining_time_options = ["12:00 PM", "12:30 PM", "1:00 PM", "1:30 PM", "2:00 PM", "2:30 PM"]
    autodining_time = prompt_time_equals.strip() if prompt_time_equals else "2:00 PM"
    if not prompt_time_equals and prompt_time_not_equals:
        blocked = str(prompt_time_not_equals).strip().lower()
        fallback = next((t for t in autodining_time_options if t.lower() != blocked), "2:00 PM")
        autodining_time = fallback
    if autodining_time not in autodining_time_options:
        autodining_time = "2:00 PM"

    autodining_people = 2
    if prompt_people_equals.isdigit():
        autodining_people = max(1, min(8, int(prompt_people_equals)))
    elif prompt_people_greater_equal.isdigit():
        autodining_people = max(1, min(8, max(int(prompt_people_greater_equal), 8)))

    autodining_date = prompt_date_iso or "2026-04-03"
    autodining_country_code = (prompt_country_code or "US").upper()
    if not prompt_country_code and prompt_country_code_not_equals:
        blocked_code = str(prompt_country_code_not_equals).strip().upper()
        for code in ("US", "IN", "VN", "AR", "FR"):
            if code != blocked_code:
                autodining_country_code = code
                break
    if len(autodining_country_code) != 2:
        autodining_country_code = "US"

    autodining_occasion = (prompt_occasion_equals or "").strip().lower()
    if not autodining_occasion:
        blocked = str(prompt_occasion_not_equals or "").strip().lower()
        for item in ("anniversary", "birthday", "business", "other"):
            if not blocked or item != blocked:
                autodining_occasion = item
                break
    if not autodining_occasion:
        autodining_occasion = "anniversary"

    autodining_search_hint = _extract_prompt_value(
        prompt,
        [
            r"query\s*(?:contains|equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"search\s+for\s+(?:restaurants?\s+where\s+)?['\"]([^'\"]+)['\"]",
            r"details\s+for\s+['\"]([^'\"]+)['\"]",
        ],
    )
    if not autodining_search_hint:
        autodining_search_hint = search_query

    prompt_lower = str(prompt or "").lower()

    def _pick_restaurant_id(default_id: str = "11") -> str:
        name_map = {
            "le jardin": "1",
            "bella vista": "2",
            "sakura sushi": "3",
            "the steakhouse": "4",
            "thai garden": "7",
            "seoul kitchen": "9",
            "pho saigon": "11",
            "the noodle house": "18",
        }
        if prompt_restaurant_name:
            mapped = name_map.get(str(prompt_restaurant_name).strip().lower())
            if mapped:
                return mapped
        if "korean" in prompt_lower or "227" in prompt_lower:
            return "9"
        if "thai garden" in prompt_lower or "thai" in prompt_lower:
            return "7"
        if "italian" in prompt_lower:
            return "2"
        if "asian fusion" in prompt_lower:
            return "18"
        if "vietnamese" in prompt_lower:
            return "11"
        return default_id

    autodining_restaurant_id = _pick_restaurant_id()
    autodining_time_segment = autodining_time.replace(":", "%3A").replace(" ", "%20")
    autodining_booking_url = (
        f"http://localhost:8003/booking/{autodining_restaurant_id}/{autodining_time_segment}"
        f"?seed=1&people={autodining_people}&date={autodining_date}"
    )

    autodining_home_url = "http://localhost:8003/?seed=1"
    autodining_search_xpath = (
        "//*[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' "
        "or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']"
    )
    autodining_search_input_xpath = (
        "//input[@id='search-input' or @id='search-input-help' or @id='search-box' or @id='search-field' "
        "or @id='query-box' or @id='restaurant-search' or @id='search-restaurants' or @id='search-text']"
    )
    autodining_time_picker_xpath = (
        "//*[@id='time_picker' or @id='time-picker' or @id='time-selector' or @id='time-input' "
        "or @id='booking-time' or @id='reservation-time' or @id='time-trigger' or @id='checkin-time' or @id='time-field']"
    )
    autodining_people_picker_xpath = (
        "//*[@id='people_picker' or @id='people-picker' or @id='guest-picker' or @id='guests-picker' "
        "or @id='people-selector' or @id='guest-selector' or @id='booking-people' or @id='reservation-people' "
        "or @id='people-input' or @id='guests-input']"
    )
    autodining_nav_about_xpath = (
        "//*[@id='nav-about' or @id='about-link' or @id='about-us-link' or @id='about-nav' "
        "or @id='about-us-nav' or @id='about-button' or @id='about-us-button' or @id='about-menu-item' "
        "or @id='about-us-menu-item' or @id='about-navigation']"
    )
    autodining_nav_help_xpath = (
        "//*[@id='nav-help' or @id='help-link' or @id='support-link' or @id='help-nav' or @id='support-nav' "
        "or @id='help-button' or @id='support-button' or @id='help-menu-item' or @id='support-menu-item' "
        "or @id='help-navigation']"
    )
    autodining_nav_contact_xpath = (
        "//*[@id='nav-contact' or @id='contact-link' or @id='contact-us-link' or @id='contact-nav' "
        "or @id='contact-us-nav' or @id='contact-button' or @id='contact-us-button' or @id='contact-menu-item' "
        "or @id='contact-us-menu-item' or @id='contact-navigation']"
    )
    autodining_country_xpath = (
        "//*[@id='country-select' or @id='country-dropdown' or @id='country-picker' or @id='country-selector' "
        "or @id='country-choice' or @id='country-option' or @id='country-field' or @id='country-input' "
        "or @id='country-selection' or @id='country-picker-dropdown']"
    )
    autodining_occasion_xpath = (
        "//*[@id='occasion-select' or @id='occasion-dropdown' or @id='occasion-picker' or @id='occasion-selector' "
        "or @id='occasion-choice' or @id='occasion-option' or @id='occasion-field' or @id='occasion-input' "
        "or @id='occasion-selection' or @id='occasion-picker-dropdown']"
    )

    if normalized_use_case == "VIEW_RESTAURANT":
        return [_make_nav(f"http://localhost:8003/restaurant/{autodining_restaurant_id}?seed=1")]

    if normalized_use_case == "TIME_DROPDOWN_OPENED":
        time_literal = _xpath_literal(autodining_time)
        return [
            _make_nav(autodining_home_url),
            _make_click(autodining_time_picker_xpath),
            _make_click(f"(//button[normalize-space()={time_literal}])[1]"),
        ]

    if normalized_use_case == "PEOPLE_DROPDOWN_OPENED":
        return [
            _make_nav(autodining_home_url),
            _make_click(autodining_people_picker_xpath),
            _make_click(
                f"(//button[contains(normalize-space(), '{autodining_people}') "
                "and (contains(normalize-space(), 'Guest') or contains(normalize-space(), 'Guests'))])[1]"
            ),
        ]

    if normalized_use_case == "SEARCH_RESTAURANT":
        query_text = autodining_search_hint or "Thai Garden"
        return [
            _make_nav(autodining_home_url),
            _make_click(autodining_search_xpath),
            _make_type(autodining_search_input_xpath, query_text),
            # SEARCH_RESTAURANT is debounced in UI (~500ms); these no-op key steps
            # give enough replay time for the event to be logged before final scoring.
            _make_send_keys("ArrowLeft"),
            _make_send_keys("ArrowRight"),
            _make_send_keys("ArrowLeft"),
        ]

    if normalized_use_case == "SCROLL_VIEW":
        section_hint = prompt_scroll_section or "Expensive"
        if not prompt_scroll_section and prompt_scroll_section_not_contains:
            blocked = str(prompt_scroll_section_not_contains).strip().lower()
            for title in ("Mid ticket", "Cheap", "Expensive"):
                if blocked not in title.lower():
                    section_hint = title
                    break
        section_literal = _xpath_literal(section_hint)
        right_xpath = (
            f"(//section[.//h2[contains(normalize-space(), {section_literal})]]"
            "//*[@data-testid='scroll-right-1' or starts-with(@data-testid,'scroll-right-') "
            "or @id='scroll-right-button' or contains(@id,'scroll-right')]"
            " | //*[@data-testid='scroll-right-1' or starts-with(@data-testid,'scroll-right-') "
            "or @id='scroll-right-button' or contains(@id,'scroll-right')])[1]"
        )
        left_xpath = (
            f"(//section[.//h2[contains(normalize-space(), {section_literal})]]"
            "//*[@data-testid='scroll-left-1' or starts-with(@data-testid,'scroll-left-') "
            "or @id='scroll-left-button' or contains(@id,'scroll-left')]"
            " | //*[@data-testid='scroll-left-1' or starts-with(@data-testid,'scroll-left-') "
            "or @id='scroll-left-button' or contains(@id,'scroll-left')])[1]"
        )
        return [
            _make_nav(autodining_home_url),
            _make_click(right_xpath),
            _make_click(left_xpath),
        ]

    if normalized_use_case == "BOOK_RESTAURANT":
        return [
            _make_nav(autodining_booking_url),
            _make_click(
                "//*[@id='full-name-input' or @id='name-input' or @id='full-name' or @id='booking-name' "
                "or @id='reservation-name' or @id='customer-name' or @id='fullname-input' or @id='name-field' "
                "or @id='guest-name' or @id='full-name-field']"
            ),
        ]

    if normalized_use_case == "COUNTRY_SELECTED":
        return [
            _make_nav(autodining_booking_url),
            _make_select(autodining_country_xpath, autodining_country_code),
        ]

    if normalized_use_case == "OCCASION_SELECTED":
        return [
            _make_nav(autodining_booking_url),
            _make_select(autodining_occasion_xpath, autodining_occasion),
        ]

    if normalized_use_case == "RESERVATION_COMPLETE":
        confirm_xpath = (
            "//*[@id='confirm_button' or @id='confirm-booking-button' or @id='complete-reservation-button' "
            "or @id='finalize-reservation-button' or @id='submit-reservation-button' or @id='reservation-confirm-button' "
            "or @id='finish-booking-button' or @id='complete-booking-button' or @id='confirm-reservation-button' "
            "or @id='reservation-submit-button']"
        )
        phone_xpath = (
            "//*[@id='phone-number-input' or @id='phone-input' or @id='booking-phone' or @id='reservation-phone' "
            "or @id='customer-phone' or @id='phone-field' or @id='mobile-input' or @id='contact-phone' "
            "or @id='phone-number' or @id='phone']"
        )
        phone_input_xpath = (
            "//input[@id='phone-number-input' or @id='phone-input' or @id='booking-phone' or @id='reservation-phone' "
            "or @id='customer-phone' or @id='phone-field' or @id='mobile-input' or @id='contact-phone' "
            "or @id='phone-number' or @id='phone']"
        )
        return [
            _make_nav(autodining_booking_url),
            _make_type(phone_input_xpath, "666777888"),
            _make_select(autodining_country_xpath, autodining_country_code),
            _make_select(autodining_occasion_xpath, autodining_occasion),
            _make_click(confirm_xpath),
        ]

    if normalized_use_case == "CONTACT_FORM_SUBMIT":
        return [
            _make_nav(autodining_home_url),
            _make_click(autodining_nav_contact_xpath),
            _make_type(
                "//input[@id='contact-name-input' or @id='name-input-contact' or @id='contact-name-field' "
                "or @id='name-field-contact' or @id='contact-name-text-input' or @id='name-text-input-contact' "
                "or @id='contact-name-entry' or @id='name-entry-contact' or @id='contact-name-textbox' "
                "or @id='name-textbox-contact']",
                contact_name,
            ),
            _make_type(
                "//input[@id='contact-email-input' or @id='email-input-contact' or @id='contact-email-field' "
                "or @id='email-field-contact' or @id='contact-email-text-input' or @id='email-text-input-contact' "
                "or @id='contact-email-entry' or @id='email-entry-contact' or @id='contact-email-textbox' "
                "or @id='email-textbox-contact']",
                contact_email,
            ),
            _make_type(
                "//input[@id='contact-subject-input' or @id='subject-input-contact' or @id='contact-subject-field' "
                "or @id='subject-field-contact' or @id='contact-subject-text-input' or @id='subject-text-input-contact' "
                "or @id='contact-subject-entry' or @id='subject-entry-contact' or @id='contact-subject-textbox' "
                "or @id='subject-textbox-contact']",
                contact_subject,
            ),
            _make_type(
                "//textarea[@id='contact-message-textarea' or @id='message-textarea-contact' or @id='contact-message-field' "
                "or @id='message-field-contact' or @id='contact-message-text-area' or @id='message-text-area-contact' "
                "or @id='contact-message-entry' or @id='message-entry-contact' or @id='contact-message-textbox' "
                "or @id='message-textbox-contact']",
                contact_message,
            ),
            _make_click(
                "//*[@id='send-message-button' or @id='send-message-btn' or @id='submit-contact-form' "
                "or @id='contact-submit-button' or @id='message-submit-button' or @id='send-contact-button' "
                "or @id='contact-send-button' or @id='submit-message-button' or @id='contact-form-submit' or @id='send-btn']"
            ),
        ]

    if normalized_use_case == "ABOUT_FEATURE_CLICK":
        feature_value = prompt_feature_value or ""
        if feature_value:
            feature_value_l = str(feature_value).strip().lower()
            if feature_value_l in {"feature", "about", "about page", "the about page"} or "that is not" in feature_value_l:
                feature_value = ""
        if not feature_value and prompt_feature_not_value:
            blocked = str(prompt_feature_not_value).strip().lower()
            for option in ("Easy Reservations", "Curated Restaurants", "Community Driven", "Verified Reviews", "Trending Spots"):
                if blocked not in option.lower():
                    feature_value = option
                    break
        if not feature_value:
            feature_value = "Trending Spots"
        feature_literal = _xpath_literal(feature_value)
        return [
            _make_nav(autodining_home_url),
            _make_click(autodining_nav_about_xpath),
            _make_click(f"(//h3[normalize-space()={feature_literal}]/parent::div)[1]"),
        ]

    if normalized_use_case == "CONTACT_CARD_CLICK":
        card_value = str(prompt_card_type or "").strip().lower()
        if card_value in {"card", "contact card", "type", "contact"} or "that is not" in card_value:
            card_value = ""
        if not card_value and prompt_card_type_not:
            blocked = str(prompt_card_type_not).strip().lower()
            for option in ("Visit Us", "Business Hours", "Email Us", "Call Us"):
                if blocked not in option.lower():
                    card_value = option.lower()
                    break
        if not card_value:
            card_value = "call us"
        card_map = {
            "email": "Email Us",
            "email us": "Email Us",
            "phone": "Call Us",
            "call": "Call Us",
            "call us": "Call Us",
            "chat": "Call Us",
            "office": "Visit Us",
            "visit": "Visit Us",
            "visit us": "Visit Us",
            "business": "Business Hours",
            "business hours": "Business Hours",
        }
        target_card = card_map.get(card_value, "Call Us")
        card_literal = _xpath_literal(target_card)
        return [
            _make_nav(autodining_home_url),
            _make_click(autodining_nav_contact_xpath),
            _make_click(f"(//a[.//h3[contains(normalize-space(), {card_literal})]])[1]"),
        ]

    if normalized_use_case == "HELP_CATEGORY_SELECTED":
        categories = ["Getting Started", "Bookings", "Payments", "Account", "Restaurants", "Technical"]
        category_target = prompt_help_category_equals.strip() if prompt_help_category_equals else ""
        if category_target and category_target.lower() in {"help", "the help", "help page", "help category"}:
            category_target = ""
        if category_target and category_target not in categories:
            category_target = ""
        if not category_target and prompt_help_category_not_equals:
            blocked = prompt_help_category_not_equals.strip().lower()
            category_target = next((c for c in categories if c.lower() != blocked), "Bookings")
        if not category_target and prompt_help_category_not_contains:
            blocked = prompt_help_category_not_contains.strip().lower()
            category_target = next((c for c in categories if blocked not in c.lower()), "Bookings")
        if not category_target:
            category_target = "Bookings"
        category_literal = _xpath_literal(category_target)
        category_key = re.sub(r"[^a-z0-9-]", "", category_target.strip().lower().replace(" ", "-"))
        return [
            _make_nav(autodining_home_url),
            _make_click(autodining_nav_help_xpath),
            _make_click(
                f"(//*[@id='help-category-{category_key}' or @id='category-{category_key}' "
                f"or @id='help-{category_key}-category' or @id='{category_key}-category' "
                f"or @id='help-category-{category_key}-btn' or @id='category-{category_key}-btn' "
                f"or @id='help-{category_key}-filter' or @id='{category_key}-filter' "
                f"or @id='help-{category_key}-category-button' or @id='{category_key}-category-button' "
                f"or normalize-space()={category_literal}])[1]"
            ),
        ]

    if normalized_use_case == "HELP_FAQ_TOGGLED":
        question_target = prompt_faq_question or "How do I make a reservation?"
        question_literal = _xpath_literal(question_target)
        faq_index_by_question = {
            "how do i create an account?": 0,
            "how do i search for restaurants?": 1,
            "how do i make a reservation?": 2,
            "can i modify my reservation?": 3,
            "what is your cancellation policy?": 4,
            "is there a booking fee?": 5,
            "do i pay through autodining?": 6,
            "how do i update my profile information?": 7,
            "how do i view my reservation history?": 8,
            "how are restaurants selected?": 9,
            "can i leave a review?": 10,
            "the website is not loading properly": 11,
            "i didn't receive a confirmation email": 12,
        }
        faq_index = faq_index_by_question.get(str(question_target).strip().lower())
        faq_button_xpath = (
            f"//*[@id='faq-item-{faq_index}']//button"
            if faq_index is not None
            else f"(//button[.//*[contains(normalize-space(), {question_literal})] or contains(normalize-space(), {question_literal})])[1]"
        )
        return [
            _make_nav(autodining_home_url),
            _make_click(autodining_nav_help_xpath),
            _make_click(faq_button_xpath),
        ]

    out: List[Dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        updated = dict(action)
        selector = updated.get("selector") if isinstance(updated.get("selector"), dict) else {}
        selector_type = str(selector.get("type") or "").strip().lower()
        selector_value = str(selector.get("value") or "").strip().lower()
        action_type = str(updated.get("type") or "").strip()
        is_autozone_search_selector = (
            selector_type == "xpathselector"
            and any(
                token in selector_value
                for token in (
                    "type-to-search",
                    "search-input",
                    "query-box",
                    "filter-input",
                    "product-search",
                    "item-search",
                    "search-field",
                    "lookup-input",
                    "find-input",
                    "search-box",
                )
            )
        )

        if selector_value == "input":
            updated["selector"] = {
                "type": "xpathSelector",
                "value": "//input[@type='search']",
                "case_sensitive": False,
            }
        elif selector_value == "search-submit-button":
            updated["selector"] = {
                "type": "xpathSelector",
                "value": "//button[@type='submit']",
                "case_sensitive": False,
            }
        elif selector_value == "__year_option__":
            updated["selector"] = {
                "type": "xpathSelector",
                "value": year_option_xpath,
                "case_sensitive": False,
            }
        elif selector_value == "__year_select__":
            updated["selector"] = {
                "type": "xpathSelector",
                "value": year_select_xpath,
                "case_sensitive": False,
            }
        elif selector_value == "__genre_option__":
            updated["selector"] = {
                "type": "xpathSelector",
                "value": genre_option_xpath,
                "case_sensitive": False,
            }
        elif selector_value == "__genre_select__":
            updated["selector"] = {
                "type": "xpathSelector",
                "value": genre_select_xpath,
                "case_sensitive": False,
            }
        elif (
            action_type == "ClickAction"
            and selector_type == "xpathselector"
            and "category-link" in selector_value
            and "technology" in selector_value
            and prompt_category
        ):
            category_token = re.sub(r"[^a-z0-9 _-]", "", str(prompt_category).strip().lower())
            if category_token:
                replaced = re.sub(r"technology", category_token, str(selector.get("value") or ""), flags=re.I)
                _set_xpath_selector(updated, replaced)
        elif (
            action_type == "ClickAction"
            and selector_type == "xpathselector"
            and "carousel-" in selector_value
            and ("right" in selector_value or "left" in selector_value)
        ):
            if carousel_direction == "LEFT":
                _set_xpath_selector(updated, autozone_carousel_left_xpath)
            else:
                _set_xpath_selector(updated, autozone_carousel_right_xpath)
        elif (
            action_type == "ClickAction"
            and selector_type == "xpathselector"
            and any(token in selector_value for token in ("add-cart-btn", "add-basket", "add-to-basket", "cart-item-add", "add-product"))
        ):
            _set_xpath_selector(updated, autozone_add_cart_xpath)
        elif (
            action_type == "ClickAction"
            and selector_type == "xpathselector"
            and any(token in selector_value for token in ("qty-input", "quantity-field", "qty-field", "quantity-select", "item-qty", "product-qty"))
        ):
            _set_xpath_selector(updated, autozone_quantity_xpath)
        elif (
            normalized_use_case == "CHECKOUT_STARTED"
            and action_type == "ClickAction"
            and selector_type == "xpathselector"
            and any(token in selector_value for token in ("order-now", "checkout-btn", "checkout-now", "proceed-checkout"))
        ):
            _set_xpath_selector(updated, autozone_buy_now_xpath)
        elif (
            normalized_use_case == "ADD_TO_WISHLIST"
            and action_type == "ClickAction"
            and selector_type == "xpathselector"
            and any(token in selector_value for token in ("wishlist-btn", "add-wishlist", "save-later", "wishlist-add", "favorite-btn"))
        ):
            _set_xpath_selector(updated, autozone_detail_wishlist_xpath)
        elif (
            normalized_use_case == "DETAILS_TOGGLE"
            and action_type == "ClickAction"
            and selector_type == "xpathselector"
            and any(token in selector_value for token in ("toggle-btn", "toggle-button", "switch-btn", "toggle-control", "toggle-action"))
        ):
            _set_xpath_selector(updated, autozone_details_toggle_xpath)

        if action_type == "TypeAction":
            if is_autozone_search_selector:
                _set_xpath_selector(updated, autozone_search_input_xpath)
                if autozone_query_hint:
                    updated["text"] = autozone_query_hint
            raw_text = str(updated.get("text") or "")
            if raw_text == "__AUTHOR__":
                updated["text"] = prompt_author or "book"
            elif raw_text == "__GENRE__":
                updated["text"] = prompt_genre or "Fiction"
            elif raw_text == "__RATING__":
                updated["text"] = prompt_rating or "5.0"
            elif raw_text == "__PAGES__":
                updated["text"] = prompt_pages or "500"
            elif raw_text == "__YEAR__":
                updated["text"] = prompt_year or "2020"
            elif raw_text == "__SEARCH_QUERY__":
                updated["text"] = search_query or "book"
            elif raw_text == "__READING_LIST_QUERY__":
                updated["text"] = "Romeo and Juliet"
            elif raw_text == "__CART_QUERY__":
                updated["text"] = "Romeo and Juliet"
            elif raw_text == "__COMMENTER_NAME__":
                updated["text"] = commenter_name or "Alicia"
            elif raw_text == "__COMMENT_MESSAGE__":
                updated["text"] = content_text
            elif raw_text == "__CONTACT_NAME__":
                updated["text"] = contact_name
            elif raw_text == "__CONTACT_EMAIL__":
                updated["text"] = contact_email
            elif raw_text == "__CONTACT_SUBJECT__":
                updated["text"] = contact_subject
            elif raw_text == "__CONTACT_MESSAGE__":
                updated["text"] = contact_message
            elif raw_text == "__FIRST_NAME__":
                updated["text"] = first_name_value
            elif raw_text == "__LAST_NAME__":
                updated["text"] = last_name_value
            elif raw_text == "__WEBSITE__":
                updated["text"] = website_value
            elif raw_text == "__SIGNUP_USERNAME__":
                updated["text"] = signup_username
            elif raw_text == "__SIGNUP_EMAIL__":
                updated["text"] = signup_email
            elif raw_text == "__SIGNUP_PASSWORD__":
                updated["text"] = signup_password
            if selector_value == "comment-name-input" and commenter_name:
                updated["text"] = commenter_name
            elif selector_value == "comment-message-textarea":
                updated["text"] = content_text
            elif (
                selector_value in {"login-username-input", "register-username-input", "username-input"}
                and username
                and raw_text != "__SIGNUP_USERNAME__"
            ):
                updated["text"] = username
            elif selector_value in {"register-email-input", "signup-email-input"} and email and raw_text != "__SIGNUP_EMAIL__":
                updated["text"] = email
            elif selector_value in {
                "login-password-input",
                "register-password-input",
                "register-confirm-password-input",
                "password-input",
                "confirm-password-input",
            } and password and raw_text != "__SIGNUP_PASSWORD__":
                updated["text"] = password
            elif selector_value in {"input", "search-input"} and search_query:
                updated["text"] = search_query
        elif action_type == "SelectAction":
            raw_value = str(updated.get("value") or "")
            if raw_value == "__GENRE_SELECT_VALUE__":
                updated["value"] = prompt_genre or "Fiction"
            elif raw_value == "__YEAR_SELECT_VALUE__":
                updated["value"] = prompt_year or "2020"
        elif action_type == "SendKeysAction":
            raw_keys = updated.get("keys")
            if isinstance(raw_keys, list) and raw_keys:
                key_token = str(raw_keys[0] or "")
                if key_token == "__YEAR_KEY_1__":
                    updated["keys"] = [year_key_values[0]]
                elif key_token == "__YEAR_KEY_2__":
                    updated["keys"] = [year_key_values[1]]
                elif key_token == "__YEAR_KEY_3__":
                    updated["keys"] = [year_key_values[2]]
                elif key_token == "__YEAR_KEY_4__":
                    updated["keys"] = [year_key_values[3]]
        out.append(updated)
    if normalized_use_case == "CAROUSEL_SCROLL" and carousel_direction == "LEFT":
        for idx, action in enumerate(out):
            if not isinstance(action, dict):
                continue
            if str(action.get("type") or "") != "ClickAction":
                continue
            selector = action.get("selector") if isinstance(action.get("selector"), dict) else {}
            selector_value = str(selector.get("value") or "").lower()
            if "carousel-" not in selector_value:
                continue
            right_action = dict(action)
            left_action = dict(action)
            right_xpath = autozone_carousel_right_xpath
            left_xpath = autozone_carousel_left_xpath
            _set_xpath_selector(right_action, right_xpath)
            _set_xpath_selector(left_action, left_xpath)
            out[idx] = right_action
            out.insert(idx + 1, left_action)
            break

    return out


def get_trajectory_bootstrap_actions(
    *,
    web_project_id: str = "",
    use_case: str = "",
    prompt: str = "",
    max_actions: int = 8,
) -> List[Dict[str, Any]]:
    selected = _find_best_trajectory(web_project_id=web_project_id, use_case=use_case, prompt=prompt)
    best_actions = selected.get("actions") if isinstance(selected.get("actions"), list) else []
    selected_use_case = str(selected.get("use_case") or use_case or "")
    adapted_actions = _apply_prompt_overrides(best_actions, prompt=prompt, use_case=selected_use_case)
    return adapted_actions[: max(1, int(max_actions))]


def get_trajectory_replay_bundle(
    *,
    web_project_id: str = "",
    use_case: str = "",
    prompt: str = "",
    apply_prompt_overrides: bool = False,
) -> Dict[str, Any]:
    selected = _find_best_trajectory(web_project_id=web_project_id, use_case=use_case, prompt=prompt)
    if not selected:
        return {}

    actions = selected.get("actions") if isinstance(selected.get("actions"), list) else []
    if bool(apply_prompt_overrides):
        selected_use_case = str(selected.get("use_case") or use_case or "")
        actions = _apply_prompt_overrides(actions, prompt=prompt, use_case=selected_use_case)

    out = dict(selected)
    out["actions"] = actions
    return out


__all__ = [
    "get_trajectory_examples",
    "get_trajectory_bootstrap_actions",
    "get_trajectory_replay_bundle",
]
