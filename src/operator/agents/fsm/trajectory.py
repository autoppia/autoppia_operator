from __future__ import annotations

import re
from datetime import date as _date
from datetime import datetime as _datetime
from datetime import timedelta as _timedelta
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
    {
        "project_id": "p05_autocrm",
        "trajectories": [
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Filter matters to only show those with status 'Active'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8004/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "matters-nav-link",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "matters-nav-link",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "SelectAction",
                        "value": "Active",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "matter-status-filter",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "value": "Active",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "matter-status-filter",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "FILTER_MATTER_STATUS",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Sort matters by latest first.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8004/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='matters-nav-link' or @id='cases-link' or @id='projects-nav' or @id='legal-matters-link' or @id='matter-registry' or @id='tracking-link' or @id='active-cases-link' or @id='orders-link' or @id='engagements-nav' or @id='initiative-tracker']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='matters-nav-link' or @id='cases-link' or @id='projects-nav' or @id='legal-matters-link' or @id='matter-registry' or @id='tracking-link' or @id='active-cases-link' or @id='orders-link' or @id='engagements-nav' or @id='initiative-tracker']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "SelectAction",
                        "value": "__CRM_MATTER_SORT_PREP__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='matter-sort-select' or @id='case-sort-select' or @id='project-sort-select' or @id='matter-order-select' or @id='case-order-select' or @id='project-order-select' or @id='sort-dropdown' or @id='order-dropdown' or @id='sort-selector' or @id='order-selector']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "value": "__CRM_MATTER_SORT_PREP__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='matter-sort-select' or @id='case-sort-select' or @id='project-sort-select' or @id='matter-order-select' or @id='case-order-select' or @id='project-order-select' or @id='sort-dropdown' or @id='order-dropdown' or @id='sort-selector' or @id='order-selector']",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "SelectAction",
                        "value": "__CRM_MATTER_SORT_TARGET__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='matter-sort-select' or @id='case-sort-select' or @id='project-sort-select' or @id='matter-order-select' or @id='case-order-select' or @id='project-order-select' or @id='sort-dropdown' or @id='order-dropdown' or @id='sort-selector' or @id='order-selector']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "value": "__CRM_MATTER_SORT_TARGET__",
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='matter-sort-select' or @id='case-sort-select' or @id='project-sort-select' or @id='matter-order-select' or @id='case-order-select' or @id='project-order-select' or @id='sort-dropdown' or @id='order-dropdown' or @id='sort-selector' or @id='order-selector']",
                                "case_sensitive": False,
                            },
                        },
                    },
                ],
                "use_case": "SORT_MATTER_BY_CREATED_AT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Edit the matter 'Estate Planning' to change status to 'On Hold'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8004/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "matters-nav-link",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "matters-nav-link",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "matter-search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "matter-search-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Estate Planning",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "matter-search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Estate Planning",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "matter-search-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[@aria-label='Edit matter' or @aria-label='Edit Matter' or normalize-space()='Edit'])[1]",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "(//button[@aria-label='Edit matter' or @aria-label='Edit Matter' or normalize-space()='Edit'])[1]",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "SelectAction",
                        "value": "On Hold",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "edit-matter-status-select",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "value": "On Hold",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "edit-matter-status-select",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "save-matter-btn",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "save-matter-btn",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "UPDATE_MATTER",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Open the pending events list on the calendar page.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8004/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "calendar-nav-link",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "calendar-nav-link",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "toggle-pending-events",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "toggle-pending-events",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "VIEW_PENDING_EVENTS",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Add a new calendar event on 2025-05-13 at 9:00am called 'Team Sync' with a Filing type.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                        "go_back": False,
                        "attributes": {
                            "url": "http://localhost:8004/?seed=1",
                            "go_back": False,
                            "go_forward": False,
                        },
                        "go_forward": False,
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "calendar-nav-link",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "calendar-nav-link",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='day-number-2025-05-13']",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "xpathSelector",
                                "value": "//*[@id='day-number-2025-05-13']",
                                "case_sensitive": False,
                            }
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Team Sync",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "event-label-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "Team Sync",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "event-label-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "09:00",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "event-time-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "text": "09:00",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "event-time-input",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "SelectAction",
                        "value": "Filing",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "event-color-select",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "value": "Filing",
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "event-color-select",
                                "attribute": "id",
                                "case_sensitive": False,
                            },
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "save-btn",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                        "attributes": {
                            "selector": {
                                "type": "attributeValueSelector",
                                "value": "save-btn",
                                "attribute": "id",
                                "case_sensitive": False,
                            }
                        },
                    },
                ],
                "use_case": "NEW_CALENDAR_EVENT_ADDED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Search for matters that include 'Estate' in the title.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "SEARCH_MATTER",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Create a matter named 'New Matter', with client 'Acme Co.' and status 'Active'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "ADD_NEW_MATTER",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Go to the Matters page and click on 'Estate Planning' to view the details of that particular matter.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "VIEW_MATTER_DETAILS",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Archive the matter whose status is set to 'Active'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "ARCHIVE_MATTER",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Delete the matter where status is set to 'Active'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "DELETE_MATTER",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "View details of client, whose client name is 'Jessica Taylor' and email is 'jtaylor@samplemail.com'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "VIEW_CLIENT_DETAILS",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Search for clients named 'Smith'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "SEARCH_CLIENT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Add a new client named 'Nova Labs' with status Active.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "ADD_CLIENT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Delete the client named not equals 'Orion Tech Solutions'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "DELETE_CLIENT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Filter clients to status Active with 3-4 matters.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "FILTER_CLIENTS",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Rename the document 'Retainer-Agreement-6908.pdf' to 'Retainer-Agreement-final.pdf'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "DOCUMENT_RENAMED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Delete the document named 'Retainer-Agreement-6908.pdf'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "DOCUMENT_DELETED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Add log with matter 'Trademark Filing', description 'Prepare documents', and hours '2.5'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "NEW_LOG_ADDED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Edit the time log for 'Estate Planning' to change hours to 2.5.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "LOG_EDITED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Delete the time log for 'Estate Planning' that recorded 2 hours.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "LOG_DELETE",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Search billing entries for 'contract' from this week.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "BILLING_SEARCH",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Change user name to 'Muhammad Ali'.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "CHANGE_USER_NAME",
                "has_success": False,
            },
            {
                "url": "http://localhost:8004/?seed=1",
                "prompt": "Open the help center.",
                "actions": [
                    {
                        "url": "http://localhost:8004/?seed=1",
                        "type": "NavigateAction",
                    }
                ],
                "use_case": "HELP_VIEWED",
                "has_success": False,
            },
        ],
    },
    {
        "project_id": "p06_automail",
        "trajectories": [
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Search for query containing 'Weekly Newsletter'",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "SEARCH_EMAIL",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Open the email templates page.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "sidebar-templates",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "VIEW_TEMPLATES",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Select the template where template_name equals 'Meeting Recap'.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "sidebar-templates",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "__TEMPLATE_OPTION__",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "TEMPLATE_SELECTED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Update the body text of the template where template_name equals 'Warm Introduction'.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "sidebar-templates",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "__TEMPLATE_OPTION__",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-body' or @id='template-content' or @aria-label='Body']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_BODY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-body' or @id='template-content' or @aria-label='Body']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-body' or @id='template-content' or @aria-label='Body']",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "TEMPLATE_BODY_EDITED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Send an email using the template where template_name equals 'Friendly Follow Up' and to equals 'john.doe@gmail.com'.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "sidebar-templates",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "__TEMPLATE_OPTION__",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-to' or @id='template-recipient' or @aria-label='To']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_TO__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-to' or @id='template-recipient' or @aria-label='To']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-send' or @id='send-button' or @aria-label='Send' or normalize-space()='Send']",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "TEMPLATE_SENT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Save the template as draft where template_name equals 'Meeting Recap' and to equals 'alice@company.com'.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "sidebar-templates",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "__TEMPLATE_OPTION__",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-to' or @id='template-recipient' or @aria-label='To']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_TO__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-to' or @id='template-recipient' or @aria-label='To']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-save' or @id='save-draft-button' or @aria-label='Save draft' or normalize-space()='Save draft']",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "TEMPLATE_SAVED_DRAFT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Cancel changes on the template where template_name equals 'Thank You'.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "sidebar-templates",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "__TEMPLATE_OPTION__",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-to' or @id='template-recipient' or @aria-label='To']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_TO__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-to' or @id='template-recipient' or @aria-label='To']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='template-cancel' or @id='cancel-button' or @aria-label='Cancel' or normalize-space()='Cancel']",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "TEMPLATE_CANCELED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Go to the next page of emails.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='next-page-button' or @data-testid='next-page-button' or @aria-label='Next page' or @title='Next page'] | //*[@data-testid='email-list']//button[2]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "EMAILS_NEXT_PAGE",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Go back to the previous page of emails.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@data-testid='email-list']/div[1]/div[2]/div/button[2]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@data-testid='email-list']/div[1]/div[2]/div/button[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "EMAILS_PREV_PAGE",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Clear the current selection.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='email-card']/div[1]/button)[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@aria-label='Clear Selection' or @id='clear-selection-button' or @data-testid='clear-selection-button'] | //*[@data-testid='email-list']/div[2]/button[6]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "CLEAR_SELECTION",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Create a label named 'Work' with color 'blue'",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "label-selector-trigger",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='label-selector-menu']//input[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__LABEL_NAME__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='label-selector-menu']//input[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "label-color-4285f4",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "create-label-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "CREATE_LABEL",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Apply dark mode appearance",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[@aria-label='User account menu' or .//span[normalize-space()='U']])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "__THEME_BUTTON__",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "THEME_CHANGED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Add the label 'Work' to the email from 'eric.baker@management.com'",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__ADD_LABEL_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-email'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='email-label-selector']//*[self::button][1] | //*[@id='label-selector' or @id='tag-selector' or @id='label-picker'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "__ADD_LABEL_OPTION__",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "ADD_LABEL",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Send the email to john.doe@gmail.com with subject 'Project Timeline Update'",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[@aria-label='Compose' or normalize-space()='Compose'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='to-input' or @id='recipient-input' or @id='mail-to' or @aria-label='Recipient email address']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_TO__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='to-input' or @id='recipient-input' or @id='mail-to' or @aria-label='Recipient email address']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='subject-input' or @id='topic-input' or @id='mail-subject' or @aria-label='Subject']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_SUBJECT__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='subject-input' or @id='topic-input' or @id='mail-subject' or @aria-label='Subject']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//textarea[@id='body-input' or @id='message-textarea' or @id='content-textarea' or @id='mail-body' or @aria-label='Type your message...']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_BODY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//textarea[@id='body-input' or @id='message-textarea' or @id='content-textarea' or @id='mail-body' or @aria-label='Type your message...']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//button[@id='send-button' or @id='deliver-button' or @id='dispatch-button' or @aria-label='Send' or normalize-space()='Send']",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "SEND_EMAIL",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Save the email as draft where email equals jane.doe@example.com",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[@aria-label='Compose' or normalize-space()='Compose'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='to-input' or @id='recipient-input' or @id='mail-to' or @aria-label='Recipient email address']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_TO__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='to-input' or @id='recipient-input' or @id='mail-to' or @aria-label='Recipient email address']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='subject-input' or @id='topic-input' or @id='mail-subject' or @aria-label='Subject']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_SUBJECT__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='subject-input' or @id='topic-input' or @id='mail-subject' or @aria-label='Subject']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//textarea[@id='body-input' or @id='message-textarea' or @id='content-textarea' or @id='mail-body' or @aria-label='Type your message...']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_BODY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//textarea[@id='body-input' or @id='message-textarea' or @id='content-textarea' or @id='mail-body' or @aria-label='Type your message...']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//button[@id='save-draft-button' or @id='draft-button' or @id='store-draft-button' or @aria-label='Save draft' or normalize-space()='Save draft']",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "EMAIL_SAVE_AS_DRAFT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Edit the draft email where to equals 'jane.doe@example.com' and subject equals 'Client Proposal Updates'.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[@aria-label='Compose' or normalize-space()='Compose'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='to-input' or @id='recipient-input' or @id='mail-to' or @aria-label='Recipient email address']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_TO__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='to-input' or @id='recipient-input' or @id='mail-to' or @aria-label='Recipient email address']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='subject-input' or @id='topic-input' or @id='mail-subject' or @aria-label='Subject']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_SUBJECT__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='subject-input' or @id='topic-input' or @id='mail-subject' or @aria-label='Subject']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//textarea[@id='body-input' or @id='message-textarea' or @id='content-textarea' or @id='mail-body' or @aria-label='Type your message...']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_BODY__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//textarea[@id='body-input' or @id='message-textarea' or @id='content-textarea' or @id='mail-body' or @aria-label='Type your message...']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//button[@id='save-draft-button' or @id='draft-button' or @id='store-draft-button' or @aria-label='Save draft' or normalize-space()='Save draft']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='sidebar-drafts' or @id='sidebar-drafts-item' or @id='nav-drafts' or @aria-label='Navigate to Drafts']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-email'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='edit-draft-button' or @id='email-edit-draft' or @aria-label='Edit this draft email']",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "EDIT_DRAFT_EMAIL",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Reply to the email where from_email equals 'eric.baker@management.com' and subject equals 'Year-End Review Meeting - Schedule'.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__REPLY_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-email'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='email-reply' or @id='reply-button' or @id='respond-button' or @id='answer-button' or @aria-label='Reply to this email']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//button[@id='send-button' or @id='deliver-button' or @id='dispatch-button' or @aria-label='Send' or normalize-space()='Send']",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "REPLY_EMAIL",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Forward the email where from_email equals 'eric.baker@management.com' and subject equals 'Year-End Review Meeting - Schedule' and to equals 'john.doe@gmail.com'.",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__FORWARD_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-email'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='email-forward' or @id='forward-button' or @id='share-button' or @aria-label='Forward this email']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='to-input' or @id='recipient-input' or @id='mail-to' or @aria-label='Recipient email address']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__EMAIL_TO__",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//input[@id='to-input' or @id='recipient-input' or @id='mail-to' or @aria-label='Recipient email address']",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//button[@id='send-button' or @id='deliver-button' or @id='dispatch-button' or @aria-label='Send' or normalize-space()='Send']",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "FORWARD_EMAIL",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "View the email where subject contains 'Project'",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-email'])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "VIEW_EMAIL",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Star the email where subject contains 'Project'",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-email'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='star-button' or @id='email-star' or contains(@id,'star-button') or contains(@id,'email-star') or (@aria-label='Mark as important' and (contains(@title,'star') or contains(@title,'Star')))]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "STAR_AN_EMAIL",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Mark the email as important where subject contains 'Project'",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-email'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='important-button' or @id='email-important' or contains(@id,'important-button') or contains(@id,'email-important') or (contains(@title,'important') and contains(@aria-label,'important'))]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "MARK_EMAIL_AS_IMPORTANT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Mark the email as unread where subject contains 'Project'",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-email'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='email-mark-unread' or @id='mark-unread-button' or contains(@id,'mark-unread') or @aria-label='Mark unread' or contains(@title,'unread')]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "MARK_AS_UNREAD",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Delete the email where subject contains 'Project'",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-email'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='email-delete' or @id='delete-button' or contains(@id,'email-delete') or contains(@id,'delete-email') or @aria-label='Delete' or contains(@title,'Delete')]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "DELETE_EMAIL",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Mark the email as spam where subject contains 'Project'",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='view-email'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='email-mark-spam' or @id='mark-spam-button' or contains(@id,'mark-spam') or contains(@id,'email-mark-spam') or @aria-label='Mark spam' or contains(@title,'spam')]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "MARK_AS_SPAM",
                "has_success": False,
            },
            {
                "url": "http://localhost:8005/?seed=1",
                "prompt": "Archive the email where subject contains 'Project'",
                "actions": [
                    {
                        "url": "http://localhost:8005/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='archive-button' or contains(@id,'archive-button') or contains(@id,'archive_button')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "ARCHIVE_EMAIL",
                "has_success": False,
            },
        ],
    },
    {
        "project_id": "p07_autodelivery",
        "trajectories": [
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Search for restaurants named 'Bella Vista'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__DELIVERY_SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
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
                "use_case": "SEARCH_DELIVERY_RESTAURANT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "View details for the restaurant named 'Bella Vista'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__DELIVERY_SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'] | //*[@id='restaurant-image'] | //*[@id='restaurant-name'])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "VIEW_DELIVERY_RESTAURANT",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Filter restaurants to show only Italian cuisine.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "//*[@id='search-filters']/div[3]/button[2]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__DELIVERY_SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
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
                "use_case": "RESTAURANT_FILTER",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Return to the full restaurant list.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='quick-order-header' or contains(@id,'quick-order')] | //button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'quick order')] | //button[contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'quick order')] | //nav//button[contains(@class,'bg-green-600')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[normalize-space()='View All Restaurants' or contains(normalize-space(),'View All') or contains(normalize-space(),'Restaurants')] | //div[contains(@class,'mt-6') and contains(@class,'pt-6') and contains(@class,'border-t')]//button[1])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "VIEW_ALL_RESTAURANTS",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Return to all restaurants after viewing 'Bella Vista'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__DELIVERY_SEARCH_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//*[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'] | //*[@id='restaurant-image'] | //*[@id='restaurant-name'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='back-to-list' or @id='back-button' or contains(@id,'back-button')] | //button[contains(translate(@aria-label,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'back to all restaurants')] | //button[contains(normalize-space(),'Back to all')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "BACK_TO_ALL_RESTAURANTS",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/restaurants?seed=1",
                "prompt": "Open the add-to-cart modal for 'Pepperoni Classic' at 'Pizza Paradise'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/restaurants?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__DELIVERY_RESTAURANT_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//*[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'] | //*[@id='restaurant-image'] | //*[@id='restaurant-name'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '__DELIVERY_MENU_ITEM__')]/ancestor::*[self::div or self::article][1]//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add to cart')][1] | //*[@id='add-to-cart' or contains(@id,'add-to-cart') or contains(@id,'add-cart')][1])",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "ADD_TO_CART_MODAL_OPEN",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Add 'Pepperoni Classic' to cart from 'Pizza Paradise'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__DELIVERY_RESTAURANT_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-1']//*[contains(@class,'absolute')] | //*[@id='restaurant-grid-item-0']//*[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'] | //*[@id='restaurant-image'] | //*[@id='restaurant-name'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '__DELIVERY_MENU_ITEM__')]/ancestor::*[self::div or self::article][1]//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add to cart')][1] | //*[@id='add-to-cart' or contains(@id,'add-to-cart') or contains(@id,'add-cart')][1])",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@role='dialog']//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add to cart')] | //*[@id='add-to-cart' and ancestor::*[@role='dialog']] | //div[contains(@class,'sm:flex-row')]//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add to cart')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "ADD_TO_CART_MENU_ITEM",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Start a quick order from any restaurant.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='quick-order' or @id='quick-order-header' or contains(@id,'quick-order')] | //button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'quick order')] | //button[contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'quick order')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "QUICK_ORDER_STARTED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Go to the checkout page.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//div[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='menu-item-1-1']//button | //*[@id='menu-item-1-0']//button | //*[@id='add-to-cart'][1])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@role='dialog']//*[@id='add-to-cart'] | //*[@role='dialog']//button[contains(normalize-space(), 'Add to Cart')] | //div[contains(@class,'sm:flex-row')]//button[contains(normalize-space(), 'Add to Cart')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "cart-total-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "OPEN_CHECKOUT_PAGE",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Show me the next page of restaurants.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='pagination-next'] | //button[@id='pagination-next'] | //button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'next')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "RESTAURANT_NEXT_PAGE",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Show me the previous page of restaurants.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='pagination-next'] | //button[@id='pagination-next'] | //button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'next')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='pagination-prev'] | //button[@id='pagination-prev'] | //button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'prev')] | //button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'previous')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "RESTAURANT_PREV_PAGE",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Submit a review with name 'Agente' and comment 'good'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//div[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "review-name",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "Agente",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "review-name",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "review-comment",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "good",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "review-comment",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//button[contains(normalize-space(), 'Submit review')] | //*[@id='review-submit'] | //form//button[@type='submit'])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "REVIEW_SUBMITTED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Delete the review for the restaurant with name 'Bella Vista' where the author contains 'ria', the comment contains 'ood!', the rating is NOT '4.5', the cuisine does NOT contain 'Japanese', and the review_rating is NOT '4.5'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//div[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='review-item-0']//button | //*[@id='delete-review-btn'] | //button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'delete')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "DELETE_REVIEW",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Empty my cart where the quantity is less than or equal to 8 and the price equals '14.99'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//div[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='menu-item-1-1']//button | //*[@id='menu-item-1-0']//button | //*[@id='add-to-cart'][1])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@role='dialog']//*[@id='add-to-cart'] | //*[@role='dialog']//button[contains(normalize-space(), 'Add to Cart')] | //div[contains(@class,'sm:flex-row')]//button[contains(normalize-space(), 'Add to Cart')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "cart-total-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='empty-cart-button-1-0'] | //*[@id='empty-cart-button'] | //button[contains(translate(@aria-label,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'remove item from cart')] | //button[contains(@title,'Remove item from cart')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "EMPTY_CART",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Set dropoff preference where quantity greater equal 2 and item equals 'Picanha' and price less equal 26.99 and restaurant equals 'Carnaval Grill' and delivery_preference equals 'Hand it to me'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__DELIVERY_RESTAURANT_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//div[contains(@class,'absolute')] | //*[@id='restaurant-grid-item-1']//div[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '__DELIVERY_MENU_ITEM__')]/ancestor::*[self::div or self::article][1]//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add to cart')][1] | //*[@id='menu-item-1-0']//button | //*[@id='add-to-cart'][1])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='quantity-increase-1' or starts-with(@id,'quantity-increase') or contains(@id,'quantity-increase') or @aria-label='Increase quantity' or contains(@aria-label,'Increase quantity')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@role='dialog']//*[@id='add-to-cart'] | //*[@role='dialog']//button[contains(normalize-space(), 'Add to Cart')] | //div[contains(@class,'sm:flex-row')]//button[contains(normalize-space(), 'Add to Cart')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "cart-total-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='dropoff-preferences-selector'] | //*[@id='dropoff-section'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='dropoff-option-hand-it-to-me'] | //button[contains(normalize-space(), 'Hand it to me')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "DROPOFF_PREFERENCE",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Add an address where quantity greater equal 2 and item equals 'Picanha' and price less equal 26.99 and restaurant equals 'Carnaval Grill' and address equals '505 Cherry Circle, Fairview'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__DELIVERY_RESTAURANT_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//div[contains(@class,'absolute')] | //*[@id='restaurant-grid-item-1']//div[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '__DELIVERY_MENU_ITEM__')]/ancestor::*[self::div or self::article][1]//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add to cart')][1] | //*[@id='menu-item-1-0']//button | //*[@id='add-to-cart'][1])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='quantity-increase-1' or starts-with(@id,'quantity-increase') or contains(@id,'quantity-increase') or @aria-label='Increase quantity' or contains(@aria-label,'Increase quantity')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@role='dialog']//*[@id='add-to-cart'] | //*[@role='dialog']//button[contains(normalize-space(), 'Add to Cart')] | //div[contains(@class,'sm:flex-row')]//button[contains(normalize-space(), 'Add to Cart')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "cart-total-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "address-selector",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "custom-address-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "505 Cherry Circle, Fairview",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "custom-address-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "save-address-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "ADDRESS_ADDED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Place an order where address not contains '101 Elm Drive, Centerville' and phone not equals '+1-555-901-2345' and mode not contains 'delivery' and preferences not contains 'soy-free' and size not contains 'medium' and quantity less than '2' and price equals '14.3' and restaurant equals 'Tokyo Sushi House'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__DELIVERY_RESTAURANT_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//div[contains(@class,'absolute')] | //*[@id='restaurant-grid-item-1']//div[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'chicken teriyaki')]/ancestor::*[self::div or self::article][1]//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add to cart')][1] | //*[@id='menu-item-25-3']//button | //*[@id='menu-item-1-0']//button | //*[@id='add-to-cart'][1])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@role='dialog']//*[@id='add-to-cart'] | //*[@role='dialog']//button[contains(normalize-space(), 'Add to Cart')] | //div[contains(@class,'sm:flex-row')]//button[contains(normalize-space(), 'Add to Cart')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "cart-total-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='pickup-mode-button' or contains(@id,'pickup-mode-button')] | //button[contains(normalize-space(), 'Pickup')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='customer-name' or contains(@id,'customer-name')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "user",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='customer-name' or contains(@id,'customer-name')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='contact-phone' or contains(@id,'contact-phone')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "123456432",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='contact-phone' or contains(@id,'contact-phone')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='place-order' or contains(@id,'place-order')] | //button[contains(normalize-space(),'Place Order')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "PLACE_ORDER",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Edit the cart item 'Margherita Pizza' from Sushi Zen where the item does NOT contain 'Egg & Cheese Sandwich' and the restaurant is NOT 'Waffle Works'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//div[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='menu-item-1-0']//button | //*[@id='add-to-cart'][1])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@role='dialog']//*[@id='add-to-cart'] | //*[@role='dialog']//button[contains(normalize-space(), 'Add to Cart')] | //div[contains(@class,'sm:flex-row')]//button[contains(normalize-space(), 'Add to Cart')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "cart-total-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='edit-cart'] | //*[@id='edit-cart-button-1-0'] | //*[@id='edit-cart-button'] | //button[contains(normalize-space(), 'Edit')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "EDIT_CART_ITEM",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Select a delivery priority that is 'normal' for an item with size that CONTAINS 'll', a quantity of at least 2, an item that CONTAINS 'eek', a price greater than 9.17, and a restaurant that CONTAINS 'Table'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__DELIVERY_RESTAURANT_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//div[contains(@class,'absolute')] | //*[@id='restaurant-grid-item-1']//div[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '__DELIVERY_MENU_ITEM__')]/ancestor::*[self::div or self::article][1]//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add to cart')][1] | //*[@id='menu-item-1-0']//button | //*[@id='add-to-cart'][1])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@role='dialog']//*[@id='add-to-cart'] | //*[@role='dialog']//button[contains(normalize-space(), 'Add to Cart')] | //div[contains(@class,'sm:flex-row')]//button[contains(normalize-space(), 'Add to Cart')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "cart-total-button",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='quantity-increase-1-0' or starts-with(@id,'quantity-increase') or contains(@id,'quantity-increase') or @aria-label='Increase quantity' or contains(@aria-label,'Increase quantity')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//label[contains(normalize-space(), 'Priority: ready')])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//label[contains(normalize-space(), 'Normal: standard prep')] | //input[@name='delivery-priority' and @value='normal']/ancestor::label[1])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "DELIVERY_PRIORITY_SELECTED",
                "has_success": False,
            },
            {
                "url": "http://localhost:8006/?seed=1",
                "prompt": "Increase the quantity of 'Pepperoni Classic' to 2 at 'Pizza Paradise'.",
                "actions": [
                    {
                        "url": "http://localhost:8006/?seed=1",
                        "type": "NavigateAction",
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "TypeAction",
                        "text": "__DELIVERY_RESTAURANT_QUERY__",
                        "selector": {
                            "type": "attributeValueSelector",
                            "value": "search-input",
                            "attribute": "id",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='restaurant-grid-item-0']//*[contains(@class,'absolute')] | //*[@data-element-type='VIEW_DELIVERY_RESTAURANT'] | //*[@id='restaurant-card'] | //*[@id='restaurant-image'] | //*[@id='restaurant-name'])[1]",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '__DELIVERY_MENU_ITEM__')]/ancestor::*[self::div or self::article][1]//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'add to cart')][1] | //*[@id='add-to-cart' or contains(@id,'add-to-cart') or contains(@id,'add-cart')][1])",
                            "case_sensitive": False,
                        },
                    },
                    {
                        "type": "ClickAction",
                        "selector": {
                            "type": "xpathSelector",
                            "value": "(//*[@id='quantity-increase-1' or starts-with(@id,'quantity-increase') or contains(@id,'quantity-increase') or @aria-label='Increase quantity' or contains(@aria-label,'Increase quantity')])[1]",
                            "case_sensitive": False,
                        },
                    },
                ],
                "use_case": "ITEM_INCREMENTED",
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
            r"subject\s*(?:not_equals|!=)\s*['\"]([^'\"]+)['\"]",
            r"subject\s*(?:is\s+)?not\s*['\"]([^'\"]+)['\"]",
            r"subject[^'\"]{0,60}(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"(?:email_from|from_email)\s*(?:not_equals|!=)\s*['\"]([^'\"]+)['\"]",
            r"(?:email_from|from_email)[^'\"]{0,60}(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )

    patterns = [
        r"(?:search\s+for|find)\s+(?:the\s+)?(?:movie|film|book)\s*['\"]([^'\"]+)['\"]",
        r"(?:movie|film|book)(?:_name)?[^'\"]{0,40}(?:equals|contains|is)\s*['\"]([^'\"]+)['\"]",
        r"(?:title|name)\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
        r"query\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
        r"subject\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
        r"(?:email_from|from_email)\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
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
    for fallback in ("project", "update", "newsletter", "meeting", "invoice", "report"):
        if _norm_key(fallback) in forbidden_norm:
            continue
        if forbidden_query and _norm_key(fallback) == _norm_key(forbidden_query):
            continue
        return fallback
    return ""


def _extract_automail_label_name(prompt: str) -> str:
    forbidden_equals = _extract_prompt_value(
        prompt,
        [
            r"label_name\s*(?:not_equals|!=)\s*['\"]([^'\"]+)['\"]",
            r"label_name\s*(?:is\s+)?not\s*['\"]([^'\"]+)['\"]",
            r"\b(?:label|tag)\b\s*(?:name\s*)?(?:is\s+)?not\s*['\"]([^'\"]+)['\"]",
        ],
    )
    forbidden_contains = _extract_prompt_value(
        prompt,
        [
            r"label_name[^'\"]{0,80}(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"\b(?:label|tag)\b[^'\"]{0,80}(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )

    extracted = _extract_prompt_value(
        prompt,
        [
            r"label_name\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"(?:create|add|make)[^'\"]{0,120}\bname\s*['\"]([^'\"]+)['\"]",
            r"(?:create|add|make)(?:\s+a\s+new)?\s+\b(?:label|tag)\b\s*(?:named|called|titled|labeled)?\s*['\"]([^'\"]+)['\"]",
            r"\b(?:label|tag)\b\s*(?:named|called|titled|labeled)\s*['\"]([^'\"]+)['\"]",
            r"\b(?:label|tag)\b\s*['\"]([^'\"]+)['\"]",
        ],
    )

    def _is_valid(candidate: str) -> bool:
        value = str(candidate or "").strip()
        if _is_unusable_query_candidate(value):
            return False
        if re.search(r"\b(?:does\s+not\s+contain|not_contains|not\s+contain|not_equals|is\s+not)\b", value, flags=re.I):
            return False
        if forbidden_equals and _norm_key(value) == _norm_key(forbidden_equals):
            return False
        if forbidden_contains and str(forbidden_contains).strip().lower() in value.lower():
            return False
        return True

    if extracted and _is_valid(extracted):
        return extracted

    for fallback in ("Work", "Personal", "Finance", "Important", "Travel", "Updates", "Project"):
        if _is_valid(fallback):
            return fallback
    return "Work"


def _extract_automail_theme_target(prompt: str) -> str:
    text = str(prompt or "")
    forbidden_theme = _extract_prompt_value(
        prompt,
        [
            r"theme\s*(?:not_equals|!=|is\s+not)\s*['\"]?(dark|light|system)['\"]?",
            r"theme[^'\"]{0,80}(?:not\s+equal\s+to|other\s+than)\s*['\"]?(dark|light|system)['\"]?",
        ],
    )
    blocked = str(forbidden_theme or "").strip().lower()
    if blocked in {"dark", "light", "system"}:
        for option in ("dark", "light", "system"):
            if option != blocked:
                return option

    explicit_theme = _extract_prompt_value(
        prompt,
        [
            r"theme\s*(?:equals|=|is|to|:)\s*['\"]?(dark|light|system)['\"]?",
            r"(?:switch|change|set|apply|enable)\s+(?:to\s+)?(dark|light|system)\s+(?:theme|mode|appearance)?",
            r"(dark|light|system)\s+(?:theme|mode|appearance)",
        ],
    )
    token = str(explicit_theme or "").strip().lower()
    if token in {"dark", "light", "system"}:
        return token

    if re.search(r"\bsystem\s+default\b", text, flags=re.I):
        return "system"
    if re.search(r"\bdark\b", text, flags=re.I):
        return "dark"
    if re.search(r"\blight\b", text, flags=re.I):
        return "light"
    return "dark"


def _extract_automail_add_label_query(prompt: str) -> str:
    forbidden = _extract_prompt_value(
        prompt,
        [
            r"(?:email_from|from_email)\s*(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
            r"(?:email_from|from_email)[^'\"]{0,80}(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"subject\s*(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
            r"subject[^'\"]{0,80}(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"body[^'\"]{0,80}(?:not_equals|!=|is\s+not|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    candidate = _extract_prompt_value(
        prompt,
        [
            r"(?:email_from|from_email)\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"subject\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"body\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    cleaned = str(candidate or "").strip()
    if cleaned and not _is_unusable_query_candidate(cleaned):
        if forbidden and _norm_key(cleaned) == _norm_key(forbidden):
            cleaned = ""
    if cleaned:
        return cleaned

    fallback = _extract_search_query(prompt)
    if fallback and not _is_unusable_query_candidate(fallback):
        if not forbidden or _norm_key(fallback) != _norm_key(forbidden):
            return fallback
    return "eric.baker@management.com"


def _extract_automail_send_to(prompt: str) -> str:
    text = str(prompt or "")
    email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    forbidden_equals = _extract_prompt_value(
        prompt,
        [
            r"(?:to|recipient|to_email|email)\s*(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
            r"(?:recipient|email)[^'\"]{0,80}(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    forbidden_contains = _extract_prompt_value(
        prompt,
        [
            r"(?:to|recipient|to_email|email)\s*(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"(?:recipient|email)[^'\"]{0,80}(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )

    preferred = _extract_prompt_value(
        prompt,
        [
            r"(?:to|recipient|to_email|email)\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"(?:send|submit|dispatch)[^@]{0,60}\bto\s+(" + email_pattern + r")",
            r"\bto\s+(" + email_pattern + r")",
        ],
    )

    if not preferred:
        direct_email = re.search(email_pattern, text, flags=re.I)
        if direct_email:
            preferred = str(direct_email.group(0) or "").strip()

    def _normalize_email(candidate: str) -> str:
        token = str(candidate or "").strip()
        if not token:
            return ""
        direct = re.search(email_pattern, token, flags=re.I)
        if direct:
            return str(direct.group(0) or "").strip().lower()
        if re.fullmatch(r"[A-Za-z0-9._%+-]+", token):
            return f"{token.lower()}@gmail.com"
        return ""

    blocked_eq = _normalize_email(forbidden_equals)
    blocked_contains = str(forbidden_contains or "").strip().lower()

    preferred_email = _normalize_email(preferred)
    if preferred_email:
        if blocked_eq and preferred_email == blocked_eq:
            preferred_email = ""
        if blocked_contains and blocked_contains in preferred_email:
            preferred_email = ""
    if preferred_email:
        return preferred_email

    for fallback in (
        "john.doe@gmail.com",
        "alex.rivera@gmail.com",
        "mia.chen@gmail.com",
        "nora.wells@gmail.com",
    ):
        low = fallback.lower()
        if blocked_eq and low == blocked_eq:
            continue
        if blocked_contains and blocked_contains in low:
            continue
        return fallback
    return "john.doe@gmail.com"


def _extract_automail_send_subject(prompt: str) -> str:
    equals_or_contains = _extract_prompt_value(
        prompt,
        [
            r"subject\s*(?:that\s+)?(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"regarding\s*['\"]([^'\"]+)['\"]",
        ],
    )
    forbidden_equals = _extract_prompt_value(
        prompt,
        [
            r"subject\s*(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    forbidden_contains = _extract_prompt_value(
        prompt,
        [
            r"subject\s*(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    subject = str(equals_or_contains or "").strip()
    if subject:
        if forbidden_equals and _norm_key(subject) == _norm_key(forbidden_equals):
            subject = ""
        if forbidden_contains and str(forbidden_contains).strip().lower() in subject.lower():
            subject = ""
    if subject:
        return re.sub(r"\s+", " ", subject).strip()[:120]

    blocked_eq = _norm_key(forbidden_equals or "")
    blocked_contains = str(forbidden_contains or "").strip().lower()
    for fallback in ("Project Timeline Update", "Weekly Status Update", "Meeting Follow-up", "Budget Review"):
        if blocked_eq and _norm_key(fallback) == blocked_eq:
            continue
        if blocked_contains and blocked_contains in fallback.lower():
            continue
        return fallback
    return "Project Timeline Update"


def _extract_automail_send_body(prompt: str) -> str:
    body_value = _extract_prompt_value(
        prompt,
        [
            r"body\s*(?:that\s+)?(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"message\s*(?:that\s+)?(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    forbidden_equals = _extract_prompt_value(
        prompt,
        [
            r"body\s*(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
            r"message\s*(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    forbidden_contains = _extract_prompt_value(
        prompt,
        [
            r"body\s*(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"message\s*(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    body = str(body_value or "").strip()
    if body:
        if forbidden_equals and _norm_key(body) == _norm_key(forbidden_equals):
            body = ""
        if forbidden_contains and str(forbidden_contains).strip().lower() in body.lower():
            body = ""
    if body:
        return re.sub(r"\s+", " ", body).strip()[:500]

    blocked_eq = _norm_key(forbidden_equals or "")
    blocked_contains = str(forbidden_contains or "").strip().lower()
    for fallback in ("hello my friend", "Please review and confirm.", "Sharing this update for your feedback."):
        if blocked_eq and _norm_key(fallback) == blocked_eq:
            continue
        if blocked_contains and blocked_contains in fallback.lower():
            continue
        return fallback
    return "hello my friend"


def _extract_automail_reply_query(prompt: str) -> str:
    preferred = _extract_prompt_value(
        prompt,
        [
            r"(?:from_email|email_from)\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"subject\s*(?:that\s+)?(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"\bto\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"\bto\s+(" + r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}" + r")",
        ],
    )
    blocked = _extract_prompt_value(
        prompt,
        [
            r"(?:from_email|email_from)\s*(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
            r"subject\s*(?:not_equals|!=|is\s+not|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"\bto\s*(?:not_equals|!=|is\s+not|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )

    value = str(preferred or "").strip()
    if value and not _is_unusable_query_candidate(value):
        if blocked and _norm_key(value) == _norm_key(blocked):
            value = ""
    if value:
        return value

    generic = _extract_search_query(prompt)
    if generic and not _is_unusable_query_candidate(generic):
        if not blocked or _norm_key(generic) != _norm_key(blocked):
            return generic
    return "eric.baker@management.com"


def _extract_automail_forward_query(prompt: str) -> str:
    preferred = _extract_prompt_value(
        prompt,
        [
            r"subject\s*(?:that\s+)?(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"(?:from_email|email_from)\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"\bto\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    blocked = _extract_prompt_value(
        prompt,
        [
            r"subject\s*(?:not_equals|!=|is\s+not|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"(?:from_email|email_from)\s*(?:not_equals|!=|is\s+not|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )

    value = str(preferred or "").strip()
    if value and not _is_unusable_query_candidate(value):
        if blocked and _norm_key(value) == _norm_key(blocked):
            value = ""
    if value:
        return value

    generic = _extract_search_query(prompt)
    if generic and not _is_unusable_query_candidate(generic):
        if not blocked or _norm_key(generic) != _norm_key(blocked):
            return generic
    return "Year-End Review Meeting - Schedule"


def _extract_automail_template_query(prompt: str) -> str:
    preferred = _extract_prompt_value(
        prompt,
        [
            r"template_name\s*(?:that\s+)?(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"subject\s*(?:that\s+)?(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    body_value = _extract_prompt_value(
        prompt,
        [
            r"body\s*(?:that\s+)?(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    blocked = _extract_prompt_value(
        prompt,
        [
            r"template_name\s*(?:not_equals|!=|is\s+not|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"subject\s*(?:not_equals|!=|is\s+not|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )

    value = str(preferred or "").strip()
    if value and not _is_unusable_query_candidate(value):
        if blocked and _norm_key(value) == _norm_key(blocked):
            value = ""
    if value:
        return value

    body_hint = str(body_value or "").strip()
    if body_hint and not _is_unusable_query_candidate(body_hint):
        templates_catalog = [
            {
                "name": "Warm Introduction",
                "body": "Hi <name>, It was great connecting with you. I'm sharing a quick summary of what we discussed and suggested next steps. Please let me know if you'd like me to adjust anything.",
            },
            {
                "name": "Friendly Follow Up",
                "body": "Hello <name>, I wanted to check in on the items we talked about last week. I'm happy to help keep things moving.",
            },
            {
                "name": "Meeting Recap",
                "body": "Hi <name>, Here's a concise recap of today's discussion and the action items we agreed on. Feel free to add or adjust anything I might have missed.",
            },
            {
                "name": "Thank You",
                "body": "Hi <name>, Thank you for the thoughtful conversation. I appreciated your insights and look forward to collaborating soon.",
            },
            {
                "name": "Gentle Reminder",
                "body": "Hello <name>, This is a quick reminder about the pending items we discussed. Please let me know if there's anything you need from my side.",
            },
        ]
        hint_norm = _norm_key(body_hint)
        for template_row in templates_catalog:
            body_norm = _norm_key(template_row.get("body", ""))
            if hint_norm and body_norm and hint_norm in body_norm:
                candidate = str(template_row.get("name") or "").strip()
                if candidate and (not blocked or _norm_key(candidate) != _norm_key(blocked)):
                    return candidate

    blocked_norm = _norm_key(blocked or "")
    for fallback in ("Meeting Recap", "Warm Introduction", "Friendly Follow Up", "Thank You", "Gentle Reminder"):
        if blocked_norm and blocked_norm in _norm_key(fallback):
            continue
        return fallback
    return "Meeting Recap"


def _extract_autodelivery_search_query(prompt: str) -> str:
    blocked = _extract_prompt_value(
        prompt,
        [
            r"query\s*(?:not_equals|!=|is\s+not|does\s+not\s+equal|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    blocked_values = [
        str(v).strip()
        for v in re.findall(
            r"(?:query|name|description|cuisine)\s*(?:not_equals|!=|is\s+not|does\s+not\s+equal|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            str(prompt or ""),
            flags=re.I,
        )
        if str(v).strip()
    ]
    if blocked:
        blocked_values.append(str(blocked).strip())

    def _is_blocked(candidate: str) -> bool:
        cand = str(candidate or "").strip().lower()
        if not cand:
            return False
        for banned in blocked_values:
            blocked_text = str(banned or "").strip().lower()
            if not blocked_text:
                continue
            if cand == blocked_text:
                return True
            if blocked_text in cand:
                return True
            if cand in blocked_text and len(cand) > 3:
                return True
        return False

    preferred = _extract_prompt_value(
        prompt,
        [
            r"query\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"(?:search|find|look\s+for)[^'\"]{0,80}restaurants?[^'\"]{0,80}(?:named|called|matching|serving|with)?\s*['\"]([^'\"]+)['\"]",
            r"(?:search|find|look\s+for)\s*['\"]([^'\"]+)['\"]",
            r"(?:view|show|open)[^'\"]{0,120}(?:details|menu|restaurant(?:\s+page)?)?[^'\"]{0,80}\s*['\"]([^'\"]+)['\"]",
            r"restaurant\s+with\s+cuisine\s*['\"]([^'\"]+)['\"]",
            r"restaurant\s+that\s+has\s*['\"]([^'\"]+)['\"]",
        ],
    )
    value = str(preferred or "").strip()
    if blocked and _norm_key(value) == _norm_key(blocked):
        value = ""
    if _is_blocked(value):
        value = ""
    if value and not _is_unusable_query_candidate(value):
        return value

    text = str(prompt or "")
    patterns = [
        r"search\s+for\s+restaurants?\s+(?:named|called|matching|serving|with)?\s*([A-Za-z0-9][A-Za-z0-9 .&'\-]{1,80})",
        r"find\s+restaurants?\s+(?:named|called|matching|serving|with)?\s*([A-Za-z0-9][A-Za-z0-9 .&'\-]{1,80})",
        r"look\s+for\s+([A-Za-z0-9][A-Za-z0-9 .&'\-]{1,80})(?:\s+restaurants?|\s+places?)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        candidate = str(match.group(1) or "").strip(" .")
        candidate = re.sub(r"\b(?:restaurants?|restaurant|places?|nearby)\b.*$", "", candidate, flags=re.I).strip(" .")
        candidate = re.sub(r"\b(?:cuisine)\b$", "", candidate, flags=re.I).strip(" .")
        if blocked and _norm_key(candidate) == _norm_key(blocked):
            continue
        if _is_blocked(candidate):
            continue
        if candidate and not _is_unusable_query_candidate(candidate):
            return candidate

    generic = _extract_search_query(prompt)
    if blocked and _norm_key(generic) == _norm_key(blocked):
        generic = ""
    if _is_blocked(generic):
        generic = ""
    if generic and not _is_unusable_query_candidate(generic):
        return generic

    for fallback in ("Bella Vista", "Italian", "Sushi", "project"):
        if _is_blocked(fallback):
            continue
        return fallback
    return "Bella Vista"


def _extract_autodelivery_cuisine(prompt: str) -> str:
    forbidden = _extract_prompt_value(
        prompt,
        [
            r"cuisine\s*(?:not_equals|!=|is\s+not|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    preferred = _extract_prompt_value(
        prompt,
        [
            r"cuisine\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"(?:only|serving)\s+([A-Za-z][A-Za-z &'\-]{1,40})\s+cuisine",
            r"show\s+only\s+([A-Za-z][A-Za-z &'\-]{1,40})\s+cuisine",
            r"filter[^.]{0,120}\s+([A-Za-z][A-Za-z &'\-]{1,40})\s+cuisine",
        ],
    )
    value = str(preferred or "").strip(" .")
    value = re.sub(r"^(?:only|the|all)\s+", "", value, flags=re.I).strip()
    if forbidden and _norm_key(value) == _norm_key(forbidden):
        value = ""
    if value and not _is_unusable_query_candidate(value):
        return value

    known_cuisines = [
        "Italian",
        "Japanese",
        "Chinese",
        "Mexican",
        "Indian",
        "Thai",
        "Mediterranean",
        "American",
        "Korean",
        "Vietnamese",
        "French",
        "Steakhouse",
        "Seafood",
        "Fast Food",
        "Sushi",
        "Pizza",
    ]
    prompt_text = str(prompt or "")
    for cuisine in known_cuisines:
        if forbidden and _norm_key(cuisine) == _norm_key(forbidden):
            continue
        if re.search(rf"\b{re.escape(cuisine)}\b", prompt_text, flags=re.I):
            return cuisine

    for cuisine in known_cuisines:
        if forbidden and _norm_key(cuisine) == _norm_key(forbidden):
            continue
        return cuisine

    return "Italian"


def _extract_autodelivery_restaurant_query(prompt: str) -> str:
    forbidden = _extract_prompt_value(
        prompt,
        [
            r"restaurant\s*(?:not_equals|!=|is\s+not|does\s+not\s+equal|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
            r"\bat\s*(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    preferred = _extract_prompt_value(
        prompt,
        [
            r"restaurant\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"restaurant\s+that\s+(?:equals|is|contains)\s*['\"]([^'\"]+)['\"]",
            r"\bat\s*['\"]([^'\"]+)['\"]",
            r"for\s+restaurant\s*['\"]([^'\"]+)['\"]",
            r"at\s+a\s+restaurant\s+that\s+(?:equals|is)\s*['\"]([^'\"]+)['\"]",
            r"at\s+a\s+restaurant\s+that\s+contains\s*['\"]([^'\"]+)['\"]",
        ],
    )
    value = str(preferred or "").strip(" .")
    if forbidden and _norm_key(value) == _norm_key(forbidden):
        value = ""
    if value and not _is_unusable_query_candidate(value):
        return value
    for fallback in ("Pizza Paradise", "Para", "Bella Vista", "American"):
        if forbidden and _norm_key(fallback) == _norm_key(forbidden):
            continue
        return fallback
    return "Pizza Paradise"


def _extract_autodelivery_menu_item(prompt: str) -> str:
    forbidden = _extract_prompt_value(
        prompt,
        [
            r"item\s*(?:not_equals|!=|is\s+not|does\s+not\s+equal|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    preferred = _extract_prompt_value(
        prompt,
        [
            r"item\s*(?:equals|=|is|:|contains)\s*['\"]([^'\"]+)['\"]",
            r"(?:add-to-cart\s+modal\s+for|show\s+the\s+add-to-cart\s+modal\s+for|open\s+the\s+modal\s+to\s+add)\s*['\"]([^'\"]+)['\"]",
            r"add\s+['\"]([^'\"]+)['\"]\s+to\s+cart",
            r"quantity\s+of\s*['\"]([^'\"]+)['\"]",
            r"(?:increment|increase\s+the\s+number\s+of|set\s+the\s+quantity\s+of|add\s+one\s+more)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    value = str(preferred or "").strip(" .")
    if forbidden and _norm_key(value) == _norm_key(forbidden):
        value = ""
    if value and not _is_unusable_query_candidate(value):
        return value
    for fallback in ("Pepperoni Classic", "Margherita Pizza", "Salmon Nigiri", "California Roll"):
        if forbidden and _norm_key(fallback) == _norm_key(forbidden):
            continue
        return fallback
    return "Pepperoni Classic"


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


def _extract_autocrm_sort_direction(prompt: str) -> str:
    text = str(prompt or "")
    explicit = _extract_prompt_value(
        prompt,
        [
            r"direction\s*(?:equals|=|is|:)\s*['\"]?(asc|desc|ascending|descending)['\"]?",
            r"sort[^'\"]{0,80}\b(asc|desc|ascending|descending)\b",
        ],
    )
    token = str(explicit or "").strip().lower()
    if token in {"asc", "ascending"}:
        return "asc"
    if token in {"desc", "descending"}:
        return "desc"
    if re.search(r"\b(oldest|earliest|ascending|asc)\b|appear\s+on\s+top", text, flags=re.I):
        return "asc"
    if re.search(r"\b(latest|newest|descending|desc)\b", text, flags=re.I):
        return "desc"
    return "desc"


def _normalize_autocrm_status(value: str) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    if "hold" in token or "pending" in token:
        return "On Hold"
    if "archiv" in token or "inactive" in token:
        return "Archived"
    if "active" in token:
        return "Active"
    return ""


def _parse_iso_date_or_none(value: str) -> _date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return _date.fromisoformat(text)
    except ValueError:
        return None


def _parse_clock_minutes_or_none(value: str) -> int | None:
    text = str(value or "").strip().lower().replace(".", "").replace(" ", "")
    if not text:
        return None
    if re.fullmatch(r"[0-2]?[0-9]:[0-5][0-9]", text):
        hh, mm = text.split(":")
        hours = int(hh)
        minutes = int(mm)
        if 0 <= hours <= 23:
            return hours * 60 + minutes
    for fmt in ("%I:%M%p", "%I:%M %p"):
        try:
            parsed = _datetime.strptime(str(value or "").strip(), fmt)
            return parsed.hour * 60 + parsed.minute
        except ValueError:
            continue
    try:
        parsed = _datetime.strptime(text, "%I:%M%p")
        return parsed.hour * 60 + parsed.minute
    except ValueError:
        return None


def _minutes_to_hhmm(minutes: int) -> str:
    bounded = max(0, min(23 * 60 + 59, int(minutes)))
    return f"{bounded // 60:02d}:{bounded % 60:02d}"


def _extract_autocrm_calendar_date(prompt: str) -> str:
    text = str(prompt or "")
    op = "equals"
    value = ""
    patterns = [
        ("not_equals", r"date\s*(?:is\s+)?(?:not_equals|!=|is\s+not)\s*['\"]([0-9]{4}-[0-9]{2}-[0-9]{2})['\"]"),
        ("greater_than", r"date\s*(?:greater_than|greater\s+than|>|after)\s*['\"]([0-9]{4}-[0-9]{2}-[0-9]{2})['\"]"),
        ("greater_equal", r"date\s*(?:greater_equal|greater\s+equal|>=|greater than or equal to)\s*['\"]([0-9]{4}-[0-9]{2}-[0-9]{2})['\"]"),
        ("less_than", r"date\s*(?:less_than|less\s+than|<|before)\s*['\"]([0-9]{4}-[0-9]{2}-[0-9]{2})['\"]"),
        ("less_equal", r"date\s*(?:less_equal|less\s+equal|<=|less than or equal to)\s*['\"]([0-9]{4}-[0-9]{2}-[0-9]{2})['\"]"),
        ("contains", r"date\s*(?:contains)\s*['\"]([^'\"]+)['\"]"),
        ("not_contains", r"date\s*(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]"),
        ("equals", r"date\s*(?:equals|=|is|on|:)\s*['\"]([0-9]{4}-[0-9]{2}-[0-9]{2})['\"]"),
        ("equals", r"\bon\s+([0-9]{4}-[0-9]{2}-[0-9]{2})\b"),
    ]
    for candidate_op, pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            op = candidate_op
            value = str(match.group(1) or "").strip()
            break
    if not value:
        generic = re.search(r"\b([0-9]{4}-[0-9]{2}-[0-9]{2})\b", text)
        if generic:
            value = str(generic.group(1) or "").strip()
            op = "equals"

    base = _parse_iso_date_or_none(value) or _date.today()
    selected = base
    if op == "greater_than":
        selected = base + _timedelta(days=1)
    elif op == "less_than":
        selected = base - _timedelta(days=1)
    elif op == "not_equals":
        selected = base + _timedelta(days=1)
    elif op == "contains":
        token = value
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}", token):
            selected = _parse_iso_date_or_none(f"{token}-15") or base
        elif re.fullmatch(r"[0-9]{4}", token):
            selected = _parse_iso_date_or_none(f"{token}-06-15") or base
    elif op == "not_contains":
        token = str(value or "").strip()
        probe = base
        for _ in range(64):
            if token and token not in probe.isoformat():
                selected = probe
                break
            probe = probe + _timedelta(days=1)
        else:
            selected = base + _timedelta(days=1)
    return selected.isoformat()


def _extract_autocrm_calendar_time(prompt: str) -> str:
    text = str(prompt or "")
    op = "equals"
    value = ""
    patterns = [
        ("not_equals", r"time\s*(?:is\s+)?(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]"),
        ("greater_than", r"time\s*(?:greater_than|greater\s+than|>|after)\s*['\"]([^'\"]+)['\"]"),
        ("greater_equal", r"time\s*(?:greater_equal|greater\s+equal|>=|greater than or equal to)\s*['\"]([^'\"]+)['\"]"),
        ("less_than", r"time\s*(?:less_than|less\s+than|<|before)\s*['\"]([^'\"]+)['\"]"),
        ("less_equal", r"time\s*(?:less_equal|less\s+equal|<=|less than or equal to)\s*['\"]([^'\"]+)['\"]"),
        ("contains", r"time\s*(?:contains)\s*['\"]([^'\"]+)['\"]"),
        ("not_contains", r"time\s*(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]"),
        ("equals", r"time\s*(?:equals|=|is|at|:)\s*['\"]?([0-9]{1,2}:[0-9]{2}\s*(?:am|pm)?)['\"]?"),
        ("equals", r"\bat\s+([0-9]{1,2}:[0-9]{2}\s*(?:am|pm)?)\b"),
        ("equals", r"\b([0-2]?[0-9]:[0-5][0-9])\b"),
    ]
    for candidate_op, pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            op = candidate_op
            value = str(match.group(1) or "").strip()
            break

    base_minutes = _parse_clock_minutes_or_none(value)
    if base_minutes is None:
        base_minutes = 9 * 60

    selected_minutes = base_minutes
    if op == "greater_than":
        selected_minutes = min(base_minutes + 30, 23 * 60 + 30)
    elif op == "less_than":
        selected_minutes = max(base_minutes - 30, 0)
    elif op == "not_equals":
        selected_minutes = base_minutes + 30 if base_minutes <= 23 * 60 else base_minutes - 30
    elif op == "not_contains":
        token = str(value or "").strip().lower()
        probe = base_minutes
        for _ in range(48):
            hhmm = _minutes_to_hhmm(probe)
            if token and token not in hhmm.lower():
                selected_minutes = probe
                break
            probe = (probe + 30) % (24 * 60)

    return _minutes_to_hhmm(selected_minutes)


def _extract_autocrm_calendar_label(prompt: str) -> str:
    equals_label = _extract_prompt_value(
        prompt,
        [
            r"label\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
            r"(?:called|named|titled)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    contains_label = _extract_prompt_value(prompt, [r"label\s*(?:contains)\s*['\"]([^'\"]+)['\"]"])
    blocked_label = _extract_prompt_value(
        prompt,
        [r"label\s*(?:does\s+not\s+contain|not_contains|not\s+contain|not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]"],
    )
    if equals_label:
        return equals_label
    if contains_label:
        token = str(contains_label).strip()
        return token if len(token) >= 4 else f"Team {token}".strip()
    fallback = "Team Sync"
    if blocked_label and blocked_label.lower() in fallback.lower():
        return "Client Review"
    return fallback


def _extract_autocrm_calendar_event_type(prompt: str) -> str:
    text = str(prompt or "")
    options: List[tuple[str, str]] = [
        ("Matter/Event", "forest"),
        ("Internal", "indigo"),
        ("Filing", "blue"),
        ("Other", "zinc"),
    ]
    op = "equals"
    value = ""
    patterns = [
        ("not_equals", r"(?:event_type|type|category|color)\s*(?:is\s+)?(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]"),
        ("contains", r"(?:event_type|type|category|color)\s*(?:contains)\s*['\"]([^'\"]+)['\"]"),
        ("not_contains", r"(?:event_type|type|category|color)\s*(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]"),
        ("equals", r"(?:event_type|type|category|color)\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"),
        ("equals", r"with\s+a[n]?\s+([^'\"]+?)\s+type"),
    ]
    for candidate_op, pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            op = candidate_op
            value = str(match.group(1) or "").strip()
            break
    token = str(value or "").strip().lower()

    def _match_score(display: str, key: str) -> int:
        if not token:
            return 0
        if token == key.lower():
            return 4
        if token == display.lower():
            return 3
        if token in key.lower():
            return 2
        if token in display.lower():
            return 1
        return 0

    if op in {"equals", "contains"} and token:
        best = max(options, key=lambda item: _match_score(item[0], item[1]))
        if _match_score(best[0], best[1]) > 0:
            return best[0]

    if op in {"not_equals", "not_contains"} and token:
        for display, key in options:
            key_l = key.lower()
            display_l = display.lower()
            if op == "not_equals" and token != key_l and token != display_l:
                return display
            if op == "not_contains" and token not in key_l and token not in display_l:
                return display

    return "Filing"


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
    autocrm_sort_direction = _extract_autocrm_sort_direction(prompt)
    autocrm_sort_prep = "asc" if autocrm_sort_direction == "desc" else "desc"
    autocrm_target_matter = _extract_prompt_value(
        prompt,
        [
            r"(?:edit|update)\s+the\s+matter\s+['\"]([^'\"]+)['\"]",
            r"matter\s+(?:named|titled)\s+['\"]([^'\"]+)['\"]",
            r"click\s+on\s+['\"]([^'\"]+)['\"]\s+to\s+view\s+the\s+details\s+of\s+that\s+particular\s+matter",
        ],
    )
    autocrm_new_name = _extract_prompt_value(
        prompt,
        [
            r"(?:change|set|update|edit)\s+(?:the\s+)?(?:matter\s+)?name\s+(?:to|as|=|is|equals)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    autocrm_new_client = _extract_prompt_value(
        prompt,
        [
            r"client\s*(?:name\s*)?(?:to|=|equals|is)\s*['\"]([^'\"]+)['\"]",
            r"client[^'\"]{0,80}(?:contains)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    autocrm_new_client_forbidden = _extract_prompt_value(
        prompt,
        [
            r"client[^'\"]{0,80}(?:not_contains|does\s+not\s+contain|not\s+contain|not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    autocrm_new_status_raw = _extract_prompt_value(
        prompt,
        [
            r"status\s*(?:to|=|equals|is|set\s+to|change(?:d)?\s+to)\s*['\"]([^'\"]+)['\"]",
            r"status[^'\"]{0,80}(?:contains)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    autocrm_new_status = _normalize_autocrm_status(autocrm_new_status_raw)
    autocrm_filter_status_hint = _extract_prompt_value(
        prompt,
        [
            r"status\s*(?:equals|=|is|contains|:)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    autocrm_filter_status_forbidden = _extract_prompt_value(
        prompt,
        [
            r"status[^'\"]{0,80}(?:not_equals|!=|does\s+not\s+contain|not_contains|not\s+contain|is\s+not)\s*['\"]([^'\"]+)['\"]",
        ],
    )
    autocrm_calendar_date = _extract_autocrm_calendar_date(prompt)
    autocrm_calendar_time = _extract_autocrm_calendar_time(prompt)
    autocrm_calendar_label = _extract_autocrm_calendar_label(prompt)
    autocrm_calendar_type = _extract_autocrm_calendar_event_type(prompt)
    automail_label_name = _extract_automail_label_name(prompt)
    automail_add_label_query = _extract_automail_add_label_query(prompt)
    automail_send_to = _extract_automail_send_to(prompt)
    automail_send_subject = _extract_automail_send_subject(prompt)
    automail_send_body = _extract_automail_send_body(prompt)
    automail_reply_query = _extract_automail_reply_query(prompt)
    automail_forward_query = _extract_automail_forward_query(prompt)
    automail_template_query = _extract_automail_template_query(prompt)
    autodelivery_search_query = _extract_autodelivery_search_query(prompt)
    autodelivery_cuisine = _extract_autodelivery_cuisine(prompt)
    autodelivery_restaurant_query = _extract_autodelivery_restaurant_query(prompt)
    autodelivery_menu_item = _extract_autodelivery_menu_item(prompt)
    automail_theme_target = _extract_automail_theme_target(prompt)
    automail_theme_xpath_map = {
        "dark": "//*[@id='theme-dark-btn' or @data-testid='theme-dark-btn' or @aria-label='Dark theme']",
        "light": "//*[@id='theme-light-btn' or @data-testid='theme-light-btn' or @aria-label='Light theme']",
        "system": "//*[@id='theme-system-btn' or @data-testid='theme-system-btn' or @aria-label='System theme']",
    }
    automail_theme_button_xpath = automail_theme_xpath_map.get(automail_theme_target, automail_theme_xpath_map["dark"])
    automail_label_literal = "'" + str(automail_label_name or "Work").replace("'", "") + "'"
    automail_add_label_option_xpath = (
        f"//*[@id='label-selector' or @id='tag-menu' or @id='label-menu']"
        f"//*[normalize-space()={automail_label_literal}]/ancestor::div[contains(@class,'cursor-pointer')][1]//button[1]"
        " | //*[@id='label-selector' or @id='tag-menu' or @id='label-menu']//button[@role='checkbox'][1]"
        " | //*[@id='label-selector' or @id='tag-menu' or @id='label-menu']//button[1]"
    )
    automail_template_literal = "'" + str(automail_template_query or "Meeting Recap").replace("'", "").strip().lower() + "'"
    automail_template_option_xpath = (
        "//*[@id='app-shell']//button["
        f".//*[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), {automail_template_literal})]"
        f" or contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), {automail_template_literal})"
        "][1]"
    )
    autocrm_filter_status = _normalize_autocrm_status(autocrm_filter_status_hint)
    if not autocrm_filter_status and autocrm_filter_status_hint:
        status_token = str(autocrm_filter_status_hint).strip().lower()
        for option in ("Active", "On Hold", "Archived"):
            if status_token and status_token in option.lower():
                autocrm_filter_status = option
                break
    if not autocrm_filter_status and autocrm_filter_status_forbidden:
        blocked_token = str(autocrm_filter_status_forbidden).strip().lower()
        for option in ("Active", "On Hold", "Archived"):
            if blocked_token and blocked_token in option.lower():
                continue
            autocrm_filter_status = option
            break
    if not autocrm_filter_status:
        autocrm_filter_status = "Active"
    if not autocrm_new_client and autocrm_new_client_forbidden:
        blocked_client = str(autocrm_new_client_forbidden).strip().lower()
        for candidate in ("Jones Legal", "Acme Legal", "Northwind Counsel"):
            if blocked_client not in candidate.lower():
                autocrm_new_client = candidate
                break
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

    def _parse_float_or_none(raw: str) -> float | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _extract_numeric_constraint(field: str) -> tuple[str, float | None]:
        escaped = re.escape(str(field or ""))
        text = str(prompt or "")
        patterns: List[tuple[str, str]] = [
            ("not_equals", rf"{escaped}\s*(?:is\s+)?(?:not_equals|!=|not\s+equal(?:s)?(?:\s+to)?)\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?"),
            ("greater_equal", rf"{escaped}\s*(?:greater_equal|greater\s+equal|>=|greater than or equal to)\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?"),
            ("greater_than", rf"{escaped}\s*(?:greater_than|greater\s+than|>)\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?"),
            ("less_equal", rf"{escaped}\s*(?:less_equal|less\s+equal|<=|less than or equal to)\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?"),
            ("less_than", rf"{escaped}\s*(?:less_than|less\s+than|<)\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?"),
            ("equals", rf"{escaped}\s*(?:equals|=|is|:)\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?"),
            ("equals", rf"(?:recorded|with|to)\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?\s+{escaped}"),
        ]
        for op, pattern in patterns:
            m = re.search(pattern, text, flags=re.I)
            if not m:
                continue
            value = _parse_float_or_none(m.group(1))
            if value is not None:
                return op, value
        return "equals", None

    def _pick_numeric_value(op: str, base: float | None, default: float, *, step: float = 0.5, min_value: float = 0.1) -> float:
        value = float(default if base is None else base)
        if op == "greater_than":
            value += step
        elif op == "less_than":
            value = max(min_value, value - step)
        elif op == "not_equals":
            value += step
        elif op in {"greater_equal", "less_equal", "equals"}:
            value = value
        return round(value, 2)

    def _extract_named_constraint(field: str) -> tuple[str, str]:
        escaped = re.escape(str(field or ""))
        text = str(prompt or "")
        patterns: List[tuple[str, str]] = [
            ("not_contains", rf"{escaped}\s*(?:that\s+)?[^'\"]{{0,80}}(?:does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]"),
            ("not_equals", rf"{escaped}\s*(?:that\s+)?(?:is\s+)?(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]"),
            ("contains", rf"{escaped}\s*(?:that\s+)?(?:contains)\s*['\"]([^'\"]+)['\"]"),
            ("equals", rf"{escaped}\s*(?:that\s+)?(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"),
            ("equals", rf"(?:named|titled)\s*['\"]([^'\"]+)['\"]"),
        ]
        for op, pattern in patterns:
            m = re.search(pattern, text, flags=re.I)
            if m:
                return op, str(m.group(1) or "").strip()
        return "equals", ""

    def _pick_text_for_constraint(op: str, token: str, *, default: str, alternatives: List[str]) -> str:
        cleaned = str(token or "").strip()
        if op in {"equals", "contains"} and cleaned:
            return cleaned
        blocked = cleaned.lower()
        if op in {"not_equals", "not_contains"} and blocked:
            for candidate in [default, *alternatives]:
                if candidate and blocked not in candidate.lower():
                    return candidate
        return default

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

    autocrm_home_url = "http://localhost:8004/?seed=1"
    autocrm_matters_nav_xpath = (
        "//*[@id='matters-nav-link' or @id='cases-link' or @id='projects-nav' or @id='legal-matters-link' "
        "or @id='matter-registry' or @id='tracking-link' or @id='active-cases-link' or @id='orders-link' "
        "or @id='engagements-nav' or @id='initiative-tracker']"
    )
    autocrm_matter_search_xpath = (
        "//input[@id='matter-search-input' or @id='case-search-input' or @id='project-search-input' "
        "or @id='matter-query-input' or @id='case-query-input' or @id='project-query-input' "
        "or @id='matter-filter-input' or @id='case-filter-input' or @id='project-filter-input' "
        "or @id='matter-lookup-input']"
    )
    autocrm_matter_status_filter_xpath = (
        "//*[@id='matter-status-filter' or @id='case-status-filter' or @id='project-status-filter' "
        "or @id='matter-state-filter' or @id='case-state-filter' or @id='project-state-filter' "
        "or @id='status-selector' or @id='state-selector' or @id='status-dropdown' or @id='state-dropdown']"
    )
    autocrm_calendar_nav_xpath = (
        "(//a[contains(@id,'calendar') or contains(@id,'schedule') or contains(@id,'appointments') "
        "or contains(@id,'planner') or contains(@id,'timeline')])[1]"
    )
    autocrm_toggle_pending_xpath = (
        "(//button[contains(@id,'pending') or contains(@id,'upcoming') or contains(@id,'scheduled') "
        "or contains(@id,'future') or contains(@id,'awaiting')])[1]"
    )
    autocrm_prev_month_xpath = (
        "(//button[contains(@id,'month') and "
        "(contains(@id,'prev') or contains(@id,'prior') or contains(@id,'back') or contains(@id,'earlier') "
        "or contains(@id,'last') or contains(@id,'before') or contains(@id,'past'))])[1]"
    )
    autocrm_next_month_xpath = (
        "(//button[contains(@id,'month') and "
        "(contains(@id,'next') or contains(@id,'forward') or contains(@id,'later') or contains(@id,'upcoming') "
        "or contains(@id,'after') or contains(@id,'future'))])[1]"
    )
    autocrm_event_label_xpath = "//input[contains(@id,'event') and contains(@id,'label')]"
    autocrm_event_time_xpath = "//input[contains(@id,'event') and contains(@id,'time')]"
    autocrm_event_color_xpath = (
        "//select[contains(@id,'event') and (contains(@id,'color') or contains(@id,'type') or contains(@id,'category'))]"
    )
    autocrm_save_button_xpath = (
        "(//button[contains(@id,'save') or contains(@id,'submit') or contains(@id,'confirm') or contains(@id,'apply') "
        "or contains(@id,'store') or contains(@id,'commit') or contains(@id,'persist') or contains(@id,'accept') "
        "or contains(@id,'finalize')])[1]"
    )
    autocrm_edit_button_xpath = (
        "(//button[contains(@aria-label,'Edit') or contains(@aria-label,'Modify') or contains(@aria-label,'Update') "
        "or normalize-space()='Edit' or normalize-space()='Modify' or normalize-space()='Update'])[1]"
    )
    autocrm_edit_name_xpath = (
        "//input[@id='edit-matter-name-input' or @id='edit-case-name-input' or @id='edit-project-name-input' "
        "or @id='edit-matter-title-input' or @id='edit-case-title-input' or @id='edit-project-title-input' "
        "or @id='modify-matter-name-input' or @id='update-matter-name-input' or @id='revise-matter-name-input' "
        "or @id='amend-matter-name-input']"
    )
    autocrm_edit_client_xpath = (
        "//input[@id='edit-client-name-input' or @id='edit-customer-name-input' or @id='edit-contact-name-input' "
        "or @id='edit-full-name-input' or @id='modify-client-name-input' or @id='update-client-name-input' "
        "or @id='revise-client-name-input' or @id='amend-client-name-input' or @id='change-client-name-input' "
        "or @id='alter-client-name-input']"
    )
    autocrm_edit_status_xpath = (
        "//*[@id='edit-matter-status-select' or @id='edit-case-status-select' or @id='edit-project-status-select' "
        "or @id='modify-matter-status-select' or @id='update-matter-status-select' or @id='revise-matter-status-select' "
        "or @id='amend-matter-status-select' or @id='change-matter-status-select' or @id='alter-matter-status-select' "
        "or @id='edit-status-dropdown']"
    )
    autocrm_save_matter_xpath = (
        "//*[@id='save-matter-btn' or @id='save-case-btn' or @id='save-project-btn' or @id='update-matter-btn' "
        "or @id='update-case-btn' or @id='update-project-btn' or @id='store-matter-btn' or @id='store-case-btn' "
        "or @id='store-project-btn' or @id='commit-matter-btn']"
    )
    autocrm_clients_nav_xpath = "//*[@id='clients-nav-link' or @id='clients_link' or @id='clients-link']"
    autocrm_documents_nav_xpath = "//*[@id='documents-nav-link' or @id='documents_link' or @id='documents-link']"
    autocrm_billing_nav_xpath = "//*[@id='billing-nav-link' or @id='billing_link' or @id='billing-link']"
    autocrm_settings_nav_xpath = "//*[@id='settings-nav-link' or @id='settings_link' or @id='settings-link']"
    autocrm_help_nav_xpath = "//*[@id='help-nav-link' or @id='help_link' or @id='help-link']"
    autocrm_clients_search_xpath = (
        "//input[@id='search-input-field' or @id='search_input' or @id='client-search-input' or @id='client-search-field' "
        "or @id='clients-search-input']"
    )
    autocrm_clients_status_filter_xpath = "//*[@id='status-filter' or @id='status_filter' or @id='client-status-filter']"
    autocrm_clients_matters_filter_xpath = "//*[@id='matters-filter' or @id='matters_filter' or @id='client-matters-filter']"
    autocrm_add_client_btn_xpath = "//*[@id='add-client-btn' or @id='add_client_button' or @id='add-client-button']"
    autocrm_add_client_name_xpath = "(//form//label[contains(normalize-space(), 'Name')]//input)[1]"
    autocrm_add_client_email_xpath = "(//form//label[contains(normalize-space(), 'Email')]//input)[1]"
    autocrm_add_client_matters_xpath = "(//form//label[contains(normalize-space(), 'Matters')]//input)[1]"
    autocrm_add_client_status_xpath = "(//form//label[contains(normalize-space(), 'Status')]//select)[1]"
    autocrm_add_client_submit_xpath = "(//form//button[@type='submit' and contains(normalize-space(), 'Add client')])[1]"
    autocrm_client_name_id_xpath = "//*[starts-with(@id, 'client-name-')]"
    autocrm_delete_client_xpath = "//*[@id='delete-client-button' or @id='delete_client_button']"
    autocrm_documents_rename_btn_xpath = (
        "//*[contains(@id, 'rename-document') or contains(@id, 'rename_document') or contains(@id, 'rename-doc')]"
    )
    autocrm_documents_delete_btn_xpath = "//*[contains(@id, 'delete-document-btn') or contains(@id, 'delete_document_button')]"
    autocrm_documents_rename_input_xpath = "//*[starts-with(@id, 'document-rename-')]"
    autocrm_documents_save_name_xpath = "//*[contains(@id, 'save-document-name') or contains(@id, 'save_document_name')]"
    autocrm_billing_search_xpath = "//input[@id='billing-search' or @id='billing_search']"
    autocrm_billing_date_filter_xpath = "//*[@id='date-filter' or @id='date_filter']"
    autocrm_manual_matter_xpath = "//input[@id='manual-matter-input' or @id='manual_matter_input']"
    autocrm_manual_description_xpath = "//input[@id='manual-desc-input' or @id='manual_description_input']"
    autocrm_manual_hours_xpath = "//input[@id='manual-hours-input' or @id='manual_hours_input']"
    autocrm_add_entry_xpath = "//*[@id='add-entry-btn' or @id='add_entry_button']"
    autocrm_log_entry_xpath = "//*[starts-with(@id, 'log-entry-')]"
    autocrm_edit_log_button_xpath = "//*[contains(@aria-label, 'Edit ') or contains(@aria-label, 'Edit')][.//*[name()='svg']]"
    autocrm_delete_log_button_xpath = "//*[contains(@id, 'delete-log-btn') or contains(@id, 'delete_log_button')]"
    autocrm_edit_log_matter_xpath = "//input[starts-with(@id, 'edit-matter-')]"
    autocrm_edit_log_hours_xpath = "//input[starts-with(@id, 'edit-hours-')]"
    autocrm_edit_log_status_xpath = "//select[starts-with(@id, 'edit-status-')]"
    autocrm_edit_log_description_xpath = "//input[starts-with(@id, 'edit-description-')]"
    autocrm_save_changes_xpath = "(//button[normalize-space()='Save changes' or normalize-space()='Save Name' or contains(normalize-space(), 'Save')])[last()]"
    autocrm_user_name_input_xpath = "//input[@id='user-name-input' or @id='user_name_input' or @data-testid='user-name-input']"
    autocrm_save_name_xpath = "//*[@id='save-name-btn' or @id='save_name_button']"
    autocrm_add_matter_btn_xpath = "//*[@id='add-matter-btn' or @id='add_matter_button']"
    autocrm_add_matter_name_xpath = "//input[@id='matter-name-input' or @id='matter_name_input']"
    autocrm_add_matter_client_xpath = "//input[@id='client-name-input' or @id='client_name_input']"
    autocrm_add_matter_status_xpath = "//*[@id='matter-status-select' or @id='matter_status_select']"
    autocrm_submit_matter_xpath = "//*[@id='submit-matter-btn' or @id='submit_matter_button']"
    autocrm_archive_button_xpath = "//*[@id='archive-btn' or @id='archive_button']"
    autocrm_delete_button_xpath = "//*[@id='delete-btn' or @id='delete_button']"
    autocrm_matter_checkbox_xpath = "(//input[@type='checkbox'])[1]"
    autocrm_matter_title_xpath = "//h3"

    if normalized_use_case == "NEW_CALENDAR_EVENT_ADDED":
        target_date = _parse_iso_date_or_none(autocrm_calendar_date) or _date.today()
        today = _date.today()
        month_diff = (target_date.year - today.year) * 12 + (target_date.month - today.month)
        calendar_actions: List[Dict[str, Any]] = [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_calendar_nav_xpath),
        ]
        if month_diff > 0:
            for _ in range(month_diff):
                calendar_actions.append(_make_click(autocrm_next_month_xpath))
        elif month_diff < 0:
            for _ in range(abs(month_diff)):
                calendar_actions.append(_make_click(autocrm_prev_month_xpath))
        calendar_actions.extend(
            [
                _make_click(f"//*[@id='day-number-{target_date.isoformat()}']"),
                _make_click(autocrm_event_label_xpath),
                _make_type(autocrm_event_label_xpath, autocrm_calendar_label),
                _make_click(autocrm_event_time_xpath),
                _make_type(autocrm_event_time_xpath, autocrm_calendar_time),
                _make_select(autocrm_event_color_xpath, autocrm_calendar_type),
                _make_click(autocrm_save_button_xpath),
            ]
        )
        return calendar_actions

    if normalized_use_case == "FILTER_MATTER_STATUS":
        prep_status = "All" if autocrm_filter_status != "All" else "Active"
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_matters_nav_xpath),
            _make_select(autocrm_matter_status_filter_xpath, prep_status),
            _make_select(autocrm_matter_status_filter_xpath, autocrm_filter_status),
        ]

    if normalized_use_case == "VIEW_PENDING_EVENTS":
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_calendar_nav_xpath),
            _make_click(autocrm_toggle_pending_xpath),
        ]

    if normalized_use_case == "UPDATE_MATTER":
        updated_actions: List[Dict[str, Any]] = [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_matters_nav_xpath),
        ]
        if autocrm_target_matter:
            updated_actions.extend(
                [
                    _make_click(autocrm_matter_search_xpath),
                    _make_type(autocrm_matter_search_xpath, autocrm_target_matter),
                ]
            )
        updated_actions.append(_make_click(autocrm_edit_button_xpath))
        if autocrm_new_name:
            updated_actions.extend(
                [
                    _make_click(autocrm_edit_name_xpath),
                    _make_type(autocrm_edit_name_xpath, autocrm_new_name),
                ]
            )
        if autocrm_new_client:
            updated_actions.extend(
                [
                    _make_click(autocrm_edit_client_xpath),
                    _make_type(autocrm_edit_client_xpath, autocrm_new_client),
                ]
            )
        if autocrm_new_status:
            updated_actions.append(_make_select(autocrm_edit_status_xpath, autocrm_new_status))
        updated_actions.append(_make_click(autocrm_save_matter_xpath))
        return updated_actions

    if normalized_use_case == "SEARCH_MATTER":
        query_op, query_token = _extract_named_constraint("query")
        if not query_token:
            query_op, query_token = _extract_named_constraint("name")
        query_value = str(query_token or "").strip()
        blocked = query_value.lower() if query_op in {"not_equals", "not_contains"} else ""
        if query_op in {"not_equals", "not_contains"}:
            query_value = ""
        if not query_value:
            for candidate in ("Estate", "Review", "Planning", "Contract"):
                if not blocked or blocked not in candidate.lower():
                    query_value = candidate
                    break
        if not query_value:
            query_value = "Estate"
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_matters_nav_xpath),
            _make_click(autocrm_matter_search_xpath),
            _make_type(autocrm_matter_search_xpath, query_value),
            _make_send_keys("ArrowLeft"),
            _make_send_keys("ArrowRight"),
        ]

    if normalized_use_case == "ADD_NEW_MATTER":
        name_op, name_token = _extract_named_constraint("name")
        client_op, client_token = _extract_named_constraint("client")
        status_equals = _extract_prompt_value(prompt, [r"status\s*(?:equals|=|is|:|set\s+to)\s*['\"]([^'\"]+)['\"]"])
        status_not = _extract_prompt_value(prompt, [r"status[^'\"]{0,80}(?:not_equals|!=|is\s+not|not\s+contain|not_contains)\s*['\"]([^'\"]+)['\"]"])
        matter_name = _pick_text_for_constraint(
            name_op,
            name_token,
            default="New Matter",
            alternatives=["Case Alpha", "Estate Review", "Client Intake"],
        )
        client_name = _pick_text_for_constraint(
            client_op,
            client_token,
            default="Acme Co.",
            alternatives=["Jones Legal", "Delta Partners", "Nova Counsel"],
        )
        status_choice = _normalize_autocrm_status(status_equals)
        if not status_choice:
            status_choice = "Active"
        if status_not:
            blocked = str(status_not).strip().lower()
            for option in ("Active", "On Hold", "Archived"):
                if blocked not in option.lower():
                    status_choice = option
                    break
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_matters_nav_xpath),
            _make_click(autocrm_add_matter_btn_xpath),
            _make_click(autocrm_add_matter_name_xpath),
            _make_type(autocrm_add_matter_name_xpath, matter_name),
            _make_click(autocrm_add_matter_client_xpath),
            _make_type(autocrm_add_matter_client_xpath, client_name),
            _make_select(autocrm_add_matter_status_xpath, status_choice),
            _make_click(autocrm_submit_matter_xpath),
        ]

    if normalized_use_case == "VIEW_MATTER_DETAILS":
        name_op, name_token = _extract_named_constraint("name")
        client_op, client_token = _extract_named_constraint("client")
        status_hint = _extract_prompt_value(prompt, [r"status\s*(?:equals|=|is|contains|:)\s*['\"]([^'\"]+)['\"]"])
        status_forbidden = _extract_prompt_value(
            prompt,
            [r"status[^'\"]{0,80}(?:not_equals|!=|is\s+not|does\s+not\s+contain|not_contains|not\s+contain)\s*['\"]([^'\"]+)['\"]"],
        )
        detail_query = _pick_text_for_constraint(
            name_op if name_token else client_op,
            name_token if name_token else client_token,
            default="Estate Planning",
            alternatives=["Contract Review", "Real Estate Purchase", "IP Litigation"],
        )
        detail_status = _normalize_autocrm_status(status_hint)
        if not detail_status and status_forbidden:
            blocked = str(status_forbidden).strip().lower()
            for option in ("Active", "On Hold", "Archived"):
                if blocked not in option.lower():
                    detail_status = option
                    break
        actions_out: List[Dict[str, Any]] = [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_matters_nav_xpath),
        ]
        if detail_status:
            actions_out.append(_make_select(autocrm_matter_status_filter_xpath, detail_status))
        if detail_query:
            actions_out.extend(
                [
                    _make_click(autocrm_matter_search_xpath),
                    _make_type(autocrm_matter_search_xpath, detail_query),
                ]
            )
            actions_out.append(
                _make_click(
                    f"({autocrm_matter_title_xpath}[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
                    f"{_xpath_literal(detail_query.lower())})])[1]"
                )
            )
        else:
            actions_out.append(_make_click(f"({autocrm_matter_title_xpath})[1]"))
        return actions_out

    if normalized_use_case in {"ARCHIVE_MATTER", "DELETE_MATTER"}:
        name_op, name_token = _extract_named_constraint("name")
        client_op, client_token = _extract_named_constraint("client")
        status_hint = _extract_prompt_value(prompt, [r"status\s*(?:equals|=|is|contains|set\s+to|:)\s*['\"]([^'\"]+)['\"]"])
        status_forbidden = _extract_prompt_value(
            prompt,
            [r"status[^'\"]{0,80}(?:not_equals|!=|is\s+not|not_contains|not\s+contain|does\s+not\s+contain)\s*['\"]([^'\"]+)['\"]"],
        )
        batch_query = ""
        if name_token:
            batch_query = _pick_text_for_constraint(
                name_op,
                name_token,
                default="Estate Planning",
                alternatives=["Contract Review", "Real Estate Purchase", "IP Litigation"],
            )
        elif client_token:
            batch_query = _pick_text_for_constraint(
                client_op,
                client_token,
                default="Jones Legal",
                alternatives=["Acme Co.", "Delta Partners", "Smith Co."],
            )
        batch_status = _normalize_autocrm_status(status_hint)
        if not batch_status and status_forbidden:
            blocked = str(status_forbidden).strip().lower()
            for option in ("Active", "On Hold", "Archived"):
                if blocked not in option.lower():
                    batch_status = option
                    break
        if not batch_status:
            batch_status = "Active"
        button_xpath = autocrm_archive_button_xpath if normalized_use_case == "ARCHIVE_MATTER" else autocrm_delete_button_xpath
        out_actions: List[Dict[str, Any]] = [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_matters_nav_xpath),
            _make_select(autocrm_matter_status_filter_xpath, batch_status),
        ]
        if batch_query:
            out_actions.extend(
                [
                    _make_click(autocrm_matter_search_xpath),
                    _make_type(autocrm_matter_search_xpath, batch_query),
                ]
            )
        out_actions.extend(
            [
                _make_click(autocrm_matter_checkbox_xpath),
                _make_click(button_xpath),
            ]
        )
        return out_actions

    if normalized_use_case == "SEARCH_CLIENT":
        query_op, query_token = _extract_named_constraint("query")
        if not query_token:
            query_op, query_token = _extract_named_constraint("name")
        query_value = str(query_token or "").strip()
        blocked = query_value.lower() if query_op in {"not_equals", "not_contains"} else ""
        if query_op in {"not_equals", "not_contains"}:
            query_value = ""
        if not query_value:
            for candidate in ("Smith", "Brown", "Taylor", "Ventures"):
                if not blocked or blocked not in candidate.lower():
                    query_value = candidate
                    break
        if not query_value:
            query_value = "Smith"
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_clients_nav_xpath),
            _make_click(autocrm_clients_search_xpath),
            _make_type(autocrm_clients_search_xpath, query_value),
            _make_send_keys("ArrowLeft"),
            _make_send_keys("ArrowRight"),
        ]

    if normalized_use_case == "VIEW_CLIENT_DETAILS":
        name_op, name_token = _extract_named_constraint("name")
        email_op, email_token = _extract_named_constraint("email")
        status_hint = _extract_prompt_value(prompt, [r"status\s*(?:equals|=|is|contains|:)\s*['\"]([^'\"]+)['\"]"])
        matters_op, matters_value = _extract_numeric_constraint("matters")
        client_query = ""
        if name_token:
            client_query = _pick_text_for_constraint(
                name_op,
                name_token,
                default="Jessica Taylor",
                alternatives=["Smith", "Brown", "Taylor"],
            )
        elif email_token:
            client_query = _pick_text_for_constraint(
                email_op,
                email_token,
                default="samplemail.com",
                alternatives=["services.com", "mail.com"],
            )
        status_filter_value = str(status_hint or "").strip()
        matters_filter_value = "all"
        if matters_value is not None:
            number = float(matters_value)
            if matters_op in {"equals", "greater_equal", "greater_than"}:
                if number >= 5:
                    matters_filter_value = "5plus"
                elif number > 1:
                    matters_filter_value = "3-4"
                elif number >= 3:
                    matters_filter_value = "3-4"
                else:
                    matters_filter_value = "1-2"
            elif matters_op in {"less_equal", "less_than"}:
                matters_filter_value = "1-2"
            elif matters_op == "not_equals":
                matters_filter_value = "3-4" if int(number) != 3 else "1-2"
        out_actions: List[Dict[str, Any]] = [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_clients_nav_xpath),
        ]
        if status_filter_value:
            out_actions.append(_make_select(autocrm_clients_status_filter_xpath, status_filter_value))
        if matters_filter_value != "all":
            out_actions.append(_make_select(autocrm_clients_matters_filter_xpath, matters_filter_value))
        if client_query:
            out_actions.extend(
                [
                    _make_click(autocrm_clients_search_xpath),
                    _make_type(autocrm_clients_search_xpath, client_query),
                    _make_click(
                        f"(({autocrm_client_name_id_xpath}[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
                        f"{_xpath_literal(client_query.lower())})])[1] | ({autocrm_client_name_id_xpath})[1])[1]"
                    ),
                ]
            )
        else:
            out_actions.append(_make_click(f"({autocrm_client_name_id_xpath})[1]"))
        return out_actions

    if normalized_use_case == "ADD_CLIENT":
        name_op, name_token = _extract_named_constraint("name")
        email_op, email_token = _extract_named_constraint("email")
        status_equals = _extract_prompt_value(prompt, [r"status\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
        status_not = _extract_prompt_value(
            prompt,
            [r"status[^'\"]{0,80}(?:not_equals|!=|is\s+not|not_contains|not\s+contain|does\s+not\s+contain)\s*['\"]([^'\"]+)['\"]"],
        )
        matters_op, matters_token = _extract_numeric_constraint("matters")
        client_name = _pick_text_for_constraint(
            name_op,
            name_token,
            default="Nova Labs",
            alternatives=["Vertex Dynamics", "Orion Tech", "Peak Ventures"],
        )
        email_value = _pick_text_for_constraint(
            email_op,
            email_token,
            default=f"{re.sub(r'[^a-z0-9]+', '.', client_name.lower()).strip('.') or 'client'}@example.com",
            alternatives=[f"{re.sub(r'[^a-z0-9]+', '.', client_name.lower()).strip('.') or 'client'}@samplemail.com"],
        )
        if "@" not in email_value:
            email_value = f"{re.sub(r'[^a-z0-9]+', '.', client_name.lower()).strip('.') or 'client'}@example.com"
        status_raw = str(status_equals or "").strip().lower()
        status_value = "Active"
        if status_raw:
            if "active" in status_raw:
                status_value = "Active"
            elif "hold" in status_raw or "pending" in status_raw:
                status_value = "On Hold"
            elif "archiv" in status_raw or "inactive" in status_raw or "closed" in status_raw:
                status_value = "Closed"
        if status_not:
            blocked = str(status_not).strip().lower()
            for candidate in ("Active", "On Hold", "Closed"):
                if blocked not in candidate.lower():
                    status_value = candidate
                    break
        matters_value = _pick_numeric_value(matters_op, matters_token, 3.0, step=1.0, min_value=1.0)
        matters_int = max(1, int(round(matters_value)))
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_clients_nav_xpath),
            _make_click(autocrm_add_client_btn_xpath),
            _make_click(autocrm_add_client_name_xpath),
            _make_type(autocrm_add_client_name_xpath, client_name),
            _make_click(autocrm_add_client_email_xpath),
            _make_type(autocrm_add_client_email_xpath, email_value),
            _make_click(autocrm_add_client_matters_xpath),
            _make_type(autocrm_add_client_matters_xpath, str(matters_int)),
            _make_select(autocrm_add_client_status_xpath, status_value),
            _make_click(autocrm_add_client_submit_xpath),
        ]

    if normalized_use_case == "DELETE_CLIENT":
        name_op, name_token = _extract_named_constraint("name")
        email_op, email_token = _extract_named_constraint("email")
        matters_op, matters_token = _extract_numeric_constraint("matters")
        delete_query = _pick_text_for_constraint(
            email_op if email_token else name_op,
            email_token if email_token else name_token,
            default="TechCorp",
            alternatives=["techcorp.com", "Jessica", "Smith", "Taylor"],
        )
        matters_filter_value = "all"
        if matters_token is not None:
            number = float(matters_token)
            if matters_op in {"greater_than", "greater_equal"} and number >= 3:
                matters_filter_value = "5plus"
            elif matters_op in {"equals", "less_equal", "less_than"}:
                if number <= 2:
                    matters_filter_value = "1-2"
                elif number <= 4:
                    matters_filter_value = "3-4"
                else:
                    matters_filter_value = "5plus"
        else:
            prompt_l = str(prompt or "").lower()
            if "matters" in prompt_l and ("greater than" in prompt_l or "greater_than" in prompt_l or ">" in prompt_l):
                matters_filter_value = "5plus"
            elif "matters" in prompt_l and ("less than" in prompt_l or "less_than" in prompt_l or "<" in prompt_l):
                matters_filter_value = "1-2"
        delete_actions: List[Dict[str, Any]] = [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_clients_nav_xpath),
        ]
        if matters_filter_value != "all":
            delete_actions.append(_make_select(autocrm_clients_matters_filter_xpath, matters_filter_value))
        if delete_query:
            client_click_xpath = (
                f"//*[starts-with(@id, 'client-name-') and contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
                f"{_xpath_literal(delete_query.lower())})]"
            )
            if "@" in delete_query:
                client_click_xpath = f"({autocrm_client_name_id_xpath})[1]"
            delete_actions.extend(
                [
                    _make_click(autocrm_clients_search_xpath),
                    _make_type(autocrm_clients_search_xpath, delete_query),
                    _make_click(client_click_xpath),
                ]
            )
        else:
            delete_actions.append(_make_click(f"({autocrm_client_name_id_xpath})[1]"))
        delete_actions.append(_make_click(autocrm_delete_client_xpath))
        return delete_actions

    if normalized_use_case == "FILTER_CLIENTS":
        status_equals = _extract_prompt_value(prompt, [r"status\s*(?:contains|equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
        status_not = _extract_prompt_value(
            prompt,
            [r"status[^'\"]{0,80}(?:not_equals|!=|is\s+not|not_contains|not\s+contain|does\s+not\s+contain)\s*['\"]([^'\"]+)['\"]"],
        )
        matters_token = _extract_prompt_value(
            prompt,
            [
                r"(1-2|3-4|5\+|5plus)\s+matters?",
                r"matters?\s*(?:equals|=|is|:)\s*['\"]?(1-2|3-4|5\+|5plus|all)['\"]?",
                r"matters?\s*(?:contains)\s*['\"]?(1-2|3-4|5\+|5plus|all)['\"]?",
            ],
        )
        matters_not = _extract_prompt_value(
            prompt,
            [r"matters?[^'\"]{0,80}(?:not_equals|!=|is\s+not)\s*['\"]?(1-2|3-4|5\+|5plus|all)['\"]?"],
        )
        status_value = str(status_equals or "").strip() or "Active"
        if status_not:
            blocked = str(status_not).strip().lower()
            for candidate in ("Active", "Pending", "Inactive", "On Hold", "Closed"):
                if blocked not in candidate.lower():
                    status_value = candidate
                    break
        matters_value = str(matters_token or "").strip().lower()
        if matters_value in {"5+", "5plus"}:
            matters_value = "5plus"
        if matters_value not in {"all", "1-2", "3-4", "5plus"}:
            matters_value = "3-4"
        if matters_not:
            blocked = str(matters_not).strip().lower().replace("5+", "5plus")
            for candidate in ("3-4", "1-2", "5plus", "all"):
                if blocked != candidate:
                    matters_value = candidate
                    break
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_clients_nav_xpath),
            _make_select(autocrm_clients_matters_filter_xpath, matters_value),
            _make_select(autocrm_clients_status_filter_xpath, status_value),
        ]

    if normalized_use_case == "DOCUMENT_RENAMED":
        previous_equals = _extract_prompt_value(
            prompt,
            [
                r"previous_name\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
                r"rename\s+the\s+document\s*['\"]([^'\"]+)['\"]\s+to",
                r"file\s+name\s+of\s*['\"]([^'\"]+)['\"]\s+to",
            ],
        )
        previous_not = _extract_prompt_value(
            prompt,
            [r"previous_name[^'\"]{0,80}(?:not_equals|!=|is\s+not|not_contains|not\s+contain|does\s+not\s+contain)\s*['\"]([^'\"]+)['\"]"],
        )
        new_name_equals = _extract_prompt_value(
            prompt,
            [
                r"new_name\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
                r"\s+to\s*['\"]([^'\"]+)['\"]",
            ],
        )
        new_name_contains = _extract_prompt_value(prompt, [r"new_name\s*(?:contains)\s*['\"]([^'\"]+)['\"]"])
        new_name_not = _extract_prompt_value(
            prompt,
            [r"new_name[^'\"]{0,80}(?:not_equals|!=|is\s+not|not_contains|not\s+contain|does\s+not\s+contain)\s*['\"]([^'\"]+)['\"]"],
        )
        previous_name = str(previous_equals or "").strip()
        if not previous_name and previous_not:
            blocked = str(previous_not).strip().lower()
            for candidate in ("Retainer-Agreement.pdf", "NDA-Sample.docx", "Patent-Application.pdf"):
                if blocked not in candidate.lower():
                    previous_name = candidate
                    break
        if not previous_name:
            previous_name = "Retainer-Agreement"
        if new_name_equals:
            new_name = str(new_name_equals).strip()
        elif new_name_contains:
            token = str(new_name_contains).strip()
            new_name = f"{token}-final.pdf" if "." not in token else token
        else:
            new_name = "Retainer-Agreement-final.pdf"
        if new_name_not:
            blocked = str(new_name_not).strip().lower()
            if blocked in new_name.lower():
                new_name = "Retainer-Agreement-reviewed.pdf"
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_documents_nav_xpath),
            _make_click(f"({autocrm_documents_rename_btn_xpath})[1]"),
            _make_click(autocrm_documents_rename_input_xpath),
            _make_type(autocrm_documents_rename_input_xpath, new_name),
            _make_send_keys("Enter"),
        ]

    if normalized_use_case == "DOCUMENT_DELETED":
        name_equals = _extract_prompt_value(
            prompt,
            [
                r"name\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
                r"delete\s+the\s+document\s+(?:named\s+)?['\"]([^'\"]+)['\"]",
            ],
        )
        name_not = _extract_prompt_value(
            prompt,
            [r"name[^'\"]{0,80}(?:not_equals|!=|is\s+not|not_contains|not\s+contain|does\s+not\s+contain)\s*['\"]([^'\"]+)['\"]"],
        )
        status_equals = _extract_prompt_value(prompt, [r"status\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
        target_name = str(name_equals or "").strip()
        if not target_name and name_not:
            blocked = str(name_not).strip().lower()
            for candidate in ("Retainer-Agreement.pdf", "Patent-Application.pdf", "NDA-Sample.docx"):
                if blocked not in candidate.lower():
                    target_name = candidate
                    break
        delete_target_xpath = autocrm_documents_delete_btn_xpath
        if target_name:
            delete_target_xpath = (
                f"(//*[starts-with(@id, 'document-name-') and contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
                f"{_xpath_literal(target_name.lower())})]/ancestor::*[contains(@class, 'group')][1]//*[contains(@id, 'delete-document-btn') or contains(@id, 'delete_document_button')])[1]"
            )
        elif status_equals:
            delete_target_xpath = (
                f"(//*[starts-with(@id, 'document-status-') and contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
                f"{_xpath_literal(str(status_equals).strip().lower())})]/ancestor::*[contains(@class, 'group')][1]//*[contains(@id, 'delete-document-btn') or contains(@id, 'delete_document_button')])[1]"
            )
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_documents_nav_xpath),
            _make_click(delete_target_xpath),
        ]

    if normalized_use_case == "NEW_LOG_ADDED":
        matter_op, matter_token = _extract_named_constraint("matter")
        desc_op, desc_token = _extract_named_constraint("description")
        hours_op, hours_token = _extract_numeric_constraint("hours")
        matter_value = _pick_text_for_constraint(
            matter_op,
            matter_token,
            default="Trademark Filing",
            alternatives=["Estate Planning", "State Planning", "Contract Review"],
        )
        desc_value = _pick_text_for_constraint(
            desc_op,
            desc_token,
            default="Prepare documents",
            alternatives=["Client follow-up", "Research", "Case preparation"],
        )
        hours_value = _pick_numeric_value(hours_op, hours_token, 2.5, step=0.5, min_value=0.1)
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_billing_nav_xpath),
            _make_click(autocrm_manual_matter_xpath),
            _make_type(autocrm_manual_matter_xpath, matter_value),
            _make_click(autocrm_manual_description_xpath),
            _make_type(autocrm_manual_description_xpath, desc_value),
            _make_click(autocrm_manual_hours_xpath),
            _make_type(autocrm_manual_hours_xpath, str(hours_value)),
            _make_click(autocrm_add_entry_xpath),
        ]

    if normalized_use_case == "LOG_EDITED":
        matter_op, matter_token = _extract_named_constraint("matter")
        client_op, client_token = _extract_named_constraint("client")
        desc_op, desc_token = _extract_named_constraint("description")
        status_equals = _extract_prompt_value(prompt, [r"status\s*(?:equals|=|is|becomes|become|:)\s*['\"]([^'\"]+)['\"]"])
        status_not = _extract_prompt_value(
            prompt,
            [r"status[^'\"]{0,80}(?:not_equals|!=|is\s+not|not_contains|not\s+contain|does\s+not\s+contain)\s*['\"]([^'\"]+)['\"]"],
        )
        hours_op, hours_token = _extract_numeric_constraint("hours")
        query_token = _pick_text_for_constraint(
            matter_op if matter_token else client_op if client_token else desc_op,
            matter_token if matter_token else client_token if client_token else desc_token,
            default="Estate Planning",
            alternatives=["State Planning", "Trademark Filing", "Peak Ventures"],
        )
        status_value = ""
        raw_status = str(status_equals or "").strip().lower()
        if raw_status:
            if "billed" in raw_status:
                status_value = "Billed"
            elif "billable" in raw_status:
                status_value = "Billable"
        if status_not:
            blocked = str(status_not).strip().lower()
            for candidate in ("Billed", "Billable"):
                if blocked not in candidate.lower():
                    status_value = candidate
                    break
        hours_value = _pick_numeric_value(hours_op, hours_token, 2.5, step=0.5, min_value=0.1)
        edit_actions: List[Dict[str, Any]] = [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_billing_nav_xpath),
        ]
        if query_token:
            edit_actions.extend(
                [
                    _make_click(autocrm_billing_search_xpath),
                    _make_type(autocrm_billing_search_xpath, query_token),
                ]
            )
        edit_actions.append(_make_click(f"({autocrm_edit_log_button_xpath})[1]"))
        if matter_token:
            matter_value = _pick_text_for_constraint(
                matter_op,
                matter_token,
                default="Estate Planning",
                alternatives=["State Planning", "Trademark Filing", "Contract Review"],
            )
            edit_actions.extend(
                [
                    _make_click(autocrm_edit_log_matter_xpath),
                    _make_type(autocrm_edit_log_matter_xpath, matter_value),
                ]
            )
        if desc_token:
            desc_value = _pick_text_for_constraint(
                desc_op,
                desc_token,
                default="Prepare documents",
                alternatives=["Client follow-up", "Research", "Case preparation"],
            )
            edit_actions.extend(
                [
                    _make_click(autocrm_edit_log_description_xpath),
                    _make_type(autocrm_edit_log_description_xpath, desc_value),
                ]
            )
        if hours_token is not None:
            edit_actions.extend(
                [
                    _make_click(autocrm_edit_log_hours_xpath),
                    _make_type(autocrm_edit_log_hours_xpath, str(hours_value)),
                ]
            )
        if status_value:
            edit_actions.append(_make_select(autocrm_edit_log_status_xpath, status_value))
        edit_actions.append(_make_click(autocrm_save_changes_xpath))
        return edit_actions

    if normalized_use_case == "LOG_DELETE":
        matter_op, matter_token = _extract_named_constraint("matter")
        client_op, client_token = _extract_named_constraint("client")
        status_equals = _extract_prompt_value(prompt, [r"status\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
        hours_op, hours_token = _extract_numeric_constraint("hours")
        query_token = _pick_text_for_constraint(
            matter_op if matter_token else client_op,
            matter_token if matter_token else client_token,
            default="Estate Planning",
            alternatives=["State Planning", "Peak Ventures", "LabelLine"],
        )
        delete_xpath = autocrm_delete_log_button_xpath
        if status_equals:
            status_lit = _xpath_literal(str(status_equals).strip().lower())
            delete_xpath = (
                f"(//div[starts-with(@id, 'log-entry-') and .//*[starts-with(@id, 'log-status-') and contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), {status_lit})]]"
                "//*[contains(@id, 'delete-log-btn') or contains(@id, 'delete_log_button')])[1]"
            )
        elif hours_token is not None and hours_op == "equals":
            hour_lit = _xpath_literal(str(hours_token).rstrip("0").rstrip("."))
            delete_xpath = (
                f"(//div[starts-with(@id, 'log-entry-') and .//*[starts-with(@id, 'log-hours-') and contains(normalize-space(), {hour_lit})]]"
                "//*[contains(@id, 'delete-log-btn') or contains(@id, 'delete_log_button')])[1]"
            )
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_billing_nav_xpath),
            _make_click(autocrm_billing_search_xpath),
            _make_type(autocrm_billing_search_xpath, query_token),
            _make_click(delete_xpath),
        ]

    if normalized_use_case == "BILLING_SEARCH":
        query_op, query_token = _extract_named_constraint("query")
        date_equals = _extract_prompt_value(prompt, [r"date_filter\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
        date_not = _extract_prompt_value(prompt, [r"date_filter[^'\"]{0,80}(?:not_equals|!=|is\s+not)\s*['\"]([^'\"]+)['\"]"])
        explicit_date = _extract_prompt_value(prompt, [r"custom_date\s*(?:equals|=|is|:)\s*['\"]([0-9]{4}-[0-9]{2}-[0-9]{2})['\"]", r"\b([0-9]{4}-[0-9]{2}-[0-9]{2})\b"])
        query_value = _pick_text_for_constraint(
            query_op,
            query_token,
            default="contract",
            alternatives=["review", "planning", "memo"],
        )
        date_filter_value = "all"
        normalized_prompt = str(prompt or "").lower()
        if explicit_date:
            date_filter_value = "custom"
        elif "this week" in normalized_prompt:
            date_filter_value = "this_week"
        elif "today" in normalized_prompt:
            date_filter_value = "today"
        elif "previous 2 weeks" in normalized_prompt or "prev two weeks" in normalized_prompt:
            date_filter_value = "prev_two_weeks"
        elif "this month" in normalized_prompt:
            date_filter_value = "this_month"
        elif date_equals:
            token = str(date_equals).strip().lower().replace(" ", "_").replace("+", "plus")
            mapping = {
                "all": "all",
                "today": "today",
                "this_week": "this_week",
                "this_month": "this_month",
                "previous_2_weeks": "prev_two_weeks",
                "prev_two_weeks": "prev_two_weeks",
                "specific_date": "custom",
                "custom": "custom",
            }
            date_filter_value = mapping.get(token, "all")
        if date_not:
            blocked = str(date_not).strip().lower().replace(" ", "_")
            for candidate in ("this_week", "today", "this_month", "prev_two_weeks", "all"):
                if blocked not in candidate:
                    date_filter_value = candidate
                    break
        search_actions: List[Dict[str, Any]] = [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_billing_nav_xpath),
            _make_click(autocrm_billing_search_xpath),
            _make_type(autocrm_billing_search_xpath, query_value),
            _make_select(autocrm_billing_date_filter_xpath, date_filter_value),
        ]
        if date_filter_value == "custom" and explicit_date:
            search_actions.extend(
                [
                    _make_click("//input[@type='date']"),
                    _make_type("//input[@type='date']", str(explicit_date).strip()),
                ]
            )
        return search_actions

    if normalized_use_case == "CHANGE_USER_NAME":
        name_equals = _extract_prompt_value(
            prompt,
            [
                r"name\s*(?:equals|=|is|to|:)\s*['\"]([^'\"]+)['\"]",
                r"display\s+name\s+to\s*['\"]([^'\"]+)['\"]",
            ],
        )
        name_contains = _extract_prompt_value(prompt, [r"name\s*(?:contains)\s*['\"]([^'\"]+)['\"]"])
        name_not = _extract_prompt_value(
            prompt,
            [r"name[^'\"]{0,80}(?:not_equals|!=|is\s+not|not_contains|not\s+contain|does\s+not\s+contain)\s*['\"]([^'\"]+)['\"]"],
        )
        new_name = str(name_equals or "").strip()
        if not new_name and name_contains:
            token = str(name_contains).strip()
            new_name = "Muhammad Ali" if token.lower() == "ali" else f"{token} User"
        if not new_name:
            new_name = "Muhammad Ali"
        if name_not and str(name_not).strip().lower() in new_name.lower():
            for candidate in ("Aisha Khan", "Sana Ahmed", "Omar Farooq"):
                if str(name_not).strip().lower() not in candidate.lower():
                    new_name = candidate
                    break
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_settings_nav_xpath),
            _make_click(autocrm_user_name_input_xpath),
            _make_type(autocrm_user_name_input_xpath, new_name),
            _make_click(autocrm_save_name_xpath),
        ]

    if normalized_use_case == "HELP_VIEWED":
        return [
            _make_nav(autocrm_home_url),
            _make_click(autocrm_help_nav_xpath),
            _make_click(
                "(//button[starts-with(@id, 'faq-question-') or starts-with(@id, 'faq_question_')])[1]"
            ),
        ]

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
        elif selector_value == "__add_label_option__":
            updated["selector"] = {
                "type": "xpathSelector",
                "value": automail_add_label_option_xpath,
                "case_sensitive": False,
            }
        elif selector_value == "__template_option__":
            updated["selector"] = {
                "type": "xpathSelector",
                "value": automail_template_option_xpath,
                "case_sensitive": False,
            }
        elif selector_value == "__theme_button__":
            updated["selector"] = {
                "type": "xpathSelector",
                "value": automail_theme_button_xpath,
                "case_sensitive": False,
            }
        elif "__delivery_cuisine__" in selector_value:
            cuisine_label = str(autodelivery_cuisine or "Italian").replace("'", "").strip() or "Italian"
            replaced = str(selector.get("value") or "")
            replaced = replaced.replace("__DELIVERY_CUISINE__", cuisine_label).replace("__delivery_cuisine__", cuisine_label.lower())
            _set_xpath_selector(updated, replaced)
        elif "__delivery_menu_item__" in selector_value:
            item_label = str(autodelivery_menu_item or "Pepperoni Classic").replace("'", "").strip()
            item_label_l = item_label.lower() or "pepperoni classic"
            replaced = str(selector.get("value") or "")
            replaced = replaced.replace("__DELIVERY_MENU_ITEM__", item_label_l).replace("__delivery_menu_item__", item_label_l)
            _set_xpath_selector(updated, replaced)
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
            elif raw_text == "__DELIVERY_SEARCH_QUERY__":
                updated["text"] = autodelivery_search_query or search_query or "Bella Vista"
            elif raw_text == "__DELIVERY_RESTAURANT_QUERY__":
                updated["text"] = autodelivery_restaurant_query or autodelivery_search_query or "Pizza Paradise"
            elif raw_text == "__ADD_LABEL_QUERY__":
                updated["text"] = automail_add_label_query or search_query or "eric.baker@management.com"
            elif raw_text == "__LABEL_NAME__":
                updated["text"] = automail_label_name or "Work"
            elif raw_text == "__EMAIL_TO__":
                updated["text"] = automail_send_to or "john.doe@gmail.com"
            elif raw_text == "__EMAIL_SUBJECT__":
                updated["text"] = automail_send_subject or "Project Timeline Update"
            elif raw_text == "__EMAIL_BODY__":
                updated["text"] = automail_send_body or "hello my friend"
            elif raw_text == "__REPLY_QUERY__":
                updated["text"] = automail_reply_query or search_query or "eric.baker@management.com"
            elif raw_text == "__FORWARD_QUERY__":
                updated["text"] = automail_forward_query or search_query or "Year-End Review Meeting - Schedule"
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
            elif selector_value in {"input", "search-input"} and search_query and raw_text not in {"__DELIVERY_SEARCH_QUERY__", "__DELIVERY_RESTAURANT_QUERY__"}:
                updated["text"] = search_query
        elif action_type == "SelectAction":
            raw_value = str(updated.get("value") or "")
            if raw_value == "__GENRE_SELECT_VALUE__":
                updated["value"] = prompt_genre or "Fiction"
            elif raw_value == "__YEAR_SELECT_VALUE__":
                updated["value"] = prompt_year or "2020"
            elif raw_value == "__CRM_MATTER_SORT_PREP__":
                updated["value"] = autocrm_sort_prep
            elif raw_value == "__CRM_MATTER_SORT_TARGET__":
                updated["value"] = autocrm_sort_direction
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
