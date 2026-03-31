from __future__ import annotations

import re
from typing import Any, Dict, List

AUTOBOOKS_URL = "http://localhost:8001/?seed=1"


def _id_selector(value: str) -> Dict[str, Any]:
    return {
        "type": "attributeValueSelector",
        "value": value,
        "attribute": "id",
        "case_sensitive": False,
    }


def _xpath_selector(value: str) -> Dict[str, Any]:
    return {
        "type": "xpathSelector",
        "value": value,
        "case_sensitive": False,
    }


def _navigate(url: str = AUTOBOOKS_URL) -> Dict[str, Any]:
    return {
        "url": url,
        "type": "NavigateAction",
        "go_back": False,
        "attributes": {
            "url": url,
            "go_back": False,
            "go_forward": False,
        },
        "go_forward": False,
    }


def _click(selector: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "ClickAction",
        "selector": selector,
        "attributes": {"selector": selector},
    }


def _type(selector: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {
        "type": "TypeAction",
        "text": text,
        "selector": selector,
        "attributes": {
            "text": text,
            "selector": selector,
        },
    }


def _click_id(value: str) -> Dict[str, Any]:
    return _click(_id_selector(value))


def _click_xpath(value: str) -> Dict[str, Any]:
    return _click(_xpath_selector(value))


def _type_id(value: str, text: str) -> Dict[str, Any]:
    return _type(_id_selector(value), text)


def _type_xpath(value: str, text: str) -> Dict[str, Any]:
    return _type(_xpath_selector(value), text)


def _login_steps() -> List[Dict[str, Any]]:
    return [
        _click_xpath("//a[normalize-space()='Login']"),
        _click_id("username-input"),
        _type_id("username-input", "<username>"),
        _click_id("password-input"),
        _type_id("password-input", "<password>"),
        _click_id("login-submit-button"),
    ]


def _open_first_book_detail_steps() -> List[Dict[str, Any]]:
    return [
        _click_xpath("(//a[contains(@href,'/books/')])[1]"),
    ]


def _open_book_detail_after_login_steps() -> List[Dict[str, Any]]:
    return [
        _click_xpath("//a[contains(@href,'/search')]"),
        *_open_first_book_detail_steps(),
    ]




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
                "url": AUTOBOOKS_URL,
                "prompt": "Register with username, email and password placeholders.",
                "actions": [
                    _navigate(),
                    _click_xpath("//a[normalize-space()='Register']"),
                    _click_id("username-input"),
                    _type_id("username-input", "<signup_username>"),
                    _click_id("signup-email-input"),
                    _type_id("signup-email-input", "<signup_email>"),
                    _click_id("password-input"),
                    _type_id("password-input", "<signup_password>"),
                    _click_id("confirm-password-input"),
                    _type_id("confirm-password-input", "<signup_password>"),
                    _click_id("signup-submit-button"),
                ],
                "use_case": "REGISTRATION_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Look for the book 'Lolita'",
                "actions": [
                    _navigate("http://localhost:8001/?seed=169"),
                    _click_xpath("//*[@id='search-field' or @id='search-input']"),
                    _type_xpath("//*[@id='search-field' or @id='search-input']", "__SEARCH_QUERY__"),
                    _click_xpath("//*[@id='submit-btn' or @id='search-submit-button' or @id='search-button']"),
                ],
                "use_case": "SEARCH_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Filter books released in the year 2021",
                "actions": [
                    _navigate("http://localhost:8001/search?year=2021&seed=169"),
                ],
                "use_case": "FILTER_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Fill and submit the contact form.",
                "actions": [
                    _navigate(),
                    _click_xpath("//a[normalize-space()='Contact']"),
                    _click_id("contact-name-input"),
                    _type_id("contact-name-input", "John Smith"),
                    _click_id("contact-email-input"),
                    _type_id("contact-email-input", "john@example.com"),
                    _click_id("contact-subject-input"),
                    _type_id("contact-subject-input", "Feedback"),
                    _click_id("contact-message-textarea"),
                    _type_id("contact-message-textarea", "Great website, I love the design"),
                    _click_id("send-message-button"),
                ],
                "use_case": "CONTACT_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login with username and password.",
                "actions": [
                    _navigate(),
                    *_login_steps(),
                ],
                "use_case": "LOGIN_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login and logout.",
                "actions": [
                    _navigate(),
                    *_login_steps(),
                    _click_xpath("//button[normalize-space()='Logout']"),
                ],
                "use_case": "LOGOUT_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login and delete one assigned book.",
                "actions": [
                    _navigate(),
                    *_login_steps(),
                    _click_xpath("//a[contains(@href,'/profile')]"),
                    _click_id("profile-tab-books"),
                    _click_xpath("(//*[starts-with(@id,'delete-book-button')])[1]"),
                ],
                "use_case": "DELETE_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login and add a new book.",
                "actions": [
                    _navigate(),
                    *_login_steps(),
                    _click_xpath("//a[contains(@href,'/profile')]"),
                    _click_id("profile-tab-add-books"),
                    _click_xpath("(//*[@id='add-books-section']//label[contains(normalize-space(),'Title')]/input)[1]"),
                    _type_xpath("(//*[@id='add-books-section']//label[contains(normalize-space(),'Title')]/input)[1]", "New Book"),
                    _click_xpath("(//*[@id='add-books-section']//label[contains(normalize-space(),'Author')]/input)[1]"),
                    _type_xpath("(//*[@id='add-books-section']//label[contains(normalize-space(),'Author')]/input)[1]", "cinema writer"),
                    _click_xpath("(//*[@id='add-books-section']//label[contains(normalize-space(),'Year')]/input)[1]"),
                    _type_xpath("(//*[@id='add-books-section']//label[contains(normalize-space(),'Year')]/input)[1]", "2010"),
                    _click_xpath("(//*[@id='add-books-section']//label[contains(normalize-space(),'Pages')]/input)[1]"),
                    _type_xpath("(//*[@id='add-books-section']//label[contains(normalize-space(),'Pages')]/input)[1]", "320"),
                    _click_xpath("(//*[@id='add-books-section']//label[contains(normalize-space(),'Rating')]/input)[1]"),
                    _type_xpath("(//*[@id='add-books-section']//label[contains(normalize-space(),'Rating')]/input)[1]", "4.7"),
                    _click_xpath("//*[@id='add-books-section']//button[normalize-space()='Fiction']"),
                    _click_xpath("//*[@id='add-books-section']//button[normalize-space()='Add Book']"),
                ],
                "use_case": "ADD_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Open a book and add a comment.",
                "actions": [
                    _navigate(),
                    *_open_first_book_detail_steps(),
                    _click_id("comment-author-input"),
                    _type_id("comment-author-input", "Alicia"),
                    _click_id("comment-message-textarea"),
                    _type_id("comment-message-textarea", "Great read with strong pacing."),
                    _click_id("share-feedback-button"),
                ],
                "use_case": "ADD_COMMENT_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login and edit user profile fields.",
                "actions": [
                    _navigate(),
                    *_login_steps(),
                    _click_xpath("//a[contains(@href,'/profile')]"),
                    _click_id("first-name-input"),
                    _type_id("first-name-input", "oliver"),
                    _click_id("location-input"),
                    _type_id("location-input", "Seoul, South Korea"),
                    _click_id("website-input"),
                    _type_id("website-input", "shadowbooks.com"),
                    _click_id("save-profile-button"),
                ],
                "use_case": "EDIT_USER_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Navigate to '1984' book page",
                "actions": [
                    _navigate("http://localhost:8001/books/book-188?seed=5"),
                ],
                "use_case": "BOOK_DETAIL",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login with username: <username> and password: <password>. Edit a book by changing the rating to 4.8.",
                "actions": [
                    _navigate("http://localhost:8001/?seed=15"),
                    *_login_steps(),
                    _click_xpath("//*[@id='user-reading-list-tab' or @id='profile-tab-reading-list']"),
                    _click_xpath("//*[@id='my-add-books-tab' or @id='profile-tab-add-books']"),
                    _click_xpath("//*[@id='books-view-tab' or @id='profile-tab-books']"),
                    _click_xpath(
                        "(//*[@id='profile-edit-books-section']//form//label[contains(translate(normalize-space(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'rating')]/input)[1]"
                    ),
                    _type_xpath("(//label[contains(normalize-space(),'Rating')]/input)[1]", "__RATING__"),
                    _click_xpath("(//*[@id='profile-edit-books-section']//form//button[@type='submit'])[1]"),
                ],
                "use_case": "EDIT_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login with username: <username> and password: <password>. After logging in, purchase the book 'The Silent Patient'.",
                "actions": [
                    _navigate("http://localhost:8001/?seed=15"),
                    *_login_steps(),
                    _click_xpath("//a[contains(@href,'/search')]"),
                    _click_xpath("//*[@id='search-field' or @id='search-input']"),
                    _type_xpath("//*[@id='search-field' or @id='search-input']", "__SEARCH_QUERY__"),
                    _click_xpath("//*[@id='view-book-details-button' or @id='spotlight-view-details-btn' or starts-with(@id,'featured-book-view-details-btn')]"),
                    _click_xpath("//*[@id='cart-button' or @id='add-cart-button' or @id='add-to-cart-detail-button']"),
                    _click_xpath("//a[contains(@href,'/cart')]"),
                    _click_xpath("//*[@id='buy-now-button' or @id='purchase-button']"),
                ],
                "use_case": "PURCHASE_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Open detail and share the book.",
                "actions": [
                    _navigate(),
                    *_open_first_book_detail_steps(),
                    _click_id("share-detail-button"),
                ],
                "use_case": "SHARE_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Open detail and start preview.",
                "actions": [
                    _navigate(),
                    *_open_first_book_detail_steps(),
                    _click_id("read-book-button"),
                ],
                "use_case": "OPEN_PREVIEW",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login with username: <username> and password: <password>. Add 'The Iliad' to your reading list.",
                "actions": [
                    _navigate(),
                    *_login_steps(),
                    _click_xpath("//a[contains(@href,'/')]"),
                    _click_xpath("//*[@id='featured-book-view-details-btn-1' or starts-with(@id,'featured-book-view-details-btn')]"),
                    _click_xpath("//*[@id='reading-list-button']"),
                ],
                "use_case": "ADD_TO_READING_LIST",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login with username: <username> and password: <password>. Remove 'The Iliad' from your reading list.",
                "actions": [
                    _navigate(),
                    *_login_steps(),
                    _click_xpath("//*[@id='featured-book-view-details-btn-1' or starts-with(@id,'featured-book-view-details-btn')]"),
                    _click_xpath("//a[contains(@href,'/wishlist')]"),
                    _click_xpath("//*[@id='remove-from-wishlist-button']"),
                ],
                "use_case": "REMOVE_FROM_READING_LIST",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login with username: <username> and password: <password>. After logging in, view your shopping cart.",
                "actions": [
                    _navigate("http://localhost:8001/?seed=5"),
                    *_login_steps(),
                    _click_xpath("//a[contains(@href,'/cart')]"),
                ],
                "use_case": "VIEW_CART_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login with username: <username> and password: <password>. After logging in, add 'Romeo and Juliet' to your shopping cart.",
                "actions": [
                    _navigate("http://localhost:8001/search?search=fourth&seed=5"),
                    *_login_steps(),
                    _click_xpath("//a[contains(@href,'/search')]"),
                    _click_xpath("//*[@id='search-field' or @id='search-input']"),
                    _type_xpath("//*[@id='search-field' or @id='search-input']", "__SEARCH_QUERY__"),
                    _click_xpath("//*[@id='view-book-details-button' or @id='spotlight-view-details-btn' or starts-with(@id,'featured-book-view-details-btn')]"),
                    _click_xpath("//*[@id='add-cart-button' or @id='cart-button' or @id='add-to-cart-detail-button']"),
                ],
                "use_case": "ADD_TO_CART_BOOK",
                "has_success": True,
            },
            {
                "url": AUTOBOOKS_URL,
                "prompt": "Login with username: <username> and password: <password>. After logging in, remove 'Romeo and Juliet' from your shopping cart.",
                "actions": [
                    _navigate(),
                    *_login_steps(),
                    _click_xpath("//a[contains(@href,'/cart')]"),
                    _click_xpath("//*[@id='delete-cart-item-button' or @id='remove-from-cart-button' or contains(@id,'remove-cart')]"),
                ],
                "use_case": "REMOVE_FROM_CART_BOOK",
                "has_success": True,
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


def _extract_search_query(prompt: str) -> str:
    forbidden_query = _extract_prompt_value(
        prompt,
        [
            r"query\s*(?:not_equals|!=)\s*['\"]([^'\"]+)['\"]",
        ],
    )

    patterns = [
        r"(?:search\s+for|find)\s+(?:the\s+)?(?:movie|film|book)\s*['\"]([^'\"]+)['\"]",
        r"(?:movie|film|book)(?:_name)?[^'\"]{0,40}(?:equals|contains|is)\s*['\"]([^'\"]+)['\"]",
        r"(?:title|name)\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
        r"query\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]",
    ]
    extracted = _extract_prompt_value(prompt, patterns)
    if extracted and (
        not forbidden_query or _norm_key(extracted) != _norm_key(forbidden_query)
    ):
        return extracted

    if forbidden_query:
        for candidate in ("book", "the", "alpha", "a"):
            if _norm_key(candidate) != _norm_key(forbidden_query):
                return candidate

    generic = re.findall(r"['\"]([^'\"]+)['\"]", str(prompt or ""))
    if generic:
        generic_value = str(generic[0]).strip()[:120]
        if not forbidden_query or _norm_key(generic_value) != _norm_key(forbidden_query):
            return generic_value
        for candidate in ("book", "the", "alpha", "a"):
            if _norm_key(candidate) != _norm_key(forbidden_query):
                return candidate
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


def _apply_prompt_overrides(actions: List[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    commenter_name = _extract_prompt_value(
        prompt,
        [
            r"(?:by\s+(?:the\s+)?)?commenter[_ ]name\s*(?:equals|=|is|:)?\s*['\"]([^'\"]+)['\"]",
            r"name\s+['\"]([^'\"]+)['\"]",
        ],
    )
    username = _extract_prompt_value(prompt, [r"username\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
    email = _extract_prompt_value(prompt, [r"email\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
    password = _extract_prompt_value(prompt, [r"password\s*(?:equals|=|is|:)\s*['\"]([^'\"]+)['\"]"])
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
    year_select_xpath = "//*[@id='library']//select[.//option[contains(translate(normalize-space(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'all years')]]"
    year_option_xpath = (
        f"({year_select_xpath}/option[@value='{prompt_year}'] | {year_select_xpath}/option[2])[1]"
        if prompt_year
        else f"{year_select_xpath}/option[2]"
    )

    out: List[Dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        updated = dict(action)
        selector = updated.get("selector") if isinstance(updated.get("selector"), dict) else {}
        selector_value = str(selector.get("value") or "").strip().lower()
        action_type = str(updated.get("type") or "").strip()

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

        if action_type == "TypeAction":
            raw_text = str(updated.get("text") or "")
            if raw_text == "__AUTHOR__":
                updated["text"] = prompt_author or "book"
            elif raw_text == "__GENRE__":
                updated["text"] = prompt_genre or "Fiction"
            elif raw_text == "__RATING__":
                updated["text"] = prompt_rating or "5.0"
            elif raw_text == "__PAGES__":
                updated["text"] = prompt_pages or "500"
            elif raw_text == "__SEARCH_QUERY__":
                updated["text"] = search_query or "a"
            if selector_value == "comment-name-input" and commenter_name:
                updated["text"] = commenter_name
            elif selector_value == "comment-message-textarea":
                updated["text"] = content_text
            elif selector_value in {"login-username-input", "register-username-input", "username-input"} and username:
                updated["text"] = username
            elif selector_value in {"register-email-input", "signup-email-input"} and email:
                updated["text"] = email
            elif selector_value in {
                "login-password-input",
                "register-password-input",
                "register-confirm-password-input",
                "password-input",
                "confirm-password-input",
            } and password:
                updated["text"] = password
            elif selector_value in {"input", "search-input"} and search_query:
                updated["text"] = search_query
        out.append(updated)
    return out


def get_trajectory_bootstrap_actions(
    *,
    web_project_id: str = "",
    use_case: str = "",
    prompt: str = "",
    max_actions: int = 8,
) -> List[Dict[str, Any]]:
    wanted_project_keys = _project_keys(web_project_id)
    wanted_use_case = str(use_case or "").strip().lower()
    prompt_terms = {t for t in re.findall(r"[a-zA-Z0-9_]{4,}", str(prompt or "").lower())}
    best_score = -1
    best_actions: List[Dict[str, Any]] = []

    for project in TRAJECTORIES:
        if not isinstance(project, dict):
            continue
        project_keys = _project_keys(str(project.get("project_id") or ""))
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
                best_actions = [dict(a) for a in actions if isinstance(a, dict)]

    adapted_actions = _apply_prompt_overrides(best_actions, prompt=prompt)
    return adapted_actions[: max(1, int(max_actions))]


__all__ = ["get_trajectory_examples", "get_trajectory_bootstrap_actions"]
