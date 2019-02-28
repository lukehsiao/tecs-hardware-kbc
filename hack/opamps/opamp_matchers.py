import logging

from fonduer.candidates.matchers import RegexMatchSpan

logger = logging.getLogger(__name__)


def get_gain_matcher():

    # match 3-digit integers, or two-digit floats up with 2 points of precision
    return RegexMatchSpan(
        rgx=r"^(?:\d{1,2}\.\d{1,2}|\d{1,3})$", longest_match_only=False
    )


def get_supply_current_matcher():

    # match 4-digit integers, or two-digit floats up with 2 points of precision
    return RegexMatchSpan(
        rgx=r"^(?:\d{1,2}\.\d{1,2}|\d{1,4})$", longest_match_only=False
    )
