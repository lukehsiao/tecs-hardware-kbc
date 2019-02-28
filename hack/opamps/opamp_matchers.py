import logging

from fonduer.candidates.matchers import Intersect, LambdaFunctionMatcher, RegexMatchSpan
from fonduer.utils.data_model_utils import get_row_ngrams, get_sentence_ngrams

logger = logging.getLogger(__name__)


def get_gain_matcher():
    def hertz_units(attr):
        related_ngrams = set(get_sentence_ngrams(attr))
        related_ngrams.add(get_row_ngrams(attr))

        for ngram in related_ngrams:
            if any(x in ngram.lower() for x in ["hz", "bandwidth", "gain"]):
                return True

        return False

    # match 3-digit integers, or two-digit floats up with 2 points of precision
    gain_rgx = RegexMatchSpan(
        rgx=r"^(?:\d{1,2}\.\d{1,2}|\d{1,3})$", longest_match_only=False
    )

    gain_lambda = LambdaFunctionMatcher(func=hertz_units)

    return Intersect(gain_rgx, gain_lambda)


def get_supply_current_matcher():
    def current_units(attr):
        related_ngrams = set(get_sentence_ngrams(attr))
        related_ngrams.add(get_row_ngrams(attr))

        for ngram in related_ngrams:
            if any(x in ngram.lower() for x in ["a", "current", "supply"]):
                return True

        return False

    # match 4-digit integers, or two-digit floats up with 2 points of precision
    current_rgx = RegexMatchSpan(
        rgx=r"^(?:\d{1,2}\.\d{1,2}|\d{1,4})$", longest_match_only=False
    )

    current_lambda = LambdaFunctionMatcher(func=current_units)

    return Intersect(current_rgx, current_lambda)
