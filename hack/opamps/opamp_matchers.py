import logging

from fonduer.candidates.matchers import Intersect, LambdaFunctionMatcher, RegexMatchSpan
from fonduer.utils.data_model_utils import (
    get_page,
    get_row_ngrams,
    get_col_ngrams,
    get_sentence_ngrams,
    get_right_ngrams,
    overlap,
)

logger = logging.getLogger(__name__)


def _first_page_or_table(attr):
    """Must be in the first page, or in a table."""
    return get_page(attr) or attr.sentence.is_tabular()


def _condition(attr):
    if overlap(["condition", "conditions"], get_col_ngrams(attr, n_max=1)):
        return False
    return True


def get_gain_matcher():
    def hertz_units(attr):
        hertz_units = ["mhz", "khz"]
        keywords = ["gain", "unity", "bandwidth", "gbp", "gbw", "gbwp"]
        #  filter_keywords = ["-3 db"]
        related_ngrams = set(get_right_ngrams(attr, n_max=1, lower=True))
        related_ngrams.update(get_row_ngrams(attr, spread=[2, 2], n_max=1, lower=True))
        #
        #  if overlap(filter_keywords, related_ngrams):
        #      return False

        if overlap(hertz_units, related_ngrams) or overlap(keywords, related_ngrams):
            return True

        return False

    # match 3-digit integers, or two-digit floats up with 2 points of precision
    gain_rgx = RegexMatchSpan(
        rgx=r"^(?:\d{1,2}\.\d{1,2}|\d{1,3})$", longest_match_only=False
    )

    hertz_lambda = LambdaFunctionMatcher(func=hertz_units)
    condition_lambda = LambdaFunctionMatcher(func=_condition)
    location_lambda = LambdaFunctionMatcher(func=_first_page_or_table)

    return Intersect(hertz_lambda, gain_rgx, location_lambda, condition_lambda)


def get_supply_current_matcher():
    def current_units(attr):
        current_units = ["ma", "μa", "ua", "a"]
        keywords = ["supply", "quiescent", "is"]
        related_ngrams = set(get_right_ngrams(attr, n_max=1, lower=True))
        related_ngrams.update(get_row_ngrams(attr, spread=[2, 2], n_max=1, lower=True))

        if overlap(current_units, related_ngrams) or overlap(keywords, related_ngrams):
            return True

        return False

    # match 4-digit integers, or two-digit floats up with 2 points of precision
    current_rgx = RegexMatchSpan(
        rgx=r"^(?:\d{1,2}\.\d{1,2}|\d{1,4})$", longest_match_only=False
    )

    current_lambda = LambdaFunctionMatcher(func=current_units)
    condition_lambda = LambdaFunctionMatcher(func=_condition)
    location_lambda = LambdaFunctionMatcher(func=_first_page_or_table)

    return Intersect(current_rgx, current_lambda, condition_lambda, location_lambda)
