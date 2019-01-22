import csv
import logging

from fonduer.candidates.matchers import (
    DictionaryMatch,
    Intersect,
    LambdaFunctionMatcher,
    RegexMatchSpan,
    Union,
)

logger = logging.getLogger(__name__)

DICT_PATH = "data/digikey_part_dictionary.csv"


def _get_digikey_parts_set(path):
    """
    Reads in the digikey part dictionary and yeilds each part.
    """
    all_parts = set()
    with open(path, "r") as csvinput:
        reader = csv.reader(csvinput)
        for line in reader:
            (part, url) = line
            all_parts.add(part)
    return all_parts


def _common_prefix_length_diff(str1, str2):
    for i in range(min(len(str1), len(str2))):
        if str1[i] != str2[i]:
            return min(len(str1), len(str2)) - i
    return 0


def _part_file_name_conditions(attr):
    file_name = attr.sentence.document.name
    if len(file_name.split("_")) != 2:
        return False
    if attr.get_span()[0] == "-":
        return False
    name = attr.get_span().replace("-", "")
    return (
        any(char.isdigit() for char in name)
        and any(char.isalpha() for char in name)
        and _common_prefix_length_diff(file_name.split("_")[1], name) <= 2
    )


def _get_temp_matcher():
    """Return the temperature matcher."""
    return RegexMatchSpan(rgx=r"(?:[1][5-9]|20)[05]", longest_match_only=False)


def _get_part_matcher():
    """Return the part matcher."""
    # Transistor Naming Conventions as Regular Expressions
    eeca_rgx = (
        r"([ABC][A-Z][WXYZ]?[0-9]{3,5}(?:[A-Z]){0,5}[0-9]?[A-Z]?"
        r"(?:-[A-Z0-9]{1,7})?(?:[-][A-Z0-9]{1,2})?(?:\/DG)?)"
    )
    jedec_rgx = r"(2N\d{3,4}[A-Z]{0,5}[0-9]?[A-Z]?)"
    jis_rgx = r"(2S[ABCDEFGHJKMQRSTVZ]{1}[\d]{2,4})"
    others_rgx = (
        r"((?:NSVBC|SMBT|MJ|MJE|MPS|MRF|RCA|TIP|ZTX|ZT|ZXT|TIS|TIPL|DTC|MMBT"
        r"|SMMBT|PZT|FZT|STD|BUV|PBSS|KSC|CXT|FCX|CMPT){1}[\d]{2,4}[A-Z]{0,5}"
        r"(?:-[A-Z0-9]{0,6})?(?:[-][A-Z0-9]{0,1})?)"
    )

    part_dict_matcher = DictionaryMatch(d=_get_digikey_parts_set(DICT_PATH))

    part_rgx = "|".join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx])
    part_rgx_matcher = RegexMatchSpan(rgx=part_rgx, longest_match_only=True)

    add_rgx = r"^[A-Z0-9\-]{5,15}$"
    part_file_name_lambda_matcher = LambdaFunctionMatcher(
        func=_part_file_name_conditions
    )
    part_file_name_matcher = Intersect(
        RegexMatchSpan(rgx=add_rgx, longest_match_only=True),
        part_file_name_lambda_matcher,
    )

    return Union(part_rgx_matcher, part_dict_matcher, part_file_name_matcher)


def get_matcher(name):
    """Return the matcher for the name.

    :param name: One of ["part", "temp", "volt", "current", "power", "polarity"].
    :rtype: Matcher
    """
    if name == "temp":
        return _get_temp_matcher()
    elif name == "part":
        return _get_part_matcher()
    else:
        logger.warning("You did not input a valid name.")
        return None
