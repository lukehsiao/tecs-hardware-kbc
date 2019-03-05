import logging
import re

from fonduer.candidates import MentionNgrams
from fonduer.candidates.models.implicit_span_mention import TemporaryImplicitSpanMention

logger = logging.getLogger(__name__)


class MentionNgramsOpamps(MentionNgrams):
    def __init__(self, n_max=2, split_tokens=["-", "/"]):
        super(MentionNgrams, self).__init__(n_max=n_max, split_tokens=split_tokens)

    def apply(self, doc):
        for ts in MentionNgrams.apply(self, doc):
            m = re.match(
                r"^([\+\-\u2010\u2011\u2012\u2013\u2014\u2212\uf02d])?(\s*)(\d+)$",
                ts.get_span(),
                re.U,
            )
            m2 = re.match(r"^(\d+)\s*\.?\s*(\d+)$", ts.get_span())
            if m:
                if m.group(1) is None:
                    temp = ""
                elif m.group(1) == "+":
                    if m.group(2) != "":
                        # If bigram '+ 150' is seen, accept the unigram '150',
                        # not both
                        continue
                    temp = ""
                else:  # m.group(1) is a type of negative sign
                    # A bigram '- 150' is different from unigram '150', so we
                    # keep the implicit '-150'
                    temp = "-"
                temp += m.group(3)
                yield TemporaryImplicitSpanMention(
                    sentence=ts.sentence,
                    char_start=ts.char_start,
                    char_end=ts.char_end,
                    expander_key="temp_expander",
                    position=0,
                    text=temp,
                    words=[temp],
                    lemmas=[temp],
                    pos_tags=[ts.get_attrib_tokens("pos_tags")[-1]],
                    ner_tags=[ts.get_attrib_tokens("ner_tags")[-1]],
                    dep_parents=[ts.get_attrib_tokens("dep_parents")[-1]],
                    dep_labels=[ts.get_attrib_tokens("dep_labels")[-1]],
                    page=[ts.get_attrib_tokens("page")[-1]]
                    if ts.sentence.is_visual()
                    else [None],
                    top=[ts.get_attrib_tokens("top")[-1]]
                    if ts.sentence.is_visual()
                    else [None],
                    left=[ts.get_attrib_tokens("left")[-1]]
                    if ts.sentence.is_visual()
                    else [None],
                    bottom=[ts.get_attrib_tokens("bottom")[-1]]
                    if ts.sentence.is_visual()
                    else [None],
                    right=[ts.get_attrib_tokens("right")[-1]]
                    if ts.sentence.is_visual()
                    else [None],
                    meta=None,
                )
            elif m2:
                # Handle case that randome spaces are inserted (e.g. "2 . 3")
                if m2.group(1) and m2.group(2):
                    temp = f"{m2.group(1)}.{m2.group(2)}"
                    yield TemporaryImplicitSpanMention(
                        sentence=ts.sentence,
                        char_start=ts.char_start,
                        char_end=ts.char_end,
                        expander_key="temp_expander",
                        position=0,
                        text=temp,
                        words=[temp],
                        lemmas=[temp],
                        pos_tags=[ts.get_attrib_tokens("pos_tags")[-1]],
                        ner_tags=[ts.get_attrib_tokens("ner_tags")[-1]],
                        dep_parents=[ts.get_attrib_tokens("dep_parents")[-1]],
                        dep_labels=[ts.get_attrib_tokens("dep_labels")[-1]],
                        page=[ts.get_attrib_tokens("page")[-1]]
                        if ts.sentence.is_visual()
                        else [None],
                        top=[ts.get_attrib_tokens("top")[-1]]
                        if ts.sentence.is_visual()
                        else [None],
                        left=[ts.get_attrib_tokens("left")[-1]]
                        if ts.sentence.is_visual()
                        else [None],
                        bottom=[ts.get_attrib_tokens("bottom")[-1]]
                        if ts.sentence.is_visual()
                        else [None],
                        right=[ts.get_attrib_tokens("right")[-1]]
                        if ts.sentence.is_visual()
                        else [None],
                        meta=None,
                    )
            else:
                yield ts
