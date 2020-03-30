from fonduer.utils.data_model_utils import (
    get_cell_ngrams,
    get_col_ngrams,
    get_horz_ngrams,
    get_neighbor_cell_ngrams,
    get_page,
    get_row_ngrams,
    get_vert_ngrams,
    overlap,
)

ABSTAIN = -1
FALSE = 0
TRUE = 1


def neg_low_page_num(c):
    if get_page(c[0]) > 8:
        return FALSE

    return ABSTAIN


def pos_gain(c):
    row_ngrams = set(get_row_ngrams(c.gain, lower=True))
    #     print("row_ngrams", row_ngrams)
    if overlap(["gain"], row_ngrams):
        return TRUE
    else:
        ABSTAIN


def pos_gain_keywords(c):
    vert_ngrams = set(get_vert_ngrams(c.gain, n_max=1, lower=True))
    row_ngrams = set(get_row_ngrams(c.gain, lower=True))
    if overlap(["typ", "typ."], vert_ngrams) and overlap(["khz", "mhz"], row_ngrams):
        return TRUE

    return ABSTAIN


def neg_keywords(c):
    horz_ngrams = set(get_horz_ngrams(c.gain, lower=True))
    if overlap(["bandwidth"], horz_ngrams) and not overlap(["gain"], horz_ngrams):
        return FALSE

    return ABSTAIN


def pos_sen_lf(c):
    if (
        pos_gain(c) == TRUE
        and pos_gain_keywords(c) == TRUE
        and neg_keywords(c) == ABSTAIN
    ):
        return TRUE

    return FALSE


# Gain LFs
def pos_gain_header_unit(c):
    horz_ngrams = set(get_horz_ngrams(c.gain, n_max=1, lower=True))
    vert_ngrams = set(get_vert_ngrams(c.gain, n_max=1, lower=True))
    right_ngrams = set(
        [
            x[0]
            for x in get_neighbor_cell_ngrams(
                c.gain, n_max=1, dist=5, directions=True, lower=False
            )
            if x[-1] == "RIGHT"
        ]
    )
    if (
        overlap(["gain", "unity"], horz_ngrams)
        and overlap(["mhz", "khz"], right_ngrams)
        and overlap(["typ", "typ."], vert_ngrams)
    ):
        return TRUE
    else:
        return ABSTAIN


def neg_gain_bandwidth_missing_gain(c):
    row_ngrams = set(get_row_ngrams(c.gain, n_max=1, lower=True))
    if "bandwidth" in row_ngrams and "gain" not in row_ngrams:
        return FALSE

    return ABSTAIN


def neg_gain_keywords_in_cell(c):
    cell_ngrams = set(get_cell_ngrams(c.gain, n_max=1, lower=True))
    if overlap(["g", "vo", "vpp", "f=", "f", "="], cell_ngrams):
        return FALSE
    else:
        return ABSTAIN


def neg_gain_too_many_words_in_cell(c):
    cell_ngrams = list(get_cell_ngrams(c.gain))
    if len(cell_ngrams) >= 4:
        return FALSE
    else:
        return ABSTAIN


def neg_gain_keywords_in_right_cell(c):
    right_ngrams = set(
        [
            x[0]
            for x in get_neighbor_cell_ngrams(
                c[0], n_max=1, dist=5, directions=True, lower=False
            )
            if x[-1] == "RIGHT"
        ]
    )
    if not overlap(["kHz", "MHz", "GHz"], right_ngrams):
        return FALSE
    else:
        return ABSTAIN


def neg_gain_keywords_in_horz(c):
    horz_ngrams = set(get_horz_ngrams(c.gain, n_max=1, lower=True))
    if overlap(
        [
            "small",
            "full",
            "flat",
            "current",
            "thd",
            "signal",
            "flatness",
            "input",
            "noise",
            "f=",
            "f",
            "-3",
            "power",
            "db",
            "dbm",
            "output",
            "impedence",
            "delay",
            "capacitance",
            "range",
            "ratio",
            "dbc",
            "temperature",
            "common",
            "voltage",
            "range",
        ],
        horz_ngrams,
    ):
        return FALSE
    else:
        return ABSTAIN


def neg_gain_keywords_in_row(c):
    row_ngrams = set(get_row_ngrams(c.gain, n_max=1, lower=True))
    if overlap(
        [
            "small",
            "full",
            "flat",
            "current",
            "thd",
            "signal",
            "flatness",
            "input",
            "noise",
            "f=",
            "f",
            "-3",
            "power",
            "db",
            "dbm",
            "output",
            "impedence",
            "delay",
            "capacitance",
            "range",
            "ratio",
            "dbc",
            "temperature",
            "common",
            "voltage",
            "range",
        ],
        row_ngrams,
    ):
        return FALSE
    else:
        return ABSTAIN


def neg_gain_keywords_in_column(c):
    col_ngrams = set(get_col_ngrams(c.gain, n_max=1, lower=True))
    if overlap(
        [
            "max",
            "min",
            "test",
            "condition",
            "conditions",
            "vgn",
            "f",
            "-3",
            "db",
            "dbc",
        ],
        col_ngrams,
    ):
        return FALSE

    else:
        return ABSTAIN


# Supply Current LFs
def pos_current(c):
    row_ngrams = list(get_row_ngrams(c.supply_current))
    keywords = ["supply", "quiescent", "iq", "is", "idd"]
    return TRUE if overlap(keywords, row_ngrams) else ABSTAIN


def pos_current_units(c):
    row_ngrams = list(get_row_ngrams(c.supply_current))
    current_units = ["ma", "μa", "ua", "µa", "\uf06da"]
    return TRUE if overlap(current_units, row_ngrams) else ABSTAIN


def pos_current_typ(c):
    return (
        TRUE
        if overlap(["typ", "typ."], get_col_ngrams(c.supply_current, lower=True))
        else ABSTAIN
    )


def neg_current_keywords_in_column(c):
    return (
        FALSE
        if overlap(
            ["over", "temperature", "vgn", "f", "-3", "db", "dbc", "min", "max"],
            get_col_ngrams(c.supply_current, lower=True),
        )
        else ABSTAIN
    )


def neg_current_keywords_in_vert(c):
    return (
        FALSE
        if overlap(
            ["over", "temperature", "vgn", "f", "-3", "db", "dbc", "min", "max"],
            get_vert_ngrams(c.supply_current, lower=True),
        )
        else ABSTAIN
    )


def neg_current_keywords_in_row(c):
    return (
        FALSE
        if overlap(
            ["output", "drive", "voltage", "io"],
            get_row_ngrams(c.supply_current, lower=True),
        )
        else ABSTAIN
    )


gain_lfs = [
    #  neg_gain_bandwidth_missing_gain,
    #  neg_gain_keywords_in_cell,
    #  neg_gain_keywords_in_column,
    #  neg_gain_keywords_in_horz,
    #  neg_gain_keywords_in_right_cell,
    #  neg_gain_keywords_in_row,
    #  neg_gain_too_many_words_in_cell,
    #  neg_low_page_num,
    pos_sen_lf
]

current_lfs = [
    neg_current_keywords_in_column,
    neg_current_keywords_in_row,
    neg_current_keywords_in_vert,
    neg_low_page_num,
    pos_current,
    pos_current_typ,
    pos_current_units,
]
