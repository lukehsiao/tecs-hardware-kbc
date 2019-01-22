from fonduer.utils.data_model_utils import is_horz_aligned, is_vert_aligned, same_table


def stg_temp_filter(c):
    (part, attr) = c
    if same_table((part, attr)):
        return is_horz_aligned((part, attr)) or is_vert_aligned((part, attr))
    return True
