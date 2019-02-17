#!/usr/bin/env python
import pdb


def split_val_condition(input_string):
    """
    Split and return a {'value': v, 'condition': c} dict for the value
    and the condition.
    Condition is empty if no condition was found.

    @param input    A string of the form XXX @ YYYY
    """
    try:
        (value, condition) = [x.strip() for x in input_string.split("@")]
        return {"value": value, "condition": condition}
    except ValueError:
        # no condition was found
        return {"value": input_string.strip(), "condition": None}


def part_family_normalizer(family):
    if family == "Y":
        return str(family)
    elif family == "N":
        return str(family)
    elif family == "N/A":  # Account for Digikey not having any part_family
        return str(family)
    else:
        print(f"[ERROR]: Invalid part family {family}")
        pdb.set_trace()


def transistor_part_normalizer(part):
    # Part number normalization
    return part.replace(" ", "").upper()


def opamp_part_normalizer(part):
    # TODO: Digikey actually has weird formatting on their part numbers,
    #       which will require more normalization than we had to do with
    #       transistors.
    return part.replace(" ", "").upper()


def gain_bandwidth_normalizer(gbp):
    """
    Normalize the gain bandwidth product into kHz.
    """
    parse = split_val_condition(gbp)

    # NOTE: We currently ignore the conditions

    # Process Units
    try:
        (value, unit) = parse["value"].split(" ")
        value = float(value)
        """
        if unit == "MHz":
            value = value * 1000
        elif unit == "kHz":
            # already kHz
            pass
        """
        return str(value) + " " + unit

    except Exception as e:
        print(f"[ERROR]: {e} on gain bandwidth {parse}")
        pdb.set_trace()


def supply_current_normalizer(supply_current):
    """
    Normalize input quiescent supply current to uA
    """
    # NOTE: Currently ignoring the conditions.
    parse = split_val_condition(supply_current)

    # Process Units
    try:
        (value, unit) = parse["value"].split(" ")
        value = float(value)
        if unit == "mA":
            value = value * 1000
        elif unit == "nA":
            value = value / 1000
        return value
    except Exception:
        print("[ERROR]: " + str(parse))
        pdb.set_trace()


def opamp_voltage_normalizer(supply_voltage):
    """
    Normalize supply voltage into absolute values (remove +/-)
    """
    parse = split_val_condition(supply_voltage)

    try:
        if parse["value"].startswith("± "):
            parse["value"] = parse["value"].replace("± ", "±")
        (value, unit) = parse["value"].split(" ")
    except Exception:
        print("[ERROR]: " + str(parse))
        pdb.set_trace()

    if unit != "V":
        print("[ERROR]: Invalid supply voltage")
        pdb.set_trace()

    return str(value) + " " + unit


"""
GENERAL NORMALIZERS
"""


def general_normalizer(value):
    # TODO: Right now this is only returning the raw values
    return value.strip()


def temperature_normalizer(temperature):
    try:
        (temp, unit) = temperature.rsplit(" ", 1)
        if unit != "C":
            print(f"[ERROR]: Invalid temperature value {temperature}")
            pdb.set_trace()
        return int(temp)
    except Exception as e:
        print(f"[ERROR]: {e} on temperature value {temperature}")
        pdb.set_trace()


def polarity_normalizer(polarity):
    if polarity in ["NPN", "PNP"]:
        return polarity
    print(f"[ERROR]: Incorrect polarity value {polarity}")
    pdb.set_trace()
    return "N/A"


def dissipation_normalizer(dissipation):
    dissipation = dissipation.strip()
    return str(abs(round(float(dissipation.split(" ")[0]), 1)))


def current_normalizer(current):
    current = current.strip()
    return str(abs(round(float(current.split(" ")[0]), 1)))


def voltage_normalizer(voltage):
    voltage = voltage.replace("K", "000")
    voltage = voltage.replace("k", "000")
    return voltage.split(" ")[0].replace("-", "")


def gain_normalizer(gain):
    gain = gain.split("@")[0]
    gain = gain.strip()
    gain = gain.replace(",", "")
    gain = gain.replace("K", "000")
    gain = gain.replace("k", "000")
    return str(abs(round(float(gain.split(" ")[0]), 1)))


def old_dev_gain_normalizer(gain):
    return str(abs(round(float(gain), 1)))
