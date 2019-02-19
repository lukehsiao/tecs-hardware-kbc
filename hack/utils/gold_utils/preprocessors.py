#!/usr/bin/env python

import pdb

# TODO: Make a list of all known manuf (in order to standardize them)
KNOWN_MANUF = [
    "ON Semiconductor",
    "Fairchild",
    "Diotec",
    "MCC",
    "Taiwan",
    "Central",
    "Diodes Incorporated",
    "Rohm",
    "Infineon",
    "JCST",
    "KEC",
    "Lite-on",
    "Minilogic",
]

"""
TRANSISTOR PREPROCESSORS:
"""


def preprocess_url(url):
    # Takes in a URL and returns it if it is valid
    if url == "-":
        return "N/A"
    return url  # We use the PDFs URL as the filename as we cannot ensure
    # standard datasheet filenames: allows for convenient file lookup


def preprocess_manuf(manuf):
    if manuf == "-":
        return "N/A"
    if manuf in KNOWN_MANUF:
        return manuf  # TODO: Complete KNOWN_MANUF list
    return manuf


def add_space(type, value):
    value = value.strip()
    if type == "current":
        if value.endswith("nA"):
            return value.replace("nA", " nA")
        elif value.endswith("mA"):
            return value.replace("mA", " mA")
        # Account for exception on line 75 of ffe00114_3.csv
        # See: https://www.digikey.com/products/en?keywords=2SD2704KT146TR-ND
        elif value.endswith("ma"):
            return value.replace("ma", " mA")
        elif value.endswith("A"):
            return value.replace("A", " A")
        else:
            print(f"[WARNING]: Invalid {type} {value}")
            pdb.set_trace()
    elif type == "voltage":
        if value.endswith("mV"):
            return value.replace("mV", " mV")
        elif value.endswith("V"):
            return value.replace("V", " V")
        else:
            print(f"[WARNING]: Invalid {type} {value}")
            pdb.set_trace()
    elif type == "frequency":
        if value.endswith("MHz"):
            return value.replace("MHz", " MHz")
        elif value.endswith("GHz"):
            return value.replace("GHZ", " GHz")
        elif value.endswith("kHz"):
            return value.replace("kHz", " kHz")
        else:
            print(f"[WARNING]: Invalid {type} {value}")
            pdb.set_trace()
    elif type == "power":
        if value.endswith("mW"):
            return value.replace("mW", " mW")
        elif value.endswith("W"):
            return value.replace("W", " W")
        else:
            print(f"[WARNING]: Invalid {type} {value}")
            pdb.set_trace()


def preprocess_dc_gain_min(gain):
    # Takes in a dc_gain_min with Digikey's standard condition syntax
    # (i.e. 200 @ 2mA, 5V)
    # And returns a tuple containing (dc_gain, Ic, Vce) with units
    if gain == "-":
        return "N/A"
    try:
        (dc_gain, conditions) = gain.split("@")
        dc_gain = dc_gain.strip()
        conditions = conditions.strip()
        # Here we also return implied values (found in conditions)
        # (i.e. dc_gain_min @ supply_current) <-- We can extract supply_current
        (implied_supply_current, implied_ce_v_max) = conditions.split(",")

        # Add space between value and unit
        # Account for unit exception on line 163 of ffe00114_15.csv
        # See: https://www.digikey.com/products/en?keywords=2SC2922
        if implied_supply_current.endswith("V"):
            implied_supply_current = add_space("voltage", implied_supply_current)
        else:
            implied_supply_current = add_space("current", implied_supply_current)
        # Account for unit exception on line 262 of ffe00114_38.csv
        # See: https://www.digikey.com/products/en?keywords=BD249C-S-ND
        if implied_ce_v_max.endswith("A"):
            implied_ce_v_max = add_space("current", implied_ce_v_max)
        else:
            implied_ce_v_max = add_space("voltage", implied_ce_v_max)

        # Return final values (formatted as the value, a space, and the unit)
        return (dc_gain, implied_supply_current, implied_ce_v_max)
    except Exception as e:
        print(
            f"[ERROR]: {e} while preprocessing dc current gain min: {gain}"
            + "returning N/A."
        )
        pdb.set_trace()  # TODO: Do we just want to return N/A or pdb?
        return "N/A"


def preprocess_vce_saturation_max(voltage):
    # Takes in a vce_saturation_max with Digikey's standard condition syntax
    # (i.e. 600mV @ 5mA, 100mA)
    # And returns a tuple containing (vce_sat_max, Ib, Ic) with units
    if voltage == "-":
        return "N/A"
    try:
        (vce_sat_max, conditions) = voltage.split("@")
        conditions = conditions.strip()

        # Here we also return implied values (found in conditions)
        # (i.e. dc_gain_min @ supply_current) <-- We can extract supply_current
        (implied_base_current, implied_supply_current) = conditions.split(", ")

        # Add space between the value and unit
        # Account for invalid unit discrepancy in ffe00114_0.csv at line 486
        # See: https://www.digikey.com/products/en?keywords=MJE18008G
        if implied_supply_current.endswith("V"):
            implied_supply_current = add_space("voltage", implied_supply_current)
        else:
            implied_supply_current = add_space("current", implied_supply_current)

        implied_base_current = add_space("current", implied_base_current)
        vce_sat_max = add_space("voltage", vce_sat_max)

        # Return final set
        return (vce_sat_max, implied_base_current, implied_supply_current)
    except Exception as e:
        print(
            f"[ERROR]: {e} while preprocessing collector emitter saturation"
            + f" voltage max: {voltage} returning N/A."
        )
        pdb.set_trace()  # TODO: Do we just want to return N/A or pdb?
        return "N/A"


def preprocess_ce_v_max(voltage):
    # Takes in a ce_v_max with Digikey's standard condition syntax
    # (i.e. 65V)
    # Checks if it's valid and returns the stripped value
    if voltage == "-":
        return "N/A"
    try:
        return add_space("voltage", voltage)
    # TODO: Why do I even have this try and except here??
    except Exception as e:
        print(
            f"[ERROR]: {e} while preprocessing collector emitter"
            + f" voltage max: {voltage} returning N/A."
        )
        pdb.set_trace()  # TODO: Do we just want to return N/A or pdb?
        return "N/A"


def preprocess_c_current_max(current):
    # Takes in a c_current_max with Digikey's standard condition syntax
    # (i.e. 100mA)
    # Checks if it's valid and returns the stripped value
    if current == "-":
        return "N/A"
    # Add space between the value and unit
    return add_space("current", current)


def preprocess_polarity(polarity):
    # Takes in a polarity (i.e. NPN) and returns it if it is valid
    if polarity in ["NPN", "PNP"]:
        return polarity
    elif polarity == "-":
        return "N/A"
    print(f"[ERROR]: Invalid polarity {polarity}")
    pdb.set_trace()
    return "N/A"


def preprocess_c_current_cutoff_max(current):
    # Takes in a c_current_cutoff_max with Digikey's standard condition syntax
    # (i.e. 15nA (ICBO))
    # Checks if it's valid and returns the stripped value
    if current == "-":
        return "N/A"
    current = current.strip("(ICBO)")
    # Add space between value and unit
    return add_space("current", current)


def preprocess_pwr_max(power):
    if power == "-":
        return "N/A"
    return add_space("power", power)


def preprocess_freq_transition(freq):
    if freq == "-":
        return "N/A"
    # Add space between value and unit
    return add_space("frequency", freq)


"""
OPAMP PREPROCESSORS:
TODO: These have not been fully vetted yet
"""


def preprocess_gbp(typ_gpb):
    if typ_gpb == "-":
        return "N/A"
    else:
        return typ_gpb


def preprocess_supply_current(current):
    supply_current = current.replace("Â", "").replace("µ", "u")

    # sometimes digikey reports a random MAX.
    supply_current = supply_current.replace("(Max)", "").strip()

    if supply_current == "-":
        return "N/A"
    else:
        return add_space("current", supply_current)


def preprocess_operating_voltage(voltage):
    # handle strings like:
    #   2.4 V ~ 6 V
    #   4.5 V ~ 36 V, Â±2.25 V ~ 18 V
    #   10 V ~ 36 V, Â±5 V ~ 18 V
    #   4.75 V ~ 5.25 V, Â±2.38 V ~ 2.63 V
    if voltage == "-":
        return ("N/A", "N/A")

    op_volt = voltage.replace("Â", "")
    if "~" not in op_volt and "," in op_volt:
        op_volt = op_volt.replace(",", "~")
    elif "~" not in op_volt:  # for when only a single value is reported
        op_volt = " ~ ".join([op_volt, op_volt])
    ranges = [r.strip() for r in op_volt.split(",")]
    min_set = set()
    max_set = set()
    for r in ranges:
        try:
            (min_val, max_val) = [val.strip() for val in r.split("~")]
            if "/" in min_val or "/" in max_val:
                continue  # -0.9 V/+1.3 V-0.9 V/+1.3 V in ffe002af_13.csv
            if " " not in min_val:
                min_val = min_val[:-1] + " " + min_val[-1:]
            if " " not in max_val:
                max_val = max_val[:-1] + " " + max_val[-1:]
            min_set.add(add_space("voltage", min_val))
            max_set.add(add_space("voltage", max_val))
        except ValueError as e:
            print(
                f"[ERROR]: {e} while preprocessing operating voltage: {voltage}"
                + "returning N/A."
            )
            pdb.set_trace()
            return ("N/A", "N/A")

    return (";".join(min_set), ";".join(max_set))


def preprocess_operating_temp(temperature):
    if temperature == "-" or temperature is None:
        return ("N/A", "N/A")
    # handle strings like:
    #   -20Â°C ~ 75Â°C
    op_temp = temperature.replace("Â", "").replace("°", " ")

    # Deal with temperatures like: 150Â°C (TJ)
    if "~" not in op_temp:  # For values like: 150 C (TJ)
        return (op_temp.strip("(TJ)"), op_temp.strip("(TJ)"))

    try:
        (min_temp, max_temp) = [val.strip() for val in op_temp.split("~")]
        # Add a space in between value and unit:
        (min_temp, max_temp) = (min_temp.strip("(TJ)"), max_temp.strip("(TJ)"))
        return (min_temp, max_temp)
    except ValueError as e:
        print(
            f"[ERROR]: {e} while preprocessing operating temperature range: "
            + f"{temperature} returning N/A."
        )
        pdb.set_trace()
        return ("N/A", "N/A")
