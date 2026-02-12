#
#  ISC License
#
#  Copyright (c) 2026, Norwegian University of Science and Technology (NTNU)
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

import numpy as np
import datetime as dt
import re
from sgp4.api import Satrec, jday
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
from astropy import units as u
from astropy.time import Time
from dataclasses import dataclass, field
from matplotlib.dates import SEC_PER_DAY
from Basilisk.utilities import orbitalMotion as om

SPUTNIK_LAUNCHDATE = dt.datetime(1957, 10, 4, 0, 0, 0)

@dataclass
class TleData:
    """
    Data class to hold TLE data (orbit elements) and metadata.
    """
    # Required (Orbital elements and time when they are valid is the minimum requirement)
    oe: om.ClassicElements
    tleEpoch: dt.datetime

    # Optional
    satName: str = field(default="BSK-Sat-00") # Satellite name [str] (max 24 characters)
    noradID: str = field(default="00000") # Satellite catalog number / NORAD ID [str] (5 digits)
    classification: str = field(default="Unknown") # Satellite classification ("Unclassified", "Classified", "Secret" | default: "Unknown")
    revAtEpoch: int = field(default=0.0) # Revolution number at epoch [int] (indicates how many orbits the satellite has completed at epoch time (since launch), tops out at 99999)
    propagator: str = field(default="0") # Ephemeris type/propagator (0: "Unknown", 1: "SGP4", 2: "SDP4", 3: "SGP8", 4: "SDP8" | default: "Unknown")
    elemSetNo: int = field(default=999) # Element set number [int] (incremented for each newly created TLE for the same object, tops out at 999)
    nDot: float = field(default=0.0) # First derivative of mean motion "dn/dt" in [rev/day^2]
    nDotDot: float = field(default=0.0) # Second derivative of mean motion "d^2n/dt^2" in [rev/day^3]
    bStar: float = field(default=0.0) # B*, the drag term in [1/Earth radii]
    launchDate: dt.datetime = field(default=None) # Launch date of the satellite (datetime object)
    launchNo: int = field(default=1) # Launch number of the year [int]
    pol: str = field(default="A") # Piece of launch  [str] (max 3 characters)
    def __setattr__(self, name, value):
        # Only allow setting attributes that are defined in the dataclass
        if name not in self.__dataclass_fields__:
            raise AttributeError(f"Cannot set attribute '{name}' on TleData instance")
        object.__setattr__(self, name, value)


# Helper functions: Coordinate conversion; TEME -> J2000/ICRS
def _teme2j2000(r_m, v_m, epoch: dt.datetime):
    t = Time(epoch)
    r_teme = CartesianRepresentation(r_m * u.m)
    v_teme = CartesianDifferential(v_m * u.m / u.s)
    teme = TEME(r_teme.with_differentials(v_teme), obstime=t)
    gcrs = teme.transform_to(GCRS(obstime=t))
    return gcrs.cartesian.xyz.to(u.m).value, gcrs.cartesian.differentials['s'].d_xyz.to(u.m / u.s).value

# Helper functions: Coordinate conversion; J2000/ICRS -> TEME
def _j20002teme(r_m, v_m, epoch: dt.datetime):
    t = Time(epoch)
    r_gcrs = CartesianRepresentation(r_m * u.m)
    v_gcrs = CartesianDifferential(v_m * u.m / u.s)
    gcrs = GCRS(r_gcrs.with_differentials(v_gcrs), obstime=t)
    teme = gcrs.transform_to(TEME(obstime=t))
    return teme.cartesian.xyz.to(u.m).value, teme.cartesian.differentials['s'].d_xyz.to(u.m / u.s).value

# Helper function to wrap angles to [0, 360) degrees
def _wrap_deg(angle_rad: float) -> float:
    return np.rad2deg(angle_rad) % 360.0

def _calcTleChecksum(stringArray: str) -> int:
    """
    Calculate TLE checksum per line.

    :param stringArray: array of strings to calculate checksum for a TLE line
    :return: checksum value (0-9) [-]
    """
    total = 0
    for elem in stringArray:
        for char in elem:
            if char.isdigit():
                total += int(char)
            elif char == "-":
                total += 1
    return total % 10

def _parseTleYear(yearStr: str) -> int:
    """Convert 2-digit TLE year to full 4-digit year."""
    year = int(yearStr)
    #First satellite launched & cataloged in 1957 (Sputnik 1) | This has to be updated in 2057
    return 1900 + year if year >= int(SPUTNIK_LAUNCHDATE.year) % 100 else 2000 + year

def _firstDeriv2ballistCoef(balisticCoeffStr: str) -> float:
    """
    Convert the first derivative of mean motion to physical units.

    :param balisticCoeffStr: string representing the first derivative of mean motion
    :return: first derivative of mean motion in [rev/day^2]
    """
    sign = -1 if balisticCoeffStr.startswith('-') else 1
    return sign * float(balisticCoeffStr[1:])

def _tleStr2physVal(tleStr: str) -> float:
    """
    Convert either the second derivative of mean motion or the B* drag term to physical units.

    :param: string representing either: second derivative of mean motion OR B* drag term
    :return: second derivative of mean motion in [rev/day^3] OR B* drag in [1/Earth radii]
    """
    sign = -1 if tleStr[0:1].startswith('-') else 1
    expSign = -1 if tleStr[6:7].startswith('-') else 1
    mantissa = float(("0." + tleStr[1:6]).strip())
    exponent = float(tleStr[7:8].strip())
    return sign * mantissa * 10 ** (expSign * exponent)

def _parseTle(tle: str) -> TleData:
    """
    Convert a single TLE to classical orbital elements.

    :param tle: a string representing either a TLE(3LE) or a 2LE
    :return: elements: classical orbital elements
             metadata: dictionary containing TLE metadata
    """
    # Check if TLE has two or three lines
    if len(tle) == 3:
        line0 = tle[0]
        line1 = tle[1]
        line2 = tle[2]
    elif len(tle) == 2:
        # If there are two lines (no title line), set line0 to empty
        line0 = ''
        line1 = tle[0]
        line2 = tle[1]
    elif len(tle) > 3:
        raise ValueError(f"_parseTle() received tle with {len(tle)} lines. Most likely a concatenation of multiple TLEs. \n Use constTLE2Elem() for satellite constellations.")
    else:
        raise ValueError(f"_parseTle() received tle with {len(tle)} lines. The TLE should have 2 or 3 lines.")

    # Extract orbital elements from the first TLE-line
    line1_str = line1[0:1]
    if line1_str != '1':
        raise ValueError(f"_parseTle() received line1 starting with '{line1_str}'. The first character of line1 should be '1'.")
    satId1_str = line1[2:7] # Satellite catalog number
    class_str = line1[7:8] # Classification (U=Unclassified, C=Classified, S=Secret)
    launchYear_str = line1[9:11] # Launch year
    launchNo_str = line1[11:14] # Launch number of the year
    pol_str = line1[14:17] # Piece of launch
    epochYear_str = line1[18:20] # Last two digits of the year
    epochDay_str = line1[20:32] # Day of the year incl fractional part [DDD.DDDDDDDD]
    nDot_str = line1[33:43] # String containint the first derivative of the mean motion
    nDotDot_str = line1[44:52] # String containing the second derivative of the mean motion (decimal point assumed)
    bStar_str = line1[53:61]
    ephemType_str = line1[62:63]
    elemSetNo_str = line1[64:68]
    checkSumLine1 = int(line1[68:69])
    calcChecksum1 = _calcTleChecksum([line1_str, satId1_str, class_str, launchYear_str,
                                           launchNo_str, pol_str,
                                           epochYear_str, epochDay_str, nDot_str,
                                           nDotDot_str, bStar_str, ephemType_str,
                                           elemSetNo_str])
    if checkSumLine1 != calcChecksum1:
        raise ValueError(f"_parseTle() checksum mismatch for line1. The calculated checksum is {calcChecksum1} while the provided checksum is {checkSumLine1}.")

    # Extract orbital elements from the second TLE-line
    line2_str = line2[0:1]
    if line2_str != '2':
        raise ValueError(f"_parseTle() received line2 starting with '{line2_str}'. The first character of line2 should be '2'.")
    satID_2_str = line2[2:7].strip()
    if satID_2_str != satId1_str:
        raise ValueError(f"_parseTle() received mismatched satellite ID numbers in line1 ('{satId1_str}') and line2 ('{satID_2_str}').")
    inclDeg_str = line2[8:16] # Inclination in degrees
    OmegaDeg_str = line2[17:25] # Right Ascension of Ascending Node in degrees
    e_str = line2[26:33] # Eccentricity (decimal point assumed)
    omega_str = line2[34:42] # Argument of Perigee in degrees
    meanAnomaly_str = line2[43:51] # Mean Anomaly in degrees
    meanMotion_str = line2[52:63] # Mean Motion in revolutions per day
    revNo_str = line2[63:68] # Revolution number at epoch
    checkSumLine2 = int(line2[68:69])
    calcChecksum2 = _calcTleChecksum([line2_str, satID_2_str, inclDeg_str, OmegaDeg_str, e_str, omega_str, meanAnomaly_str, meanMotion_str, revNo_str])
    if checkSumLine2 != calcChecksum2:
        raise ValueError(f"_parseTle() checksum mismatch for line2. The calculated checksum is {calcChecksum2} while the provided checksum is {checkSumLine2}.")

    # Parsing numerical values getting
    satName = line0.strip()
    noradId = int(satId1_str)
    classification = {'U': 'Unclassified', 'C': 'Classified', 'S': 'Secret'}.get(class_str, 'Unknown')

    epochYear = _parseTleYear(line1[18:20])
    tle_datetime = dt.datetime(epochYear, 1, 1) + dt.timedelta(days=float(epochDay_str) - 1) # subtract one day because Jan 1 is day 1 (not day 0)
    tle_datetime = tle_datetime.replace(tzinfo=dt.timezone.utc)

    # Converting TLE strings to physical values
    nDot = _firstDeriv2ballistCoef(nDot_str) # dn/dt in [rev/day^2]
    nDotDot = _tleStr2physVal(nDotDot_str) # d^2n/dt^2 in [rev/day^3]
    bStar = _tleStr2physVal(bStar_str)
    propagator = {'1': 'SGP4', '2': 'SDP4', '3': 'SGP8', '4': 'SDP8', '0': 'Unknown'}.get(ephemType_str, 'Unknown')
    elemSetNo = int(elemSetNo_str)

    # Converting TLE information to classical orbital elements and write to TleData class
    oe = om.ClassicElements()
    oe.i = np.deg2rad(float(inclDeg_str.strip()))
    oe.Omega = np.deg2rad(float(OmegaDeg_str.strip()))
    oe.e = float(("0." + e_str).strip())
    oe.omega = np.deg2rad(float(omega_str.strip()))
    revDay = float(meanMotion_str.strip())
    oe.a = (om.MU_EARTH / ((revDay * 2.0 * np.pi) / SEC_PER_DAY) ** 2) ** (1.0 / 3.0) * 1e3
    M = np.deg2rad(float(meanAnomaly_str.strip()))
    oe.f = om.E2f(om.M2E(M, oe.e), oe.e)

    tleData = TleData(oe=oe, tleEpoch=tle_datetime)
    # Optional data
    tleData.satName = satName
    tleData.noradID = noradId
    tleData.classification = classification
    tleData.revAtEpoch = float(revNo_str.strip())
    tleData.propagator = propagator
    tleData.elemSetNo = elemSetNo
    tleData.nDot = nDot
    tleData.nDotDot = nDotDot
    tleData.bStar = bStar
    tleData.launchDate = dt.datetime(_parseTleYear(launchYear_str), 1, 1)
    tleData.launchNo = int(launchNo_str)
    tleData.pol = pol_str

    return tleData

def _convertMean2osculating(line1: str, line2: str, tleData: TleData) -> om.ClassicElements:
    """
    Convert the TLE mean orbital elements (NORAD/SGP4) to osculating orbital elements used by basilisk.

    spg4MeanOE -> SPG4 -> state vector -> osculating OE

    :param tleDataList: list of TleData objects containing the mean orbital elements from the TLE
    :return: tleDataList with updated osculating orbital elements used by basilisk
    """
    # Get initial state vector from SGP4 mean orbital elements
    satellite = Satrec.twoline2rv(line1, line2)

    # Convert epoch to Julian date
    epoch = tleData.tleEpoch
    jd, fr = jday(epoch.year, epoch.month, epoch.day,
                  epoch.hour, epoch.minute,
                  epoch.second + epoch.microsecond / 1e6)

    # Propagate to epoch to get True Equator, Mean Equinox (TEME) state vector
    e, r, v = satellite.sgp4(jd, fr)

    if e != 0:
        raise ValueError(f"SPG4 propagation failed for satellite {tleData.satName} with NORAD ID {tleData.noradID} at epoch {tleData.tleEpoch}. Error code: {e}")

    # Convert km -> m for Basilisk
    r_teme_m = np.array(r) * 1e3
    v_teme_m = np.array(v) * 1e3

    # Convert TEME -> J2000/ICRF (Basilisk inertial frame)
    r_m, v_m = _teme2j2000(r_teme_m, v_teme_m, tleData.tleEpoch)

    # Convert state vector to osculating orbital elements
    osculatingOE = om.rv2elem(om.MU_EARTH*1e9, r_m, v_m)

    return osculatingOE

def _osculating2mean_j2(oe_osc: om.ClassicElements) -> om.ClassicElements:
    """
    Convert osculating to mean elements using first-order J2 mapping.

    :param oe_osc: osculating classical orbital elements
    :return: mean classical orbital elements
    """
    oe_mean = om.ClassicElements()
    om.clMeanOscMap(om.REQ_EARTH*1e3, om.J2_EARTH, oe_osc, oe_mean, sign=-1)
    return oe_mean

def _osculating2mean_sgp4(tleData: TleData, tol: float = 0.1, max_iter: int = 10) -> om.ClassicElements:
    """
    Convert osculating elements to SGP4-compatible mean elements via iteration.
    """
    oe_osc = tleData.oe
    epoch = tleData.tleEpoch

    # Start with J2 approximation as initial guess
    oe_mean = _osculating2mean_j2(oe_osc)

    jd, fr = jday(epoch.year, epoch.month, epoch.day,
                  epoch.hour, epoch.minute,
                  epoch.second + epoch.microsecond / 1e6)

    # Target state
    r_target, v_target = om.elem2rv(om.MU_EARTH * 1e9, oe_osc)
    r_target = np.array(r_target)

    for _ in range(max_iter):
        # Generate TLE from current guess and propagate
        tle_str = _generateTleFromMean(oe_mean, tleData)
        lines = tle_str.split('\n')
        sat = Satrec.twoline2rv(lines[1], lines[2])
        e, r_sgp4, v_sgp4 = sat.sgp4(jd, fr)

        if e != 0:
            break

        r_sgp4_m = np.array(r_sgp4) * 1e3
        v_sgp4_m = np.array(v_sgp4) * 1e3
        r_sgp4_j2000, v_sgp4_j2000 = _teme2j2000(r_sgp4_m, v_sgp4_m, epoch)

        # Check convergence
        pos_error = np.linalg.norm(np.array(r_target) - r_sgp4_j2000)
        if pos_error < tol:
            break

        # Compute correction
        oe_sgp4 = om.rv2elem(om.MU_EARTH * 1e9, r_sgp4_j2000, v_sgp4_j2000)

        # Update mean elements
        oe_mean.a += (oe_osc.a - oe_sgp4.a)
        oe_mean.e += (oe_osc.e - oe_sgp4.e)
        oe_mean.i += (oe_osc.i - oe_sgp4.i)
        oe_mean.Omega += (oe_osc.Omega - oe_sgp4.Omega)
        oe_mean.omega += (oe_osc.omega - oe_sgp4.omega)
        oe_mean.f += (oe_osc.f - oe_sgp4.f)

    return oe_mean

def _generateTleFromMean(oe_mean: om.ClassicElements, tleData: TleData) -> str:
    """
    Generate TLE string directly from mean elements (no conversion).
    """
    Norad_str = f"{tleData.noradID:05d}"
    epoch_str = f"{tleData.tleEpoch:%y%j}" + f"{(tleData.tleEpoch.hour * 3600 + tleData.tleEpoch.minute * 60 + tleData.tleEpoch.second + tleData.tleEpoch.microsecond / 1e6) / SEC_PER_DAY:.8f}"[1:]
    classification = tleData.classification[0].upper() if tleData.classification[0].upper() in ['U', 'C', 'S'] else 'U'

    # Format these separately to avoid slicing issues
    IntD_LYear = f"{tleData.launchDate.year % 100:02d}"
    IntD_LNo = f"{tleData.launchNo:03d}"
    pol_str = f"{tleData.pol:<3}"[:3]
    b_star_str = _str2tleFormat(tleData.bStar)

    # Line 1 - properly formatted
    line1_content = f"1 {Norad_str}{classification} {IntD_LYear}{IntD_LNo}{pol_str} {epoch_str}  .00000000  00000-0 {b_star_str} 0  {tleData.elemSetNo:03d}"
    checksum1 = _calcTleChecksum([line1_content])
    line1 = f"{line1_content}{checksum1}"

    # Line 2
    a_km = oe_mean.a / 1000.0
    n_revday = np.sqrt(om.MU_EARTH / a_km ** 3) * SEC_PER_DAY / (2.0 * np.pi)
    M = om.E2M(om.f2E(oe_mean.f, oe_mean.e), oe_mean.e)

    line2_content = f"2 {Norad_str} {np.rad2deg(oe_mean.i):8.4f} {_wrap_deg(oe_mean.Omega):8.4f} {int(oe_mean.e * 1e7):07d} {_wrap_deg(oe_mean.omega):8.4f} {np.rad2deg(M) % 360.0:8.4f} {n_revday:11.8f}{int(tleData.revAtEpoch):05d}"
    checksum2 = _calcTleChecksum([line2_content])
    line2 = f"{line2_content}{checksum2}"

    return f"{tleData.satName}\n{line1}\n{line2}"

def satTle2elem(tle_path: str):
    """
    Convert the TLEs of a constellation to classical orbital elements for each satellite.

    :param tle_path: path to a TLE with one or multiple satellites in either 2LE or 3LE format
    :return: elementsList: list of classical orbital elements for each satellite
             metadataList: list of dictionaries containing TLE metadata for each satellite
    """
    def resetTleStr() -> None:
        """
        Reset the TLE strings and flags for reading a new satellite.
        """
        nonlocal satTLE, satTleReady, readingOrderIndex
        readingOrderIndex = 0
        if type == 2:
            satTLE = ['', '']
            satTleReady = [False, False] # flags to check if all three lines are present
        elif type == 3:
            readingOrderIndex = 0
            satTLE = ['', '', '']
            satTleReady = [False, False, False] # flags to check if all three lines are present

    tle_lines = open(tle_path).readlines()
    noLines = len(tle_lines)
    if noLines < 2:
        raise ValueError(f"constTLE2Elem() received a TLE with {noLines} lines. A valid TLE must have at least two lines.")
    noOfLinesStartingWith1 = sum(1 for line in tle_lines if line.startswith('1'))
    noOfLinesStartingWith2 = sum(1 for line in tle_lines if line.startswith('2'))
    noOfLinesStartingWithOther = len(tle_lines) - noOfLinesStartingWith1 - noOfLinesStartingWith2

    # Initialize variables
    tleDataList = []

    readingOrderIndex = 0

    # Basic checks on the TLE input format (Robustness)
    if noOfLinesStartingWith1 != noOfLinesStartingWith2:
        raise ValueError(f"constTLE2Elem() received a TLE with {noOfLinesStartingWith1} lines starting with '1' and {noOfLinesStartingWith2} lines starting with '2'. The number of lines starting with '1' and '2' must be equal.")
    if noOfLinesStartingWithOther >= noOfLinesStartingWith1:
        #TLE contains at least as many titles as lines starting with '1' or '2'=> 3LE datatype
        type = 3 # 3LE
    elif noOfLinesStartingWithOther == 0 and noLines >= 2:
        # TLE has no title lines at least two or more lines => 2LE
        type = 2 # 2LE
    elif noOfLinesStartingWithOther == 1 and noLines > 2:
        # TLE has one title line and more than two lines => 2LE (first line likely title for 2LE-file)
        type = 2 # 2LE
        if tle_lines[0].startswith('1') or tle_lines[0].startswith('2'):
            raise ValueError("constTLE2Elem() received a TLE with one title line but the first line starts with '1' or '2'. --- TLE not valid ---.")

    # Define the reading order based on the TLE type
    if type == 2:
        readingOrder = ['line1', 'line2']
        satTLE = ['', '']
        satTleReady = [False, False] # flags to check if all three lines are present
    elif type == 3:
        readingOrder = ['title', 'line1', 'line2']
        satTLE = ['', '', '']
        satTleReady = [False, False, False] # flags to check if all three lines are present
    else:
        # This point should be unreachable, kept here for Robustness
        raise ValueError(f"constTLE2Elem() received an invalid TLE type '{type}'. The type must be '2LE' or '3LE'.")

    resetTleStr()
    # Read through all lines and extract TLEs
    for lineNo, line in enumerate(tle_lines):
        expected = readingOrder[readingOrderIndex] if readingOrderIndex < len(readingOrder) else None
        if line.startswith('1') and expected == 'line1':
            if satTleReady[1]:
                print(f"WARNING: constTLE2Elem() found a new TLE line1 before completing the previous TLE (line number {lineNo}). The previous TLE will be discarded.")
                # Reset for next TLE
                resetTleStr()
            # First TLE-line
            satTLE[type-2] = line
            satTleReady[type-2] = True
            readingOrderIndex += 1
            line1 = line
        elif line.startswith('2') and expected == 'line2':
            if satTleReady[type-1]:
                print(f"WARNING: constTLE2Elem() found a new TLE line2 before completing the previous TLE (line number {lineNo}). The previous TLE will be discarded.")
                # Reset for next TLE
                resetTleStr()
            # Second TLE-line
            satTLE[type-1] = line
            satTleReady[type-1] = True
            line2 = line
        elif expected == 'title' and type == 3:
            if satTleReady[0]:
                print(f"WARNING: constTLE2Elem() found a new TLE title line before completing the previous TLE (line number {lineNo}). The previous TLE will be discarded.")
                # Reset for next TLE
                resetTleStr()
            # Header (satellite ID / satellite name)
            satTLE[0] = line
            satTleReady[0] = True
            readingOrderIndex += 1
        else:
            # Line does not match expected format
            print(f"WARNING: constTLE2Elem() found an unexpected line: {line.strip()}, line number {lineNo}. This line will be ignored. The previous TLE will be discarded.")
            continue

        if all(satTleReady):
            tleDataClass = _parseTle(satTLE)
            oscOE = _convertMean2osculating(line1, line2, tleDataClass)
            tleDataClass.oe = oscOE
            tleDataList.append(tleDataClass)
            # Reset for next TLE
            resetTleStr()
    return tleDataList

def _str2tleFormat(value: float) -> str:
    """
    Convert one of the following numbers:
        - "Second derivative of mean motion [rev/day^3]"
        - "B*, the drag term [1/Earth radii]"
    from its numeric value to the TLE format:

    :param value: numeric value to convert
    :return: string in TLE format
    """
    # Sign of the value
    value = value * 10  # Multiply value by 10 to get the exponent right
    if value >= 0:
        sign = " "
    else:
        sign = "-"
    value = abs(value)  # Absolute value
    sci_notation = f"{value:.4e}"
    # Split the scientific notation into mantissa and exponent parts
    parts = sci_notation.split("e")
    matissa = parts[0].replace(".", "")[:5]
    exponentStr = parts[1].lstrip("+")
    exponent = int(exponentStr)
    if exponent > 0:
        expSign = "+"
    else:
        expSign = "-"
    # Format the string according to TLE format
    exponent_str = f"{expSign}{abs(exponent):01d}"
    tle_sub_str = f"{sign}{matissa}{exponent_str}"
    return tle_sub_str

def generateTle(tleData: TleData) -> str:
    """
    Generate TLE based on satellite orbit_n data for one satellite. This function allows to generate a TLE
    for one satellite at a time! TLEs for constellations must be concanatenated outside of this function.
    :param orbitElements: Classical orbital elements of the satellite                    [om.ClassicElements object]
    :param satelliteName: Name of the satellite (optional, default: "BSK-Sat-00")        [string, <= 24 char]
    :param launchYear_dt: Launch date of the satellite (optional, default: current year) [datetime object]
    :param launch_Noyear: Launch number of the year (optional, default: 1)               [int]
    :param NORAD_ID: NORAD ID (5 digits) of the satellite (optional, default: "00000")   [str, 5 char]
    :param classification: Satellite classification (U: Unclassified, C: Classified, S: Secret | default: "U") [str, 1 char]
    :param PoL: Piece of the launch (optional, default: "A  ")                           [str, 3 char]
    :param tleEpoch: Epoch of the TLE as datetime object (optional default: current UTC time) [datetime object]
    :param element_set_no: Element set number (optional, default: 999)                   [int]
    :param nDot: First derivative of mean motion (dn/dt)                                 [float, rev/day^2]
    :param nDotDot: Second derivative of mean motion (d^2n/dt^2)                         [float, rev/day^3]
    :param BStar: B* drag term (optional, default: 0.0)                                  [float, 1/Earth radii]
    :return: TLE string                                                                  [str, 3 lines]
    """

    meanOrbElem = _osculating2mean_sgp4(tleData)

    # Make sure NORAD_ID is an integer and no longer than 5 characters (truncate)
    Norad_str = f"{tleData.noradID:05d}"
    if len(str(tleData.noradID)) > 5:
        print(f"TLE Writing: NORAD_ID truncated to 5 digits: {tleData.noradID}")
    if tleData.launchDate is None:
        tleData.launchDate = dt.datetime.now(dt.timezone.utc)
        print(f"TLE Writing: Launch date not provided, using current UTC time: {tleData.launchDate}")
    if tleData.launchNo is None:
        tleData.launchNo = 1
        print(f"TLE Writing: launch_Noyear not provided, defaulting to: {tleData.launchNo}")
    if tleData.pol is None or len(tleData.pol) > 3:
        tleData.pol = "A"
        print(f"TLE Writing: Piece of the Launch (PoL) was not provided or is invalid, defaulting to 'A  '.")
    if tleData.tleEpoch is None:
        tleData.tleEpoch = dt.datetime.now(dt.timezone.utc)
        print(f"TLE Writing: TLE epoch not provided, using current UTC time: {tleData.tleEpoch.isoformat()}")
    epoch_str = f"{tleData.tleEpoch:%y%j}" + f"{(tleData.tleEpoch.hour * 3600 + tleData.tleEpoch.minute * 60 + tleData.tleEpoch.second + tleData.tleEpoch.microsecond / 1e6) / SEC_PER_DAY:.8f}"[1:]
    if tleData.nDot is None:
        tleData.nDot = 0.0
        print(f"TLE Writing: First derivative of mean motion / balistic coefficient (nDot) not provided, defaulting to: {tleData.nDot}")
    if tleData.nDotDot is None:
        tleData.nDotDot = 0.0
        print(f"TLE Writing: Second derivative of mean motion (nDotDot) not provided, defaulting to: {tleData.nDotDot}")
    if tleData.bStar is None:
        tleData.bStar = 0.0
        print(f"TLE Writing: BStar drag term not provided, defaulting to: {tleData.bStar}")
    tleData.classification = tleData.classification[0].upper() if tleData.classification[0].upper() in ['U', 'C', 'S'] else 'U'
    # Line 0
    line0_str = tleData.satName   # Satellite name (User defined satellite name, Default: "BSK00")
    # Line 1
    line_no_1_str = "1"  # Line number "1"
    satIDStr = Norad_str  # NORAD ID or default
    classStr = tleData.classification  # The satellite is unclassified [Can be U: Unclassified, C: Classified, S: Secret]
    IntD_LYear = f"{tleData.launchDate.year % 100:02d}"  # International Designator (last two digits of launch year) COSPAR ID
    IntD_LNo = f"{int(tleData.launchNo):03d}"  # International Designator (launch no. of the year) COSPAR ID
    IntD_PoL = f"{tleData.pol:<3}"[:3]  # International Designator (piece of the launch) COSPAR ID [default 'A']
    epochStr = epoch_str  # Epoch in TLE format [YYDDD.DDDDDDDD]
    n_dot_str = re.sub(r"-0", "-", f"{tleData.nDot:10.8f}")  # Balistic coeffizient / First derivative of the mean motion in TLE format
    n_dot_str = n_dot_str.replace("0.", " .") if tleData.nDot > 0 else n_dot_str.replace("0.", "-.")  # Remove the decimal point
    n_dotdot_str = _str2tleFormat(
        tleData.nDotDot
    )
    b_star_str = _str2tleFormat(
        tleData.bStar
    )
    ephemeris_str = "0" # Ephemeris type is 0 for BSK
    elem_st_no_str = f"{tleData.elemSetNo:03d}"  # Element set number (incremented for each new TLE for the same object)
    checksum1 = _calcTleChecksum(
        [
            line_no_1_str,
            satIDStr,
            IntD_LYear,
            IntD_LNo,
            epochStr,
            n_dot_str,
            n_dotdot_str,
            b_star_str,
            ephemeris_str,
            elem_st_no_str,
        ]
    )
    line1_str = f"{line_no_1_str} {satIDStr}{classStr} {IntD_LYear}{IntD_LNo}{IntD_PoL} {epochStr} {n_dot_str} {n_dotdot_str} {b_star_str} {ephemeris_str}  {elem_st_no_str}{checksum1}"
    # Line 2
    line_no_2_str = "2"  # Line number "2"
    satcat_str = satIDStr  # Satellite Catalog Number (5 digits) [same as satID]
    i_str = f"{np.rad2deg(meanOrbElem.i):8.4f}"  # Inclination in degrees
    raan_str = f"{_wrap_deg(meanOrbElem.Omega):8.4f}"  # Right Ascension of Ascending Node in degrees
    eccen_str = f"{int(meanOrbElem.e * 1e7):07d}"  # Eccentricity  * 10^7 as decimal value / decimal point assumed
    per_str = f"{_wrap_deg(meanOrbElem.omega):8.4f}"  # Argument of perigee in degrees
    # Calculate M from orbital elements
    M = om.E2M(om.f2E(meanOrbElem.f, meanOrbElem.e), meanOrbElem.e)
    mean_anom_str = f"{np.rad2deg(M) % 360.0:8.4f}"  # Mean anomaly
    # Calculate mean motion in [rev/day^2]
    n = np.sqrt(om.MU_EARTH / (meanOrbElem.a / 1000.0) ** 3) * SEC_PER_DAY / (2.0 * np.pi)
    mean_Mot_str = f"{n:11.8f}"  # Revolutions per day
    rev_str = f"{0:05d}"  # Revolution number at epoch [hardcoded to '00000']
    checksum2 = _calcTleChecksum(
        [
            line_no_2_str,
            satcat_str,
            i_str,
            raan_str,
            eccen_str,
            per_str,
            mean_anom_str,
            mean_Mot_str,
            rev_str,
        ]
    )  # Checksum for line 2 (modulo 10)
    line2_str = f"{line_no_2_str} {satcat_str} {i_str} {raan_str} {eccen_str} {per_str} {mean_anom_str} {mean_Mot_str}{rev_str}{checksum2}"
    # Combine the lines to the full TLE
    tle_str = f"{line0_str}\n{line1_str}\n{line2_str}"
    return tle_str
