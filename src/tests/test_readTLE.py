import os
import pathlib
import pytest
import numpy as np
import Basilisk.utilities.tleHandling as tleHandling
import Basilisk.utilities.orbitalMotion as om
from datetime import datetime, timedelta, timezone

A_TOL = 1e-14 #[-]
A_TOL_ROUNDTRIP = 1e-4 #[-] Tolerance for eccentricity (roundtrip error)
A_TOL_DEG_ROUNDTRIP = 1.5  # [deg] Tolerance for angles in degrees (roundtrip error)

DATA_DIR = pathlib.Path(__file__).parent / 'data'

###################################################################
# Expected orbital elements for oneWeb (if testing multiple satellites)
EXPECTED_OE_ONE_WEB = {
    'a': [
        7578544.779977489,  # 0
        7581725.286624685,  # 1
        7581725.0644000145, # 2
        7585736.5116564045, # 3
        7585735.969908914, # 4
        7585737.122778924 # 5
    ],
    'e': [
        0.0008817610616498866,   0.001210092720829519, 0.001258632637199018, 0.0011981315070938715, 0.0012069525700065013,
        0.0012398288414982343
    ],
    'i': [
        1.5340969599567005,  1.5341079406333271, 1.534104451987643,
        1.5341619924086083, 1.5341707140511949, 1.5341619924093775
    ],
    'f': [
        2.1381902119590572, 5.002603384280056, 5.0290291055077345,
        5.017007128018255, 5.007173564098952, 5.003260218813585
    ],
    'Omega': [
        4.888906786058248, 4.888930779554231, 4.889318242662984,
        5.421683807074992, 5.420380046148317, 5.420713403989505
    ],
    'omega': [
        1.4381317553505906, 1.2805819480498706, 1.2541565771775638,
        1.2661781036367423, 1.2760122502736482, 1.2799245057654098
    ]
}

EXPECTED_OE_2LE = {
    'a': [
        6802505.886893865,
        6770009.68464781,
        6802528.753304223
    ],
    'e': [
        0.0011264777844962191,
        0.0011530141980256035,
        0.0011288394223626691
    ],
    'i': [
        0.9015068684734451,
        0.724041375492735,
        0.9015086111773605
    ],
    'f': [
        5.428345740894182,
        5.535544343398931,
        5.428801220635774
    ],
    'Omega': [
        2.395785537940196,
        1.632024223580092,
        2.4180821212267007
    ],
    'omega': [
        0.8548384352861554,
        0.747642161264503,
        0.8543853115805702
    ]
}

# Values to generate TLE for HYPSO 1
oeHypso1 = om.ClassicElements()
oeHypso1.i = np.deg2rad(97.31452620946388)
oeHypso1.e = 0.001399608536242 # [-]
oeHypso1.a = (om.RP_EARTH + 451.68298285628407)*1e3 # [m]
oeHypso1.Omega = np.deg2rad(349.0390000036714) # [rad]
oeHypso1.omega = np.deg2rad(82.38650024248433) # [rad]
oeHypso1.f = np.deg2rad(277.61347958955423) # [rad]
hypso1noradId = 51053 # [-]
hypso1launch = datetime(2022, 1, 13, 0, 0, 0) # [UTC]
hypso1tleEpoch = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=279.47924866 - 1) # [UTC]
hypso1nDot = 0.00028852 # [rev/day^2]
hypso1bStar = 0.00056053 # [1/Earth radii]

# Expected orbital elements for Hypso1
EXPECTED_OE_HYPSO = {
    'a': [oeHypso1.a],
    'e': [oeHypso1.e],
    'i': [oeHypso1.i],
    'f': [oeHypso1.f],
    'Omega': [oeHypso1.Omega],
    'omega': [oeHypso1.omega]
}

###################################################################
def _equalCheck(v1, v2, fieldName):
    """Compare two values within tolerance"""
    if abs(v1 - v2) < A_TOL:
        return 0
    print(f'{fieldName} failed: expected {v2}, got {v1}')
    return 1

@pytest.mark.parametrize("tlePath, expectedDict", [
    (DATA_DIR / "hypso1.tle", EXPECTED_OE_HYPSO),
    (DATA_DIR / "oneWeb25.tle", EXPECTED_OE_ONE_WEB),
    (DATA_DIR / "spacestations.2le", EXPECTED_OE_2LE),
])
def test_read_tle(tlePath, expectedDict):
    eCountTle = 0

    # Read TLE file from data folder
    tleDataList = tleHandling.satTle2elem(tlePath)

    # Check each orbital element
    for idx, _ in enumerate(tleDataList):
        for key in expectedDict:
            actualValue = getattr(tleDataList[idx].oe, key)
            eCountTle += _equalCheck(actualValue, expectedDict[key][idx], f'{tlePath}_{key}')

    assert eCountTle < 1, f"{eCountTle} functions failed in tleHandling.py script, satTLE2Elem() method"

@pytest.mark.parametrize("satName, orbitalElements, noradID, launchDate, launch_Noyear, PoL, tleEpoch, nDot, bStar, expectedTlePath", [
    ('HYPSO 1', oeHypso1, hypso1noradId, hypso1launch, 2, 'BX', hypso1tleEpoch, hypso1nDot, hypso1bStar, DATA_DIR / "hypso1.tle"),
])
def test_write_tle(satName, orbitalElements, noradID, launchDate, launch_Noyear, PoL, tleEpoch, nDot, bStar, expectedTlePath):
    eCountTle = 0

    # Make the data class
    tleData = tleHandling.TleData(oe = orbitalElements, tleEpoch = tleEpoch)
    # fill in optional data
    tleData.satName = satName
    tleData.noradID = noradID
    tleData.launchDate = launchDate
    tleData.launchNo = launch_Noyear
    tleData.pol = PoL
    tleData.nDot = nDot
    tleData.nDotDot = 0.0
    tleData.bStar = bStar
    generatedTle = tleHandling.generateTle(tleData)

    # Read the actual TLE file
    with open(expectedTlePath, 'r') as file:
        expectedTLE = file.read()
    # String compaire (line 0)
    eCountTle += int((generatedTle.splitlines()[0] != expectedTLE.splitlines()[0]))
    # String compaire (line 1)
    eCountTle += int((generatedTle.splitlines()[1] != expectedTLE.splitlines()[1]))
    # String compaire (line 2)
    for i, (field_expected, field_generated) in enumerate(zip(expectedTLE.splitlines()[2].split(), generatedTle.splitlines()[2].split())):
        if i in [4]: # 4 = eccentricity (leading decimal point assumed), 5 = argument of perigee, 6 = mean anomaly
            # Eccentricity -> leading decimal point assume -> max tolerance is A_TOL_ROUNDTRIP * 1e7 => 1e-4 physical eccentricity tolerance
            eCountTle += abs(float(field_expected) - float(field_generated)) > A_TOL_ROUNDTRIP * 1e7
        elif i in [5,6]: # 5 = argument of perigee, 6 = mean anomaly (degrees)
            eCountTle += abs(float(field_expected) - float(field_generated)) > A_TOL_DEG_ROUNDTRIP
        else:
            eCountTle += abs(float(field_expected) - float(field_generated)) > A_TOL_ROUNDTRIP

    assert eCountTle < 1, f"{eCountTle} functions failed in tleHandling.py script, generateTleDataString() method"

@pytest.mark.parametrize("tlePath", [
    (DATA_DIR / "hypso1.tle"),
])
def test_read_write_tle(tlePath):
    eCountTle = 0
    # Read TLE file
    satTle2elem = tleHandling.satTle2elem(tlePath)

    generatedTle = tleHandling.generateTle(satTle2elem[0])
    # Read the actual TLE file
    with open(tlePath, 'r') as file:
        tleToBeTested = file.read()

    # String compaire (line 0)
    eCountTle += int((generatedTle.splitlines()[0] != tleToBeTested.splitlines()[0]))
    # String compaire (line 1)
    eCountTle += int((generatedTle.splitlines()[1] != tleToBeTested.splitlines()[1]))
    # String compaire (line 2)
    for i, (field_expected, field_generated) in enumerate(zip(tleToBeTested.splitlines()[2].split(), generatedTle.splitlines()[2].split())):
        if i in [4]: # 4 = eccentricity (leading decimal point assumed), 5 = argument of perigee, 6 = mean anomaly
            # Eccentricity -> leading decimal point assume -> max tolerance is A_TOL_ROUNDTRIP * 1e7 => 1e-4 physical eccentricity tolerance
            eCountTle += abs(float(field_expected) - float(field_generated)) > A_TOL_ROUNDTRIP * 1e7
        elif i in [5,6]: # 5 = argument of perigee, 6 = mean anomaly (degrees)
            eCountTle += abs(float(field_expected) - float(field_generated)) > A_TOL_DEG_ROUNDTRIP
        else:
            eCountTle += abs(float(field_expected) - float(field_generated)) > A_TOL_ROUNDTRIP

    assert eCountTle < 1, f"{eCountTle} functions failed in tleHandling.py script, generateTleDataString() method"

if __name__ == "__main__":
    # Reading TLE
    test_read_tle(os.path.join(DATA_DIR, "hypso1.tle"), EXPECTED_OE_HYPSO)
    test_read_tle(DATA_DIR / "oneWeb25.tle", EXPECTED_OE_ONE_WEB)
    test_read_tle(DATA_DIR / "spacestations.2le", EXPECTED_OE_2LE)

    # Writing TLE
    test_write_tle('HYPSO 1', oeHypso1, hypso1noradId, hypso1launch, launch_Noyear=2, PoL='BX', tleEpoch=hypso1tleEpoch, nDot=hypso1nDot, bStar=hypso1bStar, expectedTlePath=DATA_DIR / "hypso1.tle")

    # Take TLE, generate orbital elements, then generate TLE again and compare to original
    test_read_write_tle(DATA_DIR / "hypso1.tle")
