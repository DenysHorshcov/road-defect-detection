"""Single source of truth for the detection classes and their RDD2022 origins."""

CLASS_NAMES = ["crack", "pothole", "faded_lane_marking"]

CLASS_IDS = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# RDD2022 damage codes -> our classes. Codes come from the Japanese Road
# Maintenance Guidebook 2013 and are shared across the RDD2022 country subsets.
# D43 (crosswalk blur) is deliberately excluded: faded crosswalks are a
# different phenomenon from lane markings. Revisit only if faded_lane_marking
# turns out to be too data-starved to learn.
RDD_CODE_TO_CLASS = {
    "D00": "crack",              # longitudinal crack
    "D10": "crack",              # transverse crack
    "D20": "crack",              # alligator / fatigue crack
    "D40": "pothole",            # pothole
    "D44": "faded_lane_marking", # white line blur
}
