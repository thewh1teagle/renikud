"""
Classify chars into their possible diacritics to instruct while vibe coding
"""
from constants import (
    SHVA, 
    SEGOL, 
    HIRIK, 
    PATAH, 
    HOLAM, 
    QUBUTS,
    DAGESH, 
    SIN_DOT,
    HATAMA
)

VOWEL_DIACRITICS = [SHVA, SEGOL, HIRIK, PATAH, HOLAM, QUBUTS]
CAN_HAVE_DAGESH = 'בכךפףו'
CAN_HAVE_SIN_DOT = 'ש'

CLASSIFICATION_MAP = {
    'א',
    'ב',
    'ג',
    'ד',
    'ה',
    'ו',
    'ז',
    'ח',
    'ט',
    'י',
    'כ',
    'ך',
    'ל',
    'מ',
    'ם',
    'נ',
    'ן',
    'ס',
    'ע',
    'פ',
    'ף',
    'צ',
    'ץ',
    'ק',
    'ר',
    'ש',
    'ת',
}