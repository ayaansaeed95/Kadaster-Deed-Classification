"""
models/regex/pattern_sources.py — curated pattern data for the regex classifier.

All hand-curated and hand-extracted pattern data lives here so the logic
modules (patterns.py, scoring.py, pipeline.py) stay focused on behaviour.
Five sources feed the pattern table:

  - EXCEL_CODES              : Kadaster Excel curated phrases          (Source A)
  - POTHOVEN_CODES           : Pothoven docx secondary triggers        (Source B)
  - NOTEBOOK_MANUAL_PATTERNS : original hand-curated title/citation set (Source E)
  - MINED_TITLE_PATTERNS     : 14 title-zone phrases mined for tail codes and
                               vetted (precision >= 0.91 on the train overige
                               bucket; the 15th candidate, 518 "bij proces
                               verbaal", was dropped after the coverage check
                               showed only 1/162 deeds covered)
  - VALUELIST_CANDIDATES     : phrases derived from the official Kadaster
                               AardStukdeel valuelist; the pipeline auto-
                               validates each at runtime and keeps only those
                               whose title-zone precision is >= 0.85 with >= 3
                               absolute hits in the overige bucket

Provenance for the Kadaster-provided sources: Thijs Pothoven (Senior
Information Engineer Akte-AI, Kadaster, 2026-04-15) — the "Indicatieve teksten
...xlsx" and "2de rechtsfeiten.docx" attachments.
"""

from __future__ import annotations


# --------------------------------------------------------------------------
# Dutch stopwords used by the TF-IDF / chi2 mining (Sources C & D)
# --------------------------------------------------------------------------
DUTCH_STOPWORDS: list[str] = [
    "de", "het", "een", "en", "van", "in", "is", "dat", "op", "te",
    "zijn", "met", "voor", "niet", "aan", "er", "ook", "om", "naar",
    "of", "als", "maar", "die", "deze", "dit", "door", "uit", "bij",
    "dan", "wel", "nog", "zo", "ze", "hij", "haar", "zij", "hun", "u",
    "je", "jij", "we", "wij", "ons", "onze", "mijn", "jouw", "uw",
    "wat", "welke", "wie", "waarom", "hoe", "waar", "wanneer", "want",
    "tot", "tegen", "over", "onder", "boven", "achter", "binnen",
    "buiten", "tussen", "zonder", "behalve", "hierbij", "daarbij",
    "echter", "toch", "reeds", "al", "alle", "alles", "andere", "ander",
    "elk", "elke", "iedere", "veel", "weinig", "geen", "ja", "nee",
    "heeft", "heb", "hebben", "had", "hadden", "was", "waren",
    "wordt", "worden", "werd", "werden", "zal", "zullen", "zou", "zouden",
    "kan", "kunnen", "kon", "konden", "moet", "moeten", "moest", "moesten",
    "mag", "mogen", "mocht", "mochten", "wil", "willen", "wilde", "wilden",
]


# --------------------------------------------------------------------------
# Source A — Kadaster Excel: curated phrases for the top-19 codes
# (Pothoven, 2026-04-15; extracted from "Indicatieve teksten ...xlsx")
# --------------------------------------------------------------------------
EXCEL_CODES: dict[int, dict] = {
    606: {"name": "Overdracht", "synoniemen": ["Vervreemding", "Levering", "Verkoop"],
          "declaratief": ["levert...aanvaardt", "wordt geleverd… wordt aanvaard",
                          "leveren...aanvaarden"]},
    537: {"name": "Hypotheek", "synoniemen": ["Zekerheidstelling"],
          "declaratief": ["verleent", "wordt verleend", "te verlenen"]},
    545: {"name": "Kwalitatieve verplichting",
          "synoniemen": ["Afstand kwalitative verplichting", "kwalitatieve verbintenis",
                         "Afstand kwalitative verplichting", "Vervallen kwalitatieve verplichting",
                         "Vestiging kwalitatieve verplichting"],
          "declaratief": ["vastgelegd", "vestigt", "gevestigd", "verleent... aanvaardt",
                          "afstand... is vervallen", "afstand…aanvaarden", "komen overeen",
                          "zal overgaan"]},
    585: {"name": "Verklaring van erfrecht", "synoniemen": ["VVE"],
          "declaratief": ["aanvaardt", "aanvaarden", "beneficiair is aanvaard",
                          "beneficiair aanvaard"]},
    572: {"name": "Stuk betreffende erfdienstbaarheden",
          "synoniemen": ["Afstanddoening erfdienstbaarheid", "Vestiging erfdienstbaarheid"],
          "declaratief": ["wordt…gevestigd", "vestigt…aanvaardt", "vestigen…aan te nemen",
                          "verklaart afstand te doen... welke wordt aanvaardt"]},
    564: {"name": "Verbetering", "synoniemen": [], "declaratief": []},
    527: {"name": "Erfpachtcanon (wijziging)",
          "synoniemen": ["Afkoop erfpacht", "Verlenging tijdvak", "Canonherziening",
                         "Erfpachtconversie"],
          "declaratief": ["afgekocht door betaling", "wordt verlengd", "zijn overeen gekomen",
                          "overeenstemming hebben bereikt", "komen hierbij overeen",
                          "Bij deze te wijzigen"]},
    532: {"name": "Aanvullende akte", "synoniemen": [], "declaratief": []},
    538: {"name": "Hyoptheek (Doorhaling)",
          "synoniemen": ["Royement", "Ambtshalve doorhaling", "Vervallenverklaring",
                         "Afstand hypotheekrecht"],
          "declaratief": ["verklaarde dat… zijn vervallen", "gedeeltelijk opgezegd",
                          "gedeeltelijk afstand hebben gedaan", "verklaart afstand te doen",
                          "verklaart op te zeggen", "zijn opgezegd",
                          "gedeeltelijk zijn vervallen", "afstand wordt gedaan"]},
    580: {"name": "Verdeling van gemeenschap (gezamenlijk rechthebbenden)",
          "synoniemen": ["toebedeling", "toedeling"],
          "declaratief": ["geleverd…aanvaardt", "leveren… aan te nemen",
                          "geven..uitvoering aan hun overeenkomst van verdeling en leveren…aanvaardt"]},
    581: {"name": "Verdeling van gemeenschap (huwelijk - geregistreerd partnerschap)",
          "synoniemen": ["uitreksel huwelijksvoorwaarden", "toedeling", "toebedeling"],
          "declaratief": ["geleverd…aanvaardt", "leveren… aan te nemen",
                          "geven..uitvoering aan hun overeenkomst van verdeling en leveren…aanvaardt"]},
    644: {"name": "Vestiging zakelijk recht van opstalrecht nutsvoorzieningen",
          "synoniemen": ["vestiging recht van opstal"],
          "declaratief": ["verleent… recht van opstalrecht… aanvaardt",
                          "bij deze wordt gevestigd… verklaren aan te nemen",
                          "wordt gevestigd en aanvaard"]},
    543: {"name": "Koopovereenkomst beëindigen", "synoniemen": [],
          "declaratief": ["welke inschrijving door de inschrijving van een afschrift van deze "
                          "akte waardeloos zal worden."]},
    516: {"name": "Beperkt recht (wijzigen voorwaarden)",
          "synoniemen": ["verlenging recht", "wijziging bestemming", "wijziging bepalingen",
                         "wijziging recht van…", "akte inhoudende overeenkomst tot overstap van… naar",
                         "wijziging …recht"],
          "declaratief": ["verklaren te zijn overeengekomen", "wijzigen hierbij",
                          "verklaarden… te zijn aangegaan… voorts worden gewijzigd respectievelijk aangevuld",
                          "zijn overeengekomen… te wijzigen", "komen overeen",
                          "zijn… een (…)overeenkomst aangegaan"]},
    517: {"name": "Beslag", "synoniemen": ["conservatoir beslag", "excecutoriaal beslag"],
          "declaratief": ["heb ik… beslag gelegd", "heb ik… in beslag genomen"]},
    696: {"name": "koopovereenkomst, art. 7:3 BW en 9.9 Omgevingswet",
          "synoniemen": ["koopakte overeenkomst", "koopoptie", "optie"],
          "declaratief": ["verklaar overeenkomstig <artikelen> niet aan inschrijving \nvan ... "
                          "koopovereenkomst in de weg staat.",
                          "verzoek ik, notaris, namens ... een afschrift van deze verklaring in de "
                          "openbare registers \nop grond van <artikelen>in te schrijven"]},
    651: {"name": "koopovereenkomst, art. 7:3 BW en 10 WVG",
          "synoniemen": ["koopakte overeenkomst", "koopoptie", "optie"],
          "declaratief": []},
    518: {"name": "Beslag (doorhaling)", "synoniemen": [],
          "declaratief": ["heb ik... aangezegd dat gedeeltelijk is opgezegd",
                          "heb ik... aangezegd dat is opgezegd"]},
    652: {"name": "Koopovereenkomst, art. 73 BW",
          "synoniemen": ["koopakte overeenkomst", "koopoptie", "optie"],
          "declaratief": []},
}


# --------------------------------------------------------------------------
# Source B — Pothoven docx: secondary triggers for 8 less-frequent codes
# (Pothoven, 2026-04-15; extracted from "2de rechtsfeiten.docx")
# --------------------------------------------------------------------------
POTHOVEN_CODES: dict[int, dict] = {
    560: {"name": "Rangwisseling", "tekst": ["Rangwisseling", "Rang", "Wisselen"]},
    572: {"name": "Erfdienstbaarheid",
          "tekst": ["Erfdienstbaarheid", "Vestigen ten behoeve … ten laste", "Over en weer",
                    "Dienende … heersende"]},
    545: {"name": "Kwalitatieve Verplichting",
          "tekst": ["Art. 6:252 BW", "Kwalitatieve verplichting", "Kwalitatieve verbintenis",
                    "Dulden", "Niet doen"]},
    543: {"name": "Doorhaling Koopovereenkomst", "tekst": ["Waardeloos", "Koopovereenkomst"]},
    671: {"name": "Vermenging", "tekst": ["Vermenging", "door vermenging tenietgaan"]},
    613: {"name": "Vervallen recht van gebruik en bewoning",
          "tekst": ["Vervallen gebruik en bewoning",
                    "geconstateerd, komt te vervallen/is te komen vervallen"]},
    611: {"name": "Vervallen recht van vruchtgebruik",
          "tekst": ["Vervallen vruchtgebruik",
                    "geconstateerd, komt te vervallen/is te komen vervallen"]},
    644: {"name": "Vestiging opstalrecht nutsvoorzieningen", "tekst": ["opstalrecht", "gevestigd"]},
}


# --------------------------------------------------------------------------
# Source E (notebook) — original hand-curated patterns, mostly targeting the
# big codes (606/537/585/...). At cutoff 500 almost all are filtered out
# because their target codes are now in the base-model layer; only 696 and
# 518 survive into the tail set.
# --------------------------------------------------------------------------
NOTEBOOK_MANUAL_PATTERNS: list[dict] = [
    {"code": 696, "name": "Koop art. 7:3 BW + art. 9.9 Omgevingswet",
     "regex": r"(?i)\bartikel\s+9\.9\s+(?:van\s+(?:de\s+)?)?Omgevingswet\b",
     "source_text": "artikel 9.9 Omgevingswet", "match_zone": "full"},
    {"code": 606, "name": "Levering", "phrase": "Akte van levering", "match_zone": "title"},
    {"code": 585, "name": "Verklaring van erfrecht", "phrase": "Verklaring van erfrecht",
     "match_zone": "title"},
    {"code": 537, "name": "Hypotheek", "phrase": "Hypotheek", "match_zone": "title"},
    {"code": 564, "name": "Rectificatie", "phrase": "Rectificatie", "match_zone": "title"},
    {"code": 564, "name": "Akte van verbetering", "phrase": "Akte van verbetering",
     "match_zone": "title"},
    {"code": 572, "name": "Vestiging erfdienstbaarheid", "phrase": "Vestiging erfdienstbaarheid",
     "match_zone": "title"},
    {"code": 545, "name": "Kwalitatieve verplichting", "phrase": "Kwalitatieve verplichting",
     "match_zone": "title"},
    {"code": 518, "name": "Opheffing beslag", "phrase": "Opheffing beslag", "match_zone": "title"},
    {"code": 527, "name": "Omzetting erfpacht", "phrase": "Overeenkomst tot omzetting…erfpacht",
     "match_zone": "title"},
    {"code": 527, "name": "Wijziging erfpacht", "phrase": "Overeenkomst tot wijziging…erfpacht",
     "match_zone": "title"},
]


# --------------------------------------------------------------------------
# Source E (mined) — title-zone phrases mined from the overige bucket at
# cutoff 500. Each measured precision >= 0.91 (most
# 1.00) on the train overige bucket. The original mining surfaced 15
# candidates; "bij proces verbaal" (code 518) was dropped after the absolute-
# coverage check showed it matched only 1 of 162 deeds (too narrow).
# --------------------------------------------------------------------------
MINED_TITLE_PATTERNS: list[dict] = [
    {"code": 517, "phrase": "executoriaal beslag op onroerende zaken",
     "name": "Beslag (executoriaal, specific)", "match_zone": "title"},
    {"code": 517, "phrase": "beslag gelegd",
     "name": "Beslag (broader trigger)",        "match_zone": "title"},
    {"code": 518, "phrase": "opgeheven",
     "name": "Beslag doorhaling",               "match_zone": "title"},
    {"code": 607, "phrase": "akte van ruiling",
     "name": "Ruiling",                         "match_zone": "title"},
    {"code": 608, "phrase": "akte van schenking",
     "name": "Schenking",                       "match_zone": "title"},
    {"code": 617, "phrase": "ondersplitsing in appartementsrechten",
     "name": "Ondersplitsing in appartementsrechten", "match_zone": "title"},
    {"code": 618, "phrase": "onderhandse akten van volmacht",
     "name": "Onderhandse volmacht",            "match_zone": "title"},
    {"code": 628, "phrase": "ontbindende voorwaarden",
     "name": "Vervallen ontbindende voorwaarden", "match_zone": "title"},
    {"code": 628, "phrase": "akte van kwijting",
     "name": "Kwijting",                        "match_zone": "title"},
    {"code": 631, "phrase": "verklaring van verjaring",
     "name": "Verjaring (verklaring)",          "match_zone": "title"},
    {"code": 631, "phrase": "verjaring",
     "name": "Verjaring (broader)",             "match_zone": "title"},
    {"code": 634, "phrase": "afstand",
     "name": "Afstand beperkt zakelijk recht",  "match_zone": "title"},
    {"code": 635, "phrase": "akte van statutenwijziging",
     "name": "Statutenwijziging",               "match_zone": "title"},
    {"code": 641, "phrase": "afgifte legaat",
     "name": "Afgifte legaat",                  "match_zone": "title"},
]


# --------------------------------------------------------------------------
# Source E (valuelist) — title-zone phrase candidates derived from the
# official Kadaster AardStukdeel valuelist
# (https://developer.kadaster.nl/schemas/waardelijsten/AardStukdeel/).
# The pipeline validates each candidate at runtime against the overige bucket
# and keeps only those whose title-zone precision >= 0.85 AND >= 3 hits.
# --------------------------------------------------------------------------
VALUELIST_CANDIDATES: list[tuple[int, str, str]] = [
    (504, "algemene akte",                                "Algemene akte"),
    (508, "beheeroverdracht",                             "Beheeroverdracht"),
    (509, "bekrachtiging",                                "Bekrachtiging"),
    (516, "wijziging beperkt recht",                      "Beperkt recht (wijziging)"),
    (524, "cessie",                                       "Cessie"),
    (529, "fusie",                                        "Fusie"),
    (536, "huwelijkse voorwaarden",                       "Huwelijkse voorwaarden"),
    (536, "partnerschapsvoorwaarden",                     "Partnerschapsvoorwaarden"),
    (540, "kavelruil",                                    "Kavelruil"),
    (541, "kavelruilovereenkomst",                        "Kavelruil overeenkomst"),
    (546, "mandeligheid",                                 "Mandeligheid (beëindiging)"),
    (547, "mandeligheid",                                 "Mandeligheid (ontstaan)"),
    (554, "onteigening",                                  "Onteigening"),
    (560, "rangwisseling",                                "Rangwisseling"),
    (566, "registratie eigendom netwerk",                 "Reg. eigendom netwerk"),
    (569, "reglement",                                    "Reglement mede-eigenaren"),
    (579, "verdeling van gemeenschap erfgenamen",         "Verdeling (erfgenamen)"),
    (579, "erfgenamen",                                   "Verdeling (erfgenamen broad)"),
    (580, "gezamenlijk",                                  "Verdeling (gezamenlijk)"),
    (581, "geregistreerd partnerschap",                   "Verdeling (huwelijk/partnerschap)"),
    (588, "vaststellingsovereenkomst",                    "Vaststellingsovereenkomst"),
    (591, "verklaring van waardeloosheid",                "Verkl. waardeloosheid"),
    (597, "omzetting",                                    "Omzetting rechtspersoon"),
    (598, "splitsing rechtspersoon",                      "Splitsing rechtspersoon"),
    (609, "vestiging recht van erfpacht",                 "Vest. erfpacht"),
    (610, "vestiging recht van opstal",                   "Vest. opstal"),
    (611, "vestiging recht van vruchtgebruik",            "Vest. vruchtgebruik"),
    (613, "vestiging recht van gebruik en bewoning",      "Vest. gebruik/bewoning"),
    (614, "vestiging zakelijk recht",                     "Vest. zakelijk recht (overig)"),
    (616, "splitsing in appartementsrechten",             "Splitsing in app.rechten"),
    (620, "opheffing splitsing in appartementsrechten",   "Opheffing splitsing"),
    (621, "opheffing ondersplitsing",                     "Opheffing ondersplitsing"),
    (625, "vervulling opschortende voorwaarde",           "Vervulling opschortend"),
    (627, "vervallen opschortende voorwaarde",            "Vervallen opschortend"),
    (629, "afkoop grondrente",                            "Afkoop grondrente"),
    (630, "verjaring betwist",                            "Verjaring (betwist)"),
    (633, "vervallen verklaring",                         "Vervallenverklaring beperkt zr"),
    (636, "doorhaling overig",                            "Doorhaling overig"),
    (642, "overdracht om niet",                           "Overdracht om niet"),
    (644, "opstalrecht nutsvoorzieningen",                "Opstalrecht nutsvoorz."),
    (671, "vermenging",                                   "Vermenging"),
    (674, "vereniging erfpacht",                          "Vereniging erfpacht"),
    (703, "inbreng",                                      "Inbreng rechtspersoon"),
]
