# Ontario Flood Pipeline - Status Report

**Date:** 2025-11-26
**Status:** Stage 3 Complete - Article-to-Event Matching Complete

---

## Executive Summary

This pipeline extracts flood event information from 50,000+ Ontario newspaper articles using a multi-stage NLP approach. Articles are progressively filtered and enriched to identify specific flood events with dates, locations, and geocoordinates.

### Current Results

| Metric | Value |
|--------|-------|
| **Input articles** | 50,000+ |
| **Stage 1 output** | 5,247 flood candidates |
| **Stage 2 output** | 2,290 verified Ontario floods |
| **Geocoded articles** | 1,940 (84.7%) |
| **Distinct events identified** | 494 |
| **Matched to known events** | 770 articles (50.9%) |
| **Potentially new events** | 742 articles (49.1%) |

---

## Pipeline Architecture

```
Raw Articles (50K+)
       ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 1: BERT Classification                                 │
│ - Semi-supervised learning with pseudo-labeling              │
│ - High-recall filter (>95%) to avoid missing floods          │
│ - Output: 5,247 flood candidate articles                     │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 2: LLM Flood Verification                              │
│ - Confirms actual flood events (not just mentions)           │
│ - Verifies Ontario location                                  │
│ - Output: 2,290 verified Ontario flood articles              │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 3: Information Extraction & Geocoding                  │
│ - NER extraction of locations/dates                          │
│ - LLM verification and correction                            │
│ - Mapbox geocoding to lat/lon                                │
│ - Event clustering by date/location                          │
│ - Output: Geocoded articles with event assignments           │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│ STAGE 4: Impact Extraction (Planned)                         │
│ - Extract casualties, evacuations, damages                   │
│ - Identify infrastructure impacts                            │
│ - Output: Full event records with impact data                │
└──────────────────────────────────────────────────────────────┘
```

---

## Stage 3 Detailed Results

### 3.1 NER Extraction

**Method:** spaCy NER for initial location/date extraction
**Results:**
- Location extraction: 92.7%
- Date extraction: 5.2% (very low - publication dates often missing)

### 3.2 LLM Verification

**Method:** DSPy + Claude Sonnet 4 (claude-sonnet-4-20250514)
**Processing:** 2,290 articles in ~25 minutes (8 parallel threads)

**Location Results:**
- Verified (NER correct): 813 (35.5%)
- Corrected by LLM: 1,477 (64.5%)

**Date Extraction Results:**
- High confidence: 1,142 (49.9%)
- Medium confidence: 378 (16.5%)
- Low confidence: 770 (33.6%)

**Date Format Distribution (after cleanup):**
- Month YYYY format: 1,450 (63.3%)
- Year only: 129 (5.6%)
- Decade: 7 (0.3%)
- Not found: 704 (30.7%)

### 3.3 Geocoding

**Method:** Mapbox Places API with Ontario/Canada context

**Results:**
| Metric | Value |
|--------|-------|
| Total articles | 2,290 |
| Unique locations | 532 |
| Successfully geocoded | 1,940 (84.7%) |
| In Ontario | 1,911 (98.5% of geocoded) |
| Not found | 12 |
| No location extracted | 338 |

### 3.4 Event Clustering

**Method:** Spatial-temporal clustering
- Distance threshold: 50km (Haversine)
- Temporal threshold: Same year, adjacent months

**Results:**
| Metric | Value |
|--------|-------|
| Total events | 494 |
| Single-article events | 299 (60.5%) |
| Multi-article events | 195 (39.5%) |

**Largest Event Clusters:**
1. July 2013 Toronto flood - 117 articles
2. July 2004 Peterborough flood - 49 articles
3. August 2018 Toronto flood - 51 articles

---

## Known Events Database

### Source Data

The known events database was merged from **4 authoritative sources**:

| Source | Records | Key Contribution |
|--------|---------|------------------|
| Canadian Disaster Database (CDD) | 73 | Impact data (deaths, evacuations, costs) |
| Conservation Ontario | 173 | Severity ratings, flood causes |
| Natural Resources Canada (NRCAN) | 267 | Geographic coordinates, historical depth |
| Ontario Emergency Management (OEM) | 210 | Detailed text descriptions |
| **Total Raw** | **723** | |
| **After Deduplication** | **412** | Unique events (1734-2024) |

### Merge Process

The merge pipeline (documented in `../flood_history_merge/`) performed:

1. **OEM Date Parsing** - Extracted 111 dates from text descriptions
2. **Cross-Source Deduplication**
   - Stage 1: Text similarity (85% threshold) - 139 duplicates
   - Stage 2: Spatial-temporal (<50km, same month) - 224 duplicates
3. **Mapbox Geocoding** - 280/318 locations (88% success)
4. **Data Validation** - Removed 1 Halifax, NS record incorrectly labeled

**Important Note:** The merge may be imperfect. Some known events may be:
- Duplicates that weren't detected (different location names)
- Missing events not in any of the 4 sources
- Events with incorrect dates or locations

### Known Events Coverage

- Spatial coverage: 91.2% have coordinates
- Temporal span: 1734-2024 (291 years)
- Date precision: 51.8% day-level, 13.9% month-level
- Impact data: 15.6% have deaths/evacuations/costs

---

## Article-to-Event Matching

### Method

Semi-supervised matching of newspaper articles to known flood events:
- **Spatial match:** Within 50km (Haversine distance)
- **Temporal match:** Same year, ±1 month tolerance
- **Priority:** Articles can match multiple events; closest match selected

### Results

| Category | Count | Percentage |
|----------|-------|------------|
| Articles matched to known events | 770 | 50.9% |
| Unmatched articles (potential new events) | 742 | 49.1% |
| Known events with article matches | 131 | 31.8% |
| Known events without article matches | 281 | 68.2% |

### Top Matched Events

| Event | Location | Date | Articles |
|-------|----------|------|----------|
| Toronto flood | Toronto | Aug 2012 | 110 |
| Toronto flood | Toronto | Jul 2018 | 51 |
| Peterborough flood | Peterborough | Jul 2004 | 49 |
| Ottawa region | Constance Bay | Apr 2019 | 35 |
| Hurricane Hazel | Toronto | Oct 1954 | 28 |

### Sample Unmatched Articles (Potential New Events)

These articles describe floods not in the known events database:

1. **1950s Whitedog hydro flooding** - Indigenous community impacts
2. **August 1986 Scarborough floods** - Urban flooding
3. **Various historical floods** - Pre-1900 events

---

## Output Files

### Stage 3 Outputs

| File | Description |
|------|-------------|
| `results/stage3_verified.json` | LLM-verified articles with dates/locations |
| `results/stage3_geocoded.json` | Geocoded articles with lat/lon |
| `results/stage3_event_clusters.json` | Event clusters from spatial-temporal grouping |
| `results/article_event_matches.json` | Article-to-known-event matches |
| `results/geocode_cache.json` | Mapbox geocoding cache |

### Summary Files

| File | Description |
|------|-------------|
| `results/stage3_verified_summary.json` | LLM verification statistics |
| `results/stage3_geocoded_summary.json` | Geocoding statistics |
| `results/stage3_ner_summary.json` | NER extraction statistics |

---

## Next Steps

### Immediate Analysis Opportunities

1. **Analyze Unmatched Articles (742)**
   - Cluster to identify distinct new events
   - Review for events that should be in known database
   - Identify historical events missing from records

2. **Investigate Unmatched Known Events (281)**
   - Why don't 68% of known events have article matches?
   - Possible reasons: events outside newspaper date range, regional newspapers not in corpus, minor events

3. **Validate Matching Quality**
   - Review borderline matches (high distance)
   - Check temporal mismatches (month off by 1)
   - Use article text to confirm event identity

### Pipeline Improvements

1. **Improve Date Extraction**
   - 30.7% of articles have "not found" dates
   - Could use publication date as fallback
   - Better prompt engineering for ambiguous cases

2. **Refine Geocoding**
   - 338 articles have no location to geocode
   - Some vague locations could be manually resolved
   - Consider river/watershed-level geocoding

3. **Cross-Validation**
   - Use known events to validate extraction quality
   - Measure precision/recall against ground truth

---

## Technical Notes

### Models Used

- **Stage 1:** Custom BERT (semi-supervised, high-recall)
- **Stage 2:** Claude Sonnet via DSPy
- **Stage 3 NER:** spaCy (en_core_web_sm)
- **Stage 3 LLM:** Claude Sonnet 4 (claude-sonnet-4-20250514)
- **Geocoding:** Mapbox Places API

### Processing Times

| Stage | Time | Rate |
|-------|------|------|
| LLM Verification | 25 min | 91.6 articles/min |
| Geocoding | ~2 min | 0.1s/request |
| Event Clustering | <1 min | - |
| Event Matching | <1 min | - |

### Dependencies

See `requirements.txt` for full list. Key packages:
- DSPy (LLM orchestration)
- spaCy (NER)
- Mapbox API (geocoding)
- pandas, numpy (data processing)

---

## Data Quality Notes

### Known Issues

1. **Date Format Cleanup Required**
   - Original LLM output had verbose "not applicable" phrases
   - Two-pass cleanup normalized to "Month YYYY" format
   - Some edge cases may remain

2. **Geocoding Limitations**
   - Vague locations ("Southern Ontario") cannot be geocoded
   - Some historical place names may not resolve
   - First Nations community names may have issues

3. **Event Clustering Assumptions**
   - 50km threshold may be too large for urban areas
   - Same-month assumption may miss multi-month events
   - Single-article events may be noise or real minor events

### Validation Performed

- Coordinates validated within Ontario bounds
- Date formats normalized
- Event matches verified within distance/time thresholds
- Sample manual review of extraction quality

---

**Last Updated:** 2025-11-26
**Pipeline Version:** Stage 3 Complete
