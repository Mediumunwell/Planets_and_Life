# HWC Orrery v4.3

This is a first cinematic orrery pass for the PHL Habitable Worlds Catalog.

Inputs:

- `../data/hwc/hwc_data_page.html` - cached copy of the PHL HWC data page.
- `../data/hwc/hwc_potentially_habitable.csv` - parsed first HWC table, 70 potentially habitable worlds.
- `../data/hwc/nasa_pscomppars_hwc.csv` - NASA Exoplanet Archive composite fields for the same worlds.

Outputs:

- `../assets/hwc_orrery_v4_3/hwc_orrery_v4_3.gif`
- `../assets/hwc_orrery_v4_3/hwc_orrery_v4_3_still.png`

Run:

```bash
python3 The_Solar_System/anthropic_principle_talk/orrery_hwc_v4_3/hwc_orrery.py
```

The layout starts from real RA/Dec positions when available and then gently spreads systems apart so the Kepler-field cluster remains readable.
