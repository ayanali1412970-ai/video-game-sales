# Data quality report

- raw rows: **16598**
- clean rows (games): **16178**
- quarantined: **420**

## Quarantined by reason

- missing_year: 266
- partial_region_sales_annotation: 151
- release_year_before_platform_existed: 1
- release_year_after_platform_sunset: 1
- conflicting_duplicate_key: 1

## Flags kept in the table (rows NOT removed)

- in_coverage = False (year > 2015): 347
- is_sales_outlier (blockbusters): 461
- titles with name_note split out: 125
- year range: 1980-2017
- sources: {'vgsales': 16178}
