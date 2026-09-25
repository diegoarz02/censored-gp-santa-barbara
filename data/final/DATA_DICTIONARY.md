# Data dictionary - `santa_barbara_soil`

Soil geochemistry dataset for the Santa Bárbara mercury mine, Huancavelica, Peru.

- **Layout**: long format, one row per location x analyte.
- **Rows**: 1140 · **Unique locations**: 114 · **Analytes**: 10.
- **Sampling**: 2018-07-18 to 2018-09-09.
- **Source**: OEFA report INFORME N° 00341-2018-OEFA/DEAM-STEC, environmental assessment of the Santa Bárbara mining unit.
- **CRS**: UTM zone 18S (EPSG:32718) and geographic WGS84 (EPSG:4326).
- **Produced by**: `notebooks/01_dataset_construction.ipynb` (seed 20260906).

## Columns

| Column | Type | Unit | Description | Source |
|---|---|---|---|---|
| `location_id` | text | - | Stable location identifier derived from the UTM coordinates rounded to the metre. | derived |
| `point_name` | text | - | Original OEFA point name. Two names separated by ' | ' mark the single location where two field points share one recorded coordinate. | OEFA |
| `easting` | float | m | Easting, UTM zone 18S (EPSG:32718). | OEFA |
| `northing` | float | m | Northing, UTM zone 18S (EPSG:32718). | OEFA |
| `lon` | float | degrees | Longitude, WGS84 (EPSG:4326). | derived via pyproj |
| `lat` | float | degrees | Latitude, WGS84 (EPSG:4326). | derived via pyproj |
| `utm_zone_epsg` | int | - | EPSG code of the projected CRS; constant 32718. | derived |
| `elev_oefa` | float | m a.s.l. | Elevation recorded in the field by OEFA. | OEFA |
| `elev_dem` | float | m a.s.l. | Elevation sampled from the SRTM 30 m DEM at the point. | SRTM 30 m |
| `slope_deg` | float | degrees | Terrain slope, Horn (1981) 3x3 gradient on the SRTM DEM. | SRTM 30 m |
| `aspect_deg` | float | degrees | Terrain aspect, degrees clockwise from north. | SRTM 30 m |
| `stratum` | text | - | Sampling stratum: 'background', 'potential_interest' or 'unlabelled'. | OEFA |
| `sample_type` | text | - | Sample support: 'simple' or 'composite'. All background points are composite. | OEFA |
| `sector` | text | - | Sector name parsed from the OEFA field description. | derived from OEFA |
| `date` | date | - | Sampling date. | OEFA |
| `analyte` | text | - | Element symbol: Hg, Pb, As, Ba, Cd, Co, Sb, Ag, Bi, Ni. | OEFA |
| `role` | text | - | Role in the study: 'truth field', 'decision metal' or 'censoring gradient'. | study design |
| `value` | float | mg/kg | Quantified concentration. Empty when the result is censored: nothing is imputed. | OEFA |
| `censored` | bool | - | True when the result was reported as '<L' (left censored). | derived from OEFA |
| `reported_limit` | float | mg/kg | Detection limit as published by OEFA. Empty for analytes with no censored result. | OEFA |
| `resolution` | float | mg/kg | Laboratory reporting resolution inferred from the decimals of the quantified values. A value reported as 383 means [382.5, 383.5). | derived |
| `unit` | text | - | Concentration unit, homogenised to mg/kg dry weight. | OEFA |
| `eca_agricultural` | float | mg/kg | Peruvian soil standard, agricultural land use (D.S. 011-2017-MINAM). | MINAM |
| `eca_residential` | float | mg/kg | Peruvian soil standard, residential and parks. | MINAM |
| `eca_industrial` | float | mg/kg | Peruvian soil standard, commercial, industrial and extractive. | MINAM |
| `dist_waste_dump_m` | float | m | Euclidean distance in UTM 18S to the nearest of the 16 waste-rock dumps sampled by OEFA. | derived from OEFA |
| `dist_adit_m` | float | m | Distance to the nearest of the 5 outcrop/adit points sampled by OEFA. | derived from OEFA |
| `dist_any_working_m` | float | m | Distance to the nearest mine working of any type. | derived from OEFA |
| `dist_city_m` | float | m | Distance to the Plaza de Armas of Huancavelica (-74.9758, -12.7867). | derived |
| `coord_flag` | bool | - | True for the location where two field points share one recorded coordinate. | derived, QC |
| `description` | text | - | Original OEFA field description of the location. | OEFA |

## Notes on use

1. **Censoring.** `value` is empty on censored rows by design. Do not substitute `L/2`: that
   practice is biased and is precisely the comparison this study makes.
2. **Two supports.** The 30 background points are composite samples and every potential-interest
   point is simple, so stratum and support are almost perfectly aliased. Notebook 2 fits both
   effects and reports their posterior correlation before deciding what can be identified.
3. **Coordinate QC.** One location carries two field point names that share a single recorded
   coordinate (`coord_flag = True`). The coordinate could not be verified, so it was not
   corrected; the two records are averaged in log space and flagged.
4. **Reporting resolution.** Quantified values are recorded to a finite number of decimals, so
   each detection is itself interval-censored. `resolution` gives the interval width.
5. **Regulatory thresholds.** D.S. N.° 011-2017-MINAM, soil environmental quality standards
   (mg/kg dry weight). Cadmium: 1.4 agricultural, 10 residential, 22 industrial.
6. **Not available in the open data.** Sampling depth, laboratory identity and OEFA's own
   conclusions are in the report PDF, which is not published (it is a causality assessment
   feeding a sanctioning procedure). They were not requested for this work.
