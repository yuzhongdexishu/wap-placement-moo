# SP3 / SC1 experiment report (2026-11-13)

MOO used DeepSeek, SP3 at 23 dBm, SC1 priority weight 2, 600 fixed receivers per stage, 150 legal candidate WAP sites, a 4 m minimum WAP spacing, Sionna RadioMapSolver at 0.75 m cell size and 1,000,000 samples per transmitter. All 16 MCP tool calls succeeded.

The candidate dictionary now mixes legal receiver-derived sites with scene-centre and regular-grid anchors. The earlier receiver-only dictionary omitted the legal three-floor centre layout by 3.7–5.2 m. The GA starts with multiple independently found RF-feasible layouts; the prior 80-member populations had only one feasible member per stage. The objective F remains the weighted mean of normalized RSRP, RSRQ, and SINR; every one of the 600 points must separately satisfy the basic RF thresholds.

| Stage | PM WAPs | PM F on optimization map | PM F on common comparison map | Best baseline F on common map | PM failures / 600 | Minimum count proven on candidate set |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| frame | 3 | 0.997597 | 0.997597 | 0.995065 (EbM1/EbM2) | 0 | yes |
| enclosure | 12 | 0.987974 | 0.986763 | 0.784174 (EbM3) | 0 | yes |

The common comparison map contains the original PM and five baseline site sets, uses the same 600 receivers and 1,000,000 samples per transmitter, and scores every method with the same formulas. All methods have equal WAP count within each stage. `result/comparison_table.csv` contains all 12 method-stage rows and the 600-point failure counts.

Independent dense probes still fail at 6/13,818 frame positions and 185/12,071 enclosure positions for PM. Therefore the minimum-count and all-receiver claims apply to the chosen 600 points and 150 candidate sites, not the entire continuous work domain. The PM result is the highest observed F in the reported six-method comparison; a global optimum over all physical installation sites has not been proven.

Figures: `plots/uplift/`, `plots/violin/`, coordinate-registered per-floor overlays in `plots/registered_heatmaps/`, and their combined 3D views in `plots/stacked_registered_heatmaps/`. The older Matplotlib mesh views remain in `plots/3d_heatmaps/`; `plots/coordinate_alignment.md` explains the exact camera and floor registration. The previous runs are retained in `result/legacy_before_sp3_rerun_20261112.zip`, `result/sp3_50k_diagnostic_20261112.zip`, and `result/sp3_1m_receiver_only_20261113.zip`.
