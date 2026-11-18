# WAP placement MOO: code and supplementary tables

Research materials for stage-aware wireless access point (WAP) placement during the structural frame and enclosure stages of a building project.

| Directory | Contents |
| --- | --- |
| [`MOO_Framework/`](MOO_Framework/) | Modelling–Optimization–Operation (MOO) framework, six-LLM task runner and records, expert (EbM) and geometric (GcM) baselines, IFC inputs, and supporting data. |
| [`ifc2xml_ply/`](ifc2xml_ply/) | Standalone IFC-to-Mitsuba XML/PLY conversion code and its input data. |
| [`Supplementary_Tables/`](Supplementary_Tables/) | Tables S1–S5, each in its own directory with the original Excel workbook and a CSV for every worksheet. |

The six-LLM benchmark comprises 480 assigned tasks across six models, two stages, five sensor profiles, and eight scenarios. Task-level JSON records and summary files are in [`MOO_Framework/result/six_llm/`](MOO_Framework/result/six_llm/). Large generated radio maps, logs, local credentials, manuscript drafts, and working figures are not part of this release.

See the directory READMEs for scope and usage. This repository is currently **private**.
