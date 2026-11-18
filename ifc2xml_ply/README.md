# IFC to XML and PLY converter

This standalone package converts the supplied frame and enclosure IFC models into Mitsuba scene XML and geometrical PLY meshes. It does not run ray tracing or propagation prediction.

From the repository root:

```powershell
python -m pip install -r MOO_Framework/requirements.txt
python -m ifc2xml_ply.main --help
python -m ifc2xml_ply.main
```

The default inputs are `data/frame.ifc` and `data/enclosure.ifc` within this package. Generated scenes are written to the ignored `output/` directory. The `functions/` directory contains the geometry, material, and wireless-property transformations.
