# Lab 3: Chromaticity and the CIE diagram from three Gaussian cones

Build a CIE-like diagram from three Gaussian cone sensitivities: the spectral locus in cone space, its convex hull, a 2D slice, and display primaries with their gamut. Tasks are in `Lab3_Chromaticity_and_the_CIE_Diagram.pdf`.

```sh
mkdir lab3 && cd lab3
curl -LO https://utkarshmall13.github.io/cv8508-labs/lab3/setup.sh
sh setup.sh
. .venv/bin/activate
```

`setup.sh` downloads `lab3_tools.py` and creates `.venv` (numpy, scipy, matplotlib).

- `lab3_tools.py`: wavelength grid `WL`, `cone_sensitivities()` (the three clipped Gaussians) and `lms_to_srgb()` (colours for plots).
