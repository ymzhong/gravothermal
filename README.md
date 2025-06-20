# Gravothermal Evolution of Self-Interacting Dark Matter Halos with Static Plummer Baryonic Potential

This repository contains two programs that together create and evolve spherically symmetric, non-rotating, gravothermally conducting fluid halos. If this code contributes to your research, please cite [Feng, Yu, and Zhong (2025)](https://arxiv.org/abs/2506.xxxxx) [![](https://img.shields.io/badge/arXiv-2506.12345-red)](https://arxiv.org/abs/2506.xxxxx). If you have any questions in running the code, please [contact me](mailto:yiming.zhong@cityu.edu.hk).

| File            | Language | Purpose                                                                                                   |
|-----------------|----------|-----------------------------------------------------------------------------------------------------------|
| `initialize.py` | Python   | Generates the initial NFW self-interacting dark matter (SIDM) under a Plummer baryonic potential and exports radius, density, mass, energy, and luminosity of SIDM as plain text. |
| `evolution.cpp` | C++   | Integrates the SIDM profiles forward in time, solving conduction and hydrostatic relaxation  under the static baryonic potential.          |

---

## 1. Quick Start

### 1.1 Compile requirements
* **Python ≥ 3.8** with `mpmath`, `numpy`, and `matplotlib`.
* **C++ 17** compiler (tested with **GCC 12** and **Intel ICC 2020**) plus **Eigen 3** headers.  

### 1.2  Generate initial conditions
```bash
python initialize.py     
# with setup
# base_path            where tables will be placed
# mass_norm            M_b / (4 π ρ_s r_s³)
# scale_norm           Plummer scale radius a / r_s
# sigma                (σ/m) ρ_s r_s
# tag                  label for the realisation
```
All output profiles appear in `output/initial/<tag>/`, and a PDF plot with all the profiles is saved in the same directory.

### 1.3  Build and run the C++ solver
```bash
# Example using GCC
g++ -O3 -std=c++17 -I/path/to/eigen evolution.cpp -o evolve

# or Intel
icc -Ofast -std=c++17 -I/path/to/eigen evolution.cpp -o evolve
```
The executable reads the profile files created in **Step 1** (edit the `inputDir`, `sigma`, `mass_norm`, `scale_norm`, and `tag` members of `SimulationParameters` if your paths or paramters differ) and writes `result_<tag>.txt` to `output/` by default.

Run it:
```bash
./evolve
```

---

## 2. Directory Layout
```
initial/
  2025xxxx/            # tag chosen in Step 1
    RList-2025xxxxx.txt
    MList-2025xxxxx.txt
    RhoList-2025xxxxx.txt
    uList-2025xxxxx.txt
    LList-2025xxxxx.txt
    Basic-2025xxxxx.txt
    profile.pdf
output/
  result_2025xxxx.txt  # full time series appended by evolve
```
The output txt file includes (1) the header part: records all the simulation setup information and (2)the body part: records `evolution time`, `evolution step` and `radii`, `densities`, `enclosed masses`, ` specific internal energies`, `luminosities` of the SIDM Lagrangian zons for all the conduction-relaxation steps.

---

## 3. Options
* **Verbose diagnostics** – instantiate `Logger logger(true)` in `main()`.  
* **More frequent snapshots** – decrease `DEFAULT_SAVE_STEPS`.  The output file becomes larger.
* **Alternative baryon models** – switch to `MbaryonH` (Hernquist) or `MbaryonSPL` (power-law) inside `updateProfiles()`.

---

## 4. Reference
Details of the impelementation can be found in Appendix B of [Zhong, Yang, and Yu (2023)](https://arxiv.org/abs/2306.08028).

