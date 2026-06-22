# Static 3D Microstructure Visualisation

This repo starts with a simple Python/PyVista script for static 3D visualisation of validation microstructures.

Pore phase `3` is transparent because it is not rendered. The rendered phases are `1 = Ni` and `2 = YSZ`.

## Setup

Install Python 3.11 or newer, then from this folder run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

If `py` is not recognised after installing Python, use:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Use

1. Put the validation volume in `input/`.
2. Open `static_3d_visualisation.py`.
3. Set `RENDER_MODE = "full"` or `RENDER_MODE = "region"`.
4. For regional visualisation, edit `REGION_START_ZYX` and `REGION_END_ZYX`.
5. Run:

```powershell
.\.venv\Scripts\python .\static_3d_visualisation.py
```

To render a specific `.mat` file without editing the script, run:

```powershell
.\.venv\Scripts\python .\static_3d_visualisation.py --file t000000.mat
```

Set `USE_GRAYSCALE = True` in the script for grayscale output, or override it once:

```powershell
.\.venv\Scripts\python .\static_3d_visualisation.py --file t000000.mat --scheme grayscale
```

The script writes square, caption-free `.pdf` files to `output/`.
