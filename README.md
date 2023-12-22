<div align="center">

# Scalable Automated School Mapping 

<p>
<b><a href="#-description">Description</a></b>
|
<b><a href="#-dataset">Dataset</a></b>
|
<b><a href="#-code-organization">Code Organization</a></b>
|
<b><a href="#-usage">Usage</a></b>
|
<b><a href="#-file-organization">File Organization</a></b>
|
<b><a href="#acknowledgement">Acknowledgment</a></b>
|
<b><a href="#citation">Citation</a></b>
</p>

</div>

## 📜 Description
This work leverages deep learning and high-resolution satellite images for automated school mapping across X countries. This work is developed under Giga, a global initiative by UNICEF-ITU to connect every school to the internet by 2030.

## 📂 Dataset
For each school and non-school location in our dataset, we downloaded 300 x 300 m, 500 x 500 px high-resolution satellite images from Maxar with a spatial resolution of 60 cm/px. After filtering, we obtained a total of X school images and X non-school images across 42 countries.

## 💻 Code Organization 
This repository is divided into the following files and folders:
- **notebooks/**: contains all Jupyter notebooks for exploratory data analysis and model prediction.
- **utils/**: contains utility methods for data cleaning, data visualization, model development, and model training routines.
- **src/**: contains scripts runnable scripts for automated data cleaning and model training/evaluation.

## 💻 Usage

### Setup
1. Download anaconda or miniconda.
2. Create a virtual environment.
```s
conda create -n envname python=x.x anaconda
```
3. Activate virtual environment.
```s
conda activate envname
```
4. Install requirements
```s
pip install -r requirements.txt
```

## 📂 File Organization 
The datasets are organized as follows:
```
data
├── rasters
│   ├── maxar
│   │   ├── AIA
│   │   │   ├── school
│   │   │   │    ├── UNICEF-AIA-SCHOOL-00000001.tiff
│   │   │   │    └── ...
│   │   │   ├── non_school
│   │   │   │    ├── UNICEF-AIA-NON_SCHOOL-00000001.tiff
│   │   │   │    └── ...
│   │   │   └── ...
│   │   └── ...
└── vectors
    ├── school
    │   ├── unicef
    │   │   ├──AIA_school_geolocation_coverage_master.csv
    │   │   └── ...
    │   ├── osm
    │   │   ├──AIA_osm.geojson
    │   │   └── ...
    │   ├── overture
    │   │   ├──AIA_overture.geojson
    │   │   └── ...
    └── non_school
        ├── osm
        │   ├──AIA_osm.geojson
        │   └── ...
        └── overture
            ├──AIA_overture.geojson
            └── ...
    
```
