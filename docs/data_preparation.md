# Data Preparation

MemBrain-stats' workflow starts with the preparation of the data. We try to keep it as uniform as possible for all features implemented in the package.

The most important ingredients to MemBrain-stats are 
1. protein locations
2. membrane meshes

It is important to create one folder containing all data that should be analyzed together, e.g. all data from one tomogram. The files in this folder should contain above ingredients and can be structured in different ways.

Currently, we support the following formats for these:
## H5-containers
When using the .h5 containers that are output of MemBrain-pick, the data is already in the correct format. The files contain both the protein locations that are output of MemBrain-pick and the membrane meshes that were generated before the neural network prediction.


<div style="text-align: center;">
<img src="https://private-user-images.githubusercontent.com/34575029/365935802-ca6eea99-00d6-43f2-ad3c-c876148b697e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjU5NTMxNjEsIm5iZiI6MTcyNTk1Mjg2MSwicGF0aCI6Ii8zNDU3NTAyOS8zNjU5MzU4MDItY2E2ZWVhOTktMDBkNi00M2YyLWFkM2MtYzg3NjE0OGI2OTdlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTEwVDA3MjEwMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTQ5ZWQxM2M2MTQ0MGEzY2NhOWY4NDVkNDAxNWFlMzFmM2ViZmVlZDk0MDQ4NmViODdjYjRlZGQxNzI5NWMyY2EmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.aeWgGvGuZp9LOmRO35cuBTnLuM6TNd4JanQ9A9AqEgQ" alt="Folder Structure" width="50%">
</div>

## .star and .obj files
Alternatively, you can use .star files for the protein locations and .obj files for the membrane meshes. The .star files should contain the protein locations in the following format:
```
data_
loop_
_rlnCoordinateX #1
_rlnCoordinateY #2
_rlnCoordinateZ #3
_rlnClassNumber #4

1.0 1.0 1.0 1
2.0 2.0 2.0 1
3.0 3.0 3.0 1
...
```

The .obj files can be obtained from MemBrain-pick functionalities or generated using other software. 

<div style="text-align: center;">
<img src="https://private-user-images.githubusercontent.com/34575029/365935848-46e3b9e6-9067-4aea-a1a7-bac531fc81ab.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjU5NTMxNjEsIm5iZiI6MTcyNTk1Mjg2MSwicGF0aCI6Ii8zNDU3NTAyOS8zNjU5MzU4NDgtNDZlM2I5ZTYtOTA2Ny00YWVhLWExYTctYmFjNTMxZmM4MWFiLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTEwVDA3MjEwMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWY2MTU4NmMxNTkwMGU2NWUzZDIzNDg1NzZlMTI0MzZlNDBjYmRmZjEzZTVkYTIxOWM3NjE1OWI1YzRhMjUxZWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.eUeAPBWR8eDGRT5yASGalyTqZv4g7JH927FVOf32jkQ" alt="Folder Structure" width="50%">
</div>

