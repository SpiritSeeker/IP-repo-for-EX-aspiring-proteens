# Energy Curve Equalization

Usage: 
```
energy_curve.py [-h] [-o OUTPUT_FILE] [--display_graphs] I [I ...]
```

Command line arguments:
  - Input image: File name with complete or relative path
  
Optional arguments:
  - Output file: Complete or relative file path with an ```-o``` flag. Defaults to ```output.jpg```
  - Display Graphs: Flag to enable displaying graphs. ```--display_graphs``` to set. Defaults to false.

Example:
To run energy curve equalization on ```lena.jpg```, save to output to ```outputs/lena_equalized.jpg``` and to display grahps,
```shell
python energy_curve.py lena.jpg -o outputs/lena_equalized.jpg --display_graphs
```
