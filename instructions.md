# Steps to run

1.- Update mdStepsrkf directory, there's a script to do that in the
directory.

2.- run `$AMSBIN/plams rkf_to_xyz.py`

3.- run `$AMSBIN/plams region_selector.py`

4.- run `$AMSBIN/plams run_generator.py`, it doesn't just create the run
files but also update the database 

5.- run `to_rameau.sh` and in rameau there's another script to create
the launcher for every file.

6- run `to_criann.sh` in rameau

7.- in rameau ther's a wathcer.sh to know if i can send calculations
to the queue in criann

8.- criann has a master.sh to launch the calculaitons and also a
cleaner to have space.

