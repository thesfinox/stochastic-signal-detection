=========
Tutorials
=========

Though the easiest way to get started with the library is to use the script ``simulation_temp.py`` (to scan using a single *temperature* parameter) or ``simulation_traj.py`` (to scan using a different parametrization), we include two jupyter notebooks to show the basic usage of the library. In particular, we show the case of the analytical Marchenko-Pastur distribution and the case of the empirical distribution of the eigenvalues of a random matrix.

For more information on the scripts, you can use ``python <script> --help`` to get the definition of all command line parameters. Notice that numerical results will be saved in a SQLite database (path provided by the user from command line): you should foresee a utility to explore such database for further processing.
