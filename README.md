# hww

$python vision.py --help
t : histogram type
grid : Not in pdf, average over pixels
divide: in pdf, divide picture in to regions and concat their histograms
bins: per histogram bin count
example:

python vision.py -t 3 --divide 4 --bins 16
