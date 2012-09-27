reset
set terminal postscript eps enhanced "Helvetica" 20 color
set output "velocity_field_cpu.eps"

unset logscale

set xrange [0 : 1]
set yrange [0 : 1]

set nokey

factor = 0.15
plot "velocity_cpu.dat" using 1:2:($3*factor):($4*factor) with vec lc 3