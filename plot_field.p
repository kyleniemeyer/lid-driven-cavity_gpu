reset
set terminal postscript eps enhanced "Helvetica" 20 color
set output "velocity_field_cpu.eps"

unset logscale

set xrange [0 : 1]
set yrange [0 : 1]

set size square
set nokey

factor = 0.1
plot "velocity_cpu_128.dat" using 1:2:($3*factor):($4*factor) every 2 with vec lc 3

set output "velocity_field_gpu.eps"

unset logscale

set xrange [0 : 1]
set yrange [0 : 1]

set nokey

factor = 0.1
#plot "velocity_gpu.dat" using 1:2:($3*factor):($4*factor) every 2 with vec lc 3
plot "velocity_gpu_128_share.dat" using 1:2:($3*factor):($4*factor) every 2 with vec lc 3