#We consider that we have a file, nammed Statistics.txt,
#in which, on each line we have three values
#frameNumber MSE PSNR
#
#This file, passed as parameter to gnuplot :
# gnuplot example3.gnuplot
# will save the curves of the MSE against time, and PSNR against time 
# to the file out.png

set terminal png
set output "out1.png"

set xlabel "Frames"
set ylabel "Time Consumption"
#set xrange [0:110]
#set yrange [0: 700]
#set xtics 10
#set ytics 50
set style line 1 lw 5
plot 'VideoProcessingStats.txt' using 1:2 with lines title 'Time Consumption'
