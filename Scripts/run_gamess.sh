#!/bin/bash

rm ~/scr/*.dat
echo "Gotcha!!!"

sed -e "/NPRINT/ s/1/3/;" water-4_RHF_ccpvtz.inp > water-4_RHF_ccpvtz_temp.inp
~/gamess/rungms water-4_RHF_ccpvtz_temp.inp 01 > water-4_RHF_ccpvtz.dat
rm water-4_RHF_ccpvtz_temp.inp

rm ~/scr/*.dat
echo "You Fool!!!"

sed -e "/NPRINT/ s/1/4/;" water-4_RHF_ccpvtz.inp > water-4_RHF_ccpvtz_temp.inp
~/gamess/rungms water-4_RHF_ccpvtz_temp.inp 01 > water-4_RHF_ccpvtz_2eint.dat
rm water-4_RHF_ccpvtz_temp.inp