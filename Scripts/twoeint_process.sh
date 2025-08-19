#!/bin/bash

awk '{print $1,$2,$3,$4,$5,$6}' twoelecint_test.txt > out.1
column -t -s '  ' < out.1 > out.2
awk '{print $7,$8,$9,$10,$11,$12}' twoelecint_test.txt > out.3
column -t -s '  ' < out.3 > out.4
rm out.1 out.3
awk 'NF' out.2 out.4 > twoelecintegral.txt
rm out.2 out.4