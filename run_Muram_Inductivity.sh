tavg=(0 1 2 3)
slice=(0 1 2 3 4 5 6 7 8)
bingau=(0 1 2)
method=(0 1)
sigma=(0 1 2)
fov=(0 1 2)
grid=(0)

for t in ${tavg[@]}
do
    for s in ${slice[@]}
    do
        for b in ${bingau[@]}
        do
            for m in ${method[@]}
            do
                for i in ${sigma[@]}
                do
                    for f in ${fov[@]}
                    do
                        for g in ${grid[@]}
                        do
                            python3 Muram_Inductivity.py $t $s $b $m $i $f $g 0
                        done
                    done
                done
            done
        done
    done
done
