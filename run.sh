for c in {0..29}
do
if [ $((c%2)) -eq 0 ];
then
    python3.8 NAS_GA.py -sg $c --gpu 1 --steps 1;
else
    python3.8 NAS_GA.py -sg $c --gpu 2 --steps 1;
fi
    sleep 1m
done
python3.8 NAS_GA.py -sg 30 --gpu 2 