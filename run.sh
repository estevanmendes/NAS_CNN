for c in {0..31}
do
if [ $((c%2)) -eq 0 ];
then
    python3.8 NAS_GA.py -sg $c --gpu 1;
else
    python3.8 NAS_GA.py -sg $c --gpu 2;
fi
    sleep 1m
done