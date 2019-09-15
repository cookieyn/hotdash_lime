echo '
rm results/*
python main.py -a robustmpc -q log -t fcc -l
python main.py -a robustmpc -q log -t fcc -l -d decision_tree_ready/robustmpc_norway_500.pk3
python plot_results.py -rd -rn
mv faithfulness/robustmpc.csv faithfulness/robustmpc_log_fcc.csv
rm results/*
python main.py -a robustmpc -q log -t oboe -l
python main.py -a robustmpc -q log -t oboe -l -d decision_tree_ready/robustmpc_oboe_500.pk3
python plot_results.py -rd -rn
mv faithfulness/robustmpc.csv faithfulness/robustmpc_log_oboe.csv
rm results/*
python main.py -a robustmpc -q hd -t norway -l
python main.py -a robustmpc -q hd -t norway -l -d decision_tree_ready/robustmpc_fcc_500.pk3
python plot_results.py -rd -rn
mv faithfulness/robustmpc.csv faithfulness/robustmpc_hd_norway.csv
rm results/*
python main.py -a robustmpc -q hd -t fcc -l
python main.py -a robustmpc -q hd -t fcc -l -d decision_tree_ready/robustmpc_norway_500.pk3
python plot_results.py -rd -rn
mv faithfulness/robustmpc.csv faithfulness/robustmpc_hd_fcc.csv
rm results/*
python main.py -a robustmpc -q hd -t oboe -l
python main.py -a robustmpc -q hd -t oboe -l -d decision_tree_ready/robustmpc_oboe_500.pk3
python plot_results.py -rd -rn
mv faithfulness/robustmpc.csv faithfulness/robustmpc_hd_oboe.csv

python learn_dt.py -a robustmpc -t oboe -n 500 -w 32 -i 500

python learn_dt.py -a hotdash -t oboe -n 100 -i 500 &
python learn_dt.py -a hotdash -t oboe -n 200 -i 500 &
python learn_dt.py -a hotdash -t oboe -n 500 -i 500 &
python learn_dt.py -a hotdash -t oboe -n 1000 -i 500 &
python learn_dt.py -a hotdash -t oboe -n 2000 -i 500 &
python learn_dt.py -a hotdash -t norway -n 100 -i 500 &
python learn_dt.py -a hotdash -t norway -n 200 -i 500 &
python learn_dt.py -a hotdash -t norway -n 500 -i 500 &
python learn_dt.py -a hotdash -t norway -n 1000 -i 500 &
python learn_dt.py -a hotdash -t norway -n 2000 -i 500 &

# python learn_dt.py -a hotdash -t fcc -n 100 -i 500 &
# python learn_dt.py -a hotdash -t fcc -n 200 -i 500 &
# python learn_dt.py -a hotdash -t fcc -n 500 -i 500 &
# python learn_dt.py -a hotdash -t fcc -n 1000 -i 500 &
# python learn_dt.py -a hotdash -t fcc -n 2000 -i 500
# python learn_dt.py -a robustmpc -t oboe -n 1000 -w 32 -i 500
# python learn_dt.py -a robustmpc -t oboe -n 2000 -w 32 -i 500

python main.py -a robustmpc -q lin -t norway -l &
python main.py -a robustmpc -q lin -t fcc -l &
python main.py -a robustmpc -q lin -t oboe -l &
python main.py -a pensieve -q lin -t norway -l &
python main.py -a pensieve -q lin -t fcc -l &
python main.py -a pensieve -q lin -t oboe -l &
python main.py -a hotdash -q lin -t norway -l &
python main.py -a hotdash -q lin -t fcc -l &
python main.py -a hotdash -q lin -t oboe -l &
python main.py -a robustmpc -q log -t norway -l &
python main.py -a robustmpc -q log -t fcc -l &
python main.py -a robustmpc -q log -t oboe -l &
python main.py -a pensieve -q log -t norway -l &
python main.py -a pensieve -q log -t fcc -l &
python main.py -a pensieve -q log -t oboe -l &
python main.py -a hotdash -q log -t norway -l &
python main.py -a hotdash -q log -t fcc -l &
python main.py -a hotdash -q log -t oboe -l &
python main.py -a robustmpc -q hd -t norway -l &
python main.py -a robustmpc -q hd -t fcc -l &
python main.py -a robustmpc -q hd -t oboe -l &
python main.py -a pensieve -q hd -t norway -l &
python main.py -a pensieve -q hd -t fcc -l &
python main.py -a pensieve -q hd -t oboe -l &
python main.py -a hotdash -q hd -t norway -l &
python main.py -a hotdash -q hd -t fcc -l &
python main.py -a hotdash -q hd -t oboe -l &
'
python learn_dt.py -a robustmpc -t norway -n 100 -w 32 -i 500
python learn_dt.py -a robustmpc -t oboe -n 100 -w 32 -i 500
python learn_dt.py -a robustmpc -t norway -n 200 -w 32 -i 500
python learn_dt.py -a robustmpc -t oboe -n 200 -w 32 -i 500
