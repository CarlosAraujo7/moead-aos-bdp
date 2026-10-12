@echo off
cd /d C:\Pesquisas\moead-aos-bdp
python run_e20.py --cells CDP-S GDP-S --workers 3 --reps 3 --modes m_hyb_vns30,m_hyb_vns50 > results\e20_log_CG.txt 2>&1
