
env_txt="./environment.txt"

echo "-- Platform info:" > $env_txt
lsb_release -a >> $env_txt

echo "-- GPU info: " >> $env_txt
nvidia-smi >> $env_txt

echo "-- Python info: " >> $env_txt
python --version >> $env_txt
python -c "import torch; print(f'torch version: {torch.__version__}\n')" >> $env_txt

python run_FB15k.py
