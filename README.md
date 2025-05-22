Create Environment:
python -m venv myenv
source myenv/bin/activate
deactivate


myenv:
pip install tensorflow==2.16.2

EC2 instance:
sudo chmod 400 ~/FYP/BiLSTM\ \&\ Detection/EC2/cloudEC2.pem 
ssh -i ~/FYP/BiLSTM\ \&\ Detection/EC2/cloudEC2.pem ubuntu@ip


Website:
publicip:5000