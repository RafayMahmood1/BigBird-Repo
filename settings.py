import os


abs_path=os.path.abspath(os.getcwd())
path=os.path.join('data','index')
new_path=os.path.join(abs_path,path)

DEBUG=True
SRC_PATH=new_path
APIGATEWAY_ID='s8iibycy4d' # ID of the api gateway which connects with the Ec2 instance