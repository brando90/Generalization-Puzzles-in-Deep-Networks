FROM ubuntu:16.04

# note this is needed to be able to install pip3
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev
RUN apt-get install -y vim

# get python
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

#Install some stuff my lib needs
RUN pip3 install numpy
RUN pip3 install namespaces
RUN pip3 install scikit-learn
RUN pip3 install scipy
RUN pip3 install pdb
RUN pip3 install sklearn
RUN pip3 install scipy
RUN pip3 install matplotlib
RUN pip3 install maps
RUN pip3 install sympy

# install my library (only when the a container is spun)
# bash script that configures the running container
#ENTRYPOINT ["sh", "/home_simulation_research/hbf_tensorflow_code/docker_files/tf_gpu/container_setup.sh"]

# enviromnet
#ENV OMP_NUM_THREADS=2
