FROM debian
MAINTAINER ytbilly3636

RUN apt-get update

RUN apt-get install -y              \
            python3-dev             \
            python3-pip             \
            python3-tk
            
RUN pip3    install                 \
            numpy                   \
            matplotlib

RUN mkdir /dir
WORKDIR /dir
VOLUME ["/dir"]

CMD ["bash"]