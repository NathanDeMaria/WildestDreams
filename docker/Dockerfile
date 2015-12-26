FROM tleyden5iwx/caffe-cpu-master

RUN wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.0-Linux-x86_64.sh && bash Anaconda2-2.4.0-Linux-x86_64.sh -b

RUN git clone https://github.com/google/protobuf && \
    apt-get install dh-autoreconf -y && \
    cd protobuf && \
    ./autogen.sh && \
    ./configure --prefix=/usr && \
    make && make check && make install

RUN cd /protobuf/python && python setup.py build && python setup.py install

RUN pip install jupyter==1.0.0 pydot h5py lmdb jinja2
RUN apt-get install -y graphviz

WORKDIR /root
RUN mkdir .jupyter
COPY ./jupyter_notebook_config.py /root/.jupyter

EXPOSE 8888
CMD ipython notebook

