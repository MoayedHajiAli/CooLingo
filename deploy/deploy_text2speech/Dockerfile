FROM lambda-torch1.7

RUN yum -y install gcc gcc-c++
RUN yum -y install libsndfile
RUN yum -y install espeak

# copy function cpde and models into /var/task
COPY ./ ${LAMBDA_TASK_ROOT}/
COPY requirements.txt requirements.txt


# install our dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# install TTS
RUN python3 -m pip install --no-cache-dir numpy Cython
RUN python3 -m pip install --no-cache-dir ${LAMBDA_TASK_ROOT}/TTS_repo --target ${LAMBDA_TASK_ROOT}


# build monotonic_align
WORKDIR TTS_repo/TTS/tts/layers/glow_tts/monotonic_align/
RUN python3 setup.py build_ext --build-lib=${LAMBDA_TASK_ROOT}
WORKDIR ${LAMBDA_TASK_ROOT}

# remove dataclasses as it conflicts with python3.8
RUN rm -f ${LAMBDA_TASK_ROOT}/dataclasses.py

# install phenomizer 2.2
RUN rm -rf ${LAMBDA_TASK_ROOT}/phonemizer
RUN python3 -m pip install --no-cache-dir phonemizer==2.2 --target ${LAMBDA_TASK_ROOT}

RUN python3 -m pip install --no-cache-dir ${LAMBDA_TASK_ROOT}/TTS_repo --target ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler 
CMD ["handler.predict"]

