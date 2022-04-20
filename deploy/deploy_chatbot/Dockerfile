FROM lambda-torch1.7

# copy function cpde and models into /var/task
COPY ./ ${LAMBDA_TASK_ROOT}/
COPY requirements.txt requirements.txt

# install our dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# remove dataclasses as it conflicts with transformers
RUN  rm -f ${LAMBDA_TASK_ROOT}/dataclasses.py

# Set the CMD to your handler 
CMD ["handler.predict"]

