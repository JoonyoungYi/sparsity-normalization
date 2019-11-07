FROM kaixhin/cuda-lasagne:8.0

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install git+https://github.com/casperkaae/parmesan.git@86776689f81ce59b14dfa0c8af861e1fc7b90877#egg=parmesan
RUN pip install coloredlogs
