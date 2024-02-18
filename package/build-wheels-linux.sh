#!/bin/bash
PYTHON_VERSIONS=("cp38-cp38" "cp39-cp39" "cp310-cp310")

uname -a
echo "Current CentOS Version:"
cat /etc/centos-release

ls -ltrh /io/

CURRDIR=$(pwd)
echo "Num. processes to use for building: $(nproc)"

# ------ Install dependencies from the default repositories ------
cd $CURRDIR
yum install -y wget git gcc gcc-c++ cmake make build-essential libopencv-dev

# ------ Build pytlsd wheel ------
cd /io/
WHEEL_DIR="wheels/"
for PYTHON_VERSION in ${PYTHON_VERSIONS[@]}; do
    PYTHON_EXEC="/opt/python/${PYTHON_VERSION}/bin/python"
    ${PYTHON_EXEC} -m pip wheel --no-deps -w ${WHEEL_DIR} .
done

PYTHON_DEFAULT="/opt/python/${PYTHON_VERSIONS[-1]}/bin/python"
${PYTHON_DEFAULT} -m pip install auditwheel

# Bundle external shared libraries into the wheels
OUT_DIR="/io/wheelhouse"
mkdir -p ${OUT_DIR}
for whl in ${WHEEL_DIR}/*.whl; do
    auditwheel repair "$whl" -w ${OUT_DIR} --plat ${PLAT}
done
ls -ltrh ${OUT_DIR}
