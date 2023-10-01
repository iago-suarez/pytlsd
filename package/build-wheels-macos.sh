#!/bin/bash
set -x -e

PYTHON_VERSIONS=("3.8" "3.9" "3.10")

# See https://github.com/actions/setup-python/issues/577
find /usr/local/bin -lname '*/Library/Frameworks/Python.framework/*' -delete
# See https://github.com/actions/setup-python/issues/577#issuecomment-1500828576
rm /usr/local/bin/go || true
rm /usr/local/bin/gofmt || true

CURRDIR=$(pwd)
NUM_LOGICAL_CPUS=$(sysctl -n hw.logicalcpu)
echo "Number of logical CPUs is: ${NUM_LOGICAL_CPUS}"

# Updating requires Xcode 14.0, which cannot be installed on macOS 11.
brew remove swiftlint
brew remove node@18

brew update
brew upgrade
brew install git wget cmake

for PYTHON_VERSION in ${PYTHON_VERSIONS[@]}; do
    brew install --force "python@${PYTHON_VERSION}"
    python${PYTHON_VERSION} -m pip install -U pip setuptools wheel cffi
done

brew install llvm
brew info llvm
brew upgrade llvm
brew info llvm

# Install `delocate` -- OSX equivalent of `auditwheel`
# see https://pypi.org/project/delocate/ for more details

cd $CURRDIR
# flags must be passed, to avoid the issue: `Unsupported compiler -- pybind11 requires C++11 support!`
# see https://github.com/quantumlib/qsim/issues/242 for more details
WHEEL_DIR="${CURRDIR}/wheelhouse_unrepaired/"
for PYTHON_VERSION in ${PYTHON_VERSIONS[@]}; do
    CC=/usr/local/opt/llvm/bin/clang \
        CXX=/usr/local/opt/llvm/bin/clang++ \
        LDFLAGS=-L/usr/local/opt/libomp/lib \
        python${PYTHON_VERSION} -m pip wheel --no-deps -w ${WHEEL_DIR} .
done

python${PYTHON_VERSIONS[0]} -m pip install -U delocate

# Bundle external shared libraries into the wheels
OUT_DIR="${CURRDIR}/wheelhouse"
mkdir -p ${OUT_DIR}
for whl in ${WHEEL_DIR}/*.whl; do
    delocate-listdeps --all "$whl"
    delocate-wheel -w "${OUT_DIR}" -v "$whl"
    rm $whl
done
ls -ltrh ${OUT_DIR}
