export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

rm -rf $BASH_ROOT/nvbit_release

VERSION="1.5.3"
ARCHIVE="nvbit-Linux-x86_64-$VERSION.tar.bz2"

curl -L --output $ARCHIVE https://github.com/NVlabs/NVBit/releases/download/$VERSION/$ARCHIVE
tar -xf $ARCHIVE -C $BASH_ROOT

rm $ARCHIVE
