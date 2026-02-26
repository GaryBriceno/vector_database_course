instalar LLVM (Homebrew) y compilar con su clang + OpenMP

brew install llvm libomp

export CC="$(brew --prefix llvm)/bin/clang"
export CXX="$(brew --prefix llvm)/bin/clang++"
export LDFLAGS="-L$(brew --prefix libomp)/lib"
export CPPFLAGS="-I$(brew --prefix libomp)/include"
export CFLAGS="-Xpreprocessor -fopenmp"
export CXXFLAGS="-Xpreprocessor -fopenmp"

Si estás en un venv, ejecuta los exports en la misma terminal donde haces el pip install.

