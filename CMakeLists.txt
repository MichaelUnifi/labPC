cmake_minimum_required(VERSION 3.10)
project(LSH_OpenMP_Lab CXX)

# 1. Configurazione Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Ricerca OpenMP
find_package(OpenMP REQUIRED)

# 3. Definizione della libreria core (Logica LSH)
# Assicurati di aver salvato le implementazioni delle classi in structures.cpp
add_library(lsh_core structures.cpp)
target_link_libraries(lsh_core PUBLIC OpenMP::OpenMP_CXX)

# 4. Definizione degli eseguibili
add_executable(benchmark_sequential benchmark_sequential.cpp)
target_compile_options(lsh_core PRIVATE -mavx2)

add_executable(benchmark_parallel benchmark_parallel.cpp)
target_compile_options(benchmark_parallel PRIVATE -O3 -march=native -pg)
target_link_options(benchmark_parallel PRIVATE -pg)
target_link_libraries(benchmark_parallel PRIVATE lsh_core OpenMP::OpenMP_CXX)
# 5. Link delle librerie e dei flag
# Colleghiamo la logica core a entrambi
target_link_libraries(benchmark_sequential PRIVATE lsh_core)
target_link_libraries(benchmark_parallel PRIVATE lsh_core)

target_link_libraries(benchmark_sequential PRIVATE lsh_core OpenMP::OpenMP_CXX)