# プロジェクト名と最低CMakeバージョン
cmake_minimum_required(VERSION 3.10)
project(reservoir_motor)

# C++の標準を指定
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigenのインクルードディレクトリを探す
find_package(Eigen3 REQUIRED)

# 実行可能ファイルを指定
add_executable(
    main
    src/arm_dynamics.cpp
    src/train_RC_robot.cpp
    src/test_RC_robot.cpp
    src/evaluation.cpp
    src/kinematics.cpp
    src/main.cpp 
    src/construct_network.cpp
    )

# ヘッダファイルのディレクトリをインクルードパスに追加
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/header)

# Eigenのインクルードディレクトリをリンク
target_link_libraries(main PRIVATE Eigen3::Eigen)

# ビルド後のカスタムターゲットを作成
add_custom_target(clean_intermediates ALL
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles/main.dir
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/main.cpp.o
)
