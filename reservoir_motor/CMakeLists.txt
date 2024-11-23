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
    main.cpp 
    construct_network.cpp
    arm_dynamics.cpp
    train_RC_robot.cpp
    test_RC_robot.cpp
    evaluation.cpp
    kinematics.cpp
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
