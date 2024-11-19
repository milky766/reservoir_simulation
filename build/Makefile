# 変数定義
CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall $(INCLUDES)
INCLUDES = -I../include -I../include/dynamixel_sdk
LDFLAGS = -L../../../../c++/build/linux64 # ライブラリの相対パス
LIBS = -ldxl_x64_cpp

# 出力ファイル名
TARGET = main

# ソースファイルとオブジェクトファイル
SRCS = ../src/main.cpp ../src/pd_control.cpp
OBJS = $(SRCS:.cpp=.o)

# デフォルトターゲット
all: $(TARGET)

# ターゲットのビルドルール
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

# オブジェクトファイルのビルドルール
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f ../src/*.o main
