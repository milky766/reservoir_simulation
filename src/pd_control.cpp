// pd_control.cpp

#include "pd_control.h"
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <termios.h>   
#include <fcntl.h>

PDController::PDController(const std::string &device_name, int baud_rate, const std::vector<int> &motor_ids)
    : motor_ids_(motor_ids) {
    packetHandler_ = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);
    for (size_t i = 0; i < motor_ids_.size(); ++i) {
    portHandlers_.push_back(dynamixel::PortHandler::getPortHandler(device_name.c_str()));
    }
    initial_positions_.resize(motor_ids.size());
    previous_positions_.resize(motor_ids.size());
}

PDController::~PDController() {
    for (auto &portHandler : portHandlers_) {
        portHandler->closePort();
    }
}

bool PDController::initialize() {
    uint8_t error = 0;
    for (int i = 0; i < motor_ids_.size(); ++i) {
        if (!portHandlers_[i]->openPort() || !portHandlers_[i]->setBaudRate(BAUDRATE)) {
            std::cerr << "Failed to initialize port or baudrate for motor " << motor_ids_[i] << std::endl;
            return false;
        }

        int result = packetHandler_->write1ByteTxRx(portHandlers_[i], motor_ids_[i], ADDR_OPERATING_MODE, OPERATING_MODE_CURRENT, &error);
        if (result != COMM_SUCCESS || error != 0) return false;

        packetHandler_->write1ByteTxRx(portHandlers_[i], motor_ids_[i], ADDR_TORQUE_ENABLE, TORQUE_ENABLE, &error);
        
        uint32_t initial_position;
        packetHandler_->read4ByteTxRx(portHandlers_[i], motor_ids_[i], ADDR_PRESENT_POSITION, &initial_position, &error);
        initial_positions_.push_back(initial_position);
    }
    return true;
}

std::vector<std::vector<DataRecord>> PDController::runControl(double duration) {
    std::vector<std::vector<DataRecord>> data_log(motor_ids_.size());
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        if (kbhit()) {
            std::cout << "Key pressed! Stopping the motors." << std::endl;
            break;
        }

        double elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        if (elapsed_time >= duration) break;

        for (size_t i = 0; i < motor_ids_.size(); ++i) {
            int32_t present_position = 0;
            uint8_t error = 0;
            packetHandler_->read4ByteTxRx(portHandlers_[i], motor_ids_[i], ADDR_PRESENT_POSITION, (uint32_t*)&present_position, &error);
            int32_t position_error = TARGET_POSITION - present_position;
            double velocity = (present_position - previous_positions_[i]) / elapsed_time;

            int16_t goal_current = std::clamp(static_cast<int16_t>(P_GAIN * position_error - D_GAIN * velocity), 
                                              static_cast<int16_t>(-MAX_CURRENT), static_cast<int16_t>(MAX_CURRENT));
            packetHandler_->write2ByteTxRx(portHandlers_[i], motor_ids_[i], ADDR_GOAL_CURRENT, goal_current, &error);

            int16_t present_current = 0;
            packetHandler_->read2ByteTxRx(portHandlers_[i], motor_ids_[i], ADDR_PRESENT_CURRENT, (uint16_t*)&present_current, &error);
            double real_current = present_current * CURRENT_RESOLUTION;

            data_log[i].push_back({
                elapsed_time,
                real_current,
                (real_current - CURRENT_OFFSET) / CURRENT_TO_TORQUE_RATIO,
                (present_position - initial_positions_[i]) * POSITION_RESOLUTION,
                velocity * POSITION_RESOLUTION,
                position_error * POSITION_RESOLUTION
            });

            previous_positions_[i] = present_position;
        }
        usleep(10000);
    }
    
    for (size_t i = 0; i < motor_ids_.size(); ++i) {
        uint8_t error = 0;
        packetHandler_->write1ByteTxRx(portHandlers_[i], motor_ids_[i], ADDR_TORQUE_ENABLE, TORQUE_DISABLE, &error);
    }
    return data_log;
}

void PDController::saveData(const std::vector<std::vector<DataRecord>> &data_log, const std::string &filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        // ヘッダー行を作成（時間と各モータのデータ列）
        file << "Time (s)";
        for (size_t motor_index = 0; motor_index < data_log.size(); ++motor_index) {
            file << ",Current" << motor_index + 1 
                 << ",Torque" << motor_index + 1 
                 << ",Position" << motor_index + 1 
                 << ",Velocity" << motor_index + 1 
                 << ",Error" << motor_index + 1;
        }
        file << "\n";

        // データの行数を決定
        size_t max_rows = 0;
        for (const auto& log : data_log) {
            max_rows = std::max(max_rows, log.size());
        }

        // 各行のデータを出力
        for (size_t row = 0; row < max_rows; ++row) {
            // タイムスタンプ（最初のモータの時間を使用）
            if (!data_log[0].empty() && row < data_log[0].size()) {
                file << data_log[0][row].time;
            } else {
                file << "N/A";
            }

            // 各モータのデータを出力
            for (size_t motor_index = 0; motor_index < data_log.size(); ++motor_index) {
                if (row < data_log[motor_index].size()) {
                    const auto& record = data_log[motor_index][row];
                    file << "," << record.current 
                         << "," << record.torque 
                         << "," << record.position 
                         << "," << record.velocity 
                         << "," << record.error;
                } else {
                    // データがない場合は空欄
                    file << ",N/A,N/A,N/A,N/A,N/A";
                }
            }
            file << "\n";
        }

        file.close();
        std::cout << "Data saved to " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file for writing!" << std::endl;
    }
}


// タイムスタンプ取得関数の実装
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&in_time_t), "%Y%m%d%H%M%S");
    return oss.str();
}

// PDControllerのメンバー関数としてkbhitを定義
int PDController::kbhit() {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}