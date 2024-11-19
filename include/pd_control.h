#ifndef PD_CONTROL_H
#define PD_CONTROL_H

#include <vector>
#include <string>
#include <dynamixel_sdk.h>

// 定数の設定
#define P_GAIN 10
#define D_GAIN 1
#define MAX_CURRENT 20
#define CURRENT_OFFSET 40.0
#define CURRENT_TO_TORQUE_RATIO 595.9
#define POSITION_RESOLUTION 0.088
#define CURRENT_RESOLUTION 2.69

#define PROTOCOL_VERSION 2.0
#define BAUDRATE 57600
#define DXL_ID 1
#define ADDR_OPERATING_MODE 11
#define ADDR_TORQUE_ENABLE 64
#define ADDR_CURRENT_LIMIT 38
#define ADDR_GOAL_CURRENT 102
#define ADDR_PRESENT_CURRENT 126
#define ADDR_PRESENT_POSITION 132
#define OPERATING_MODE_CURRENT 0
#define TORQUE_ENABLE 1
#define TORQUE_DISABLE 0
#define TARGET_POSITION 1024
#define MAX_CURRENT 20

// データを記録する構造体
struct DataRecord {
    double time;
    double current;
    double torque;
    double position;
    double velocity;
    double error;
};

// クラス定義にして、初期化から制御、データ保存まで一括で行う
class PDController {
public:
    PDController(const std::string &device_name, int baud_rate, const std::vector<int> &motor_ids);
    ~PDController();
    bool initialize();
    std::vector<std::vector<DataRecord>> runControl(double duration);
    void saveData(const std::vector<std::vector<DataRecord>> &data_log, const std::string &filename);
    
private:
    std::vector<int> motor_ids_;
    std::vector<int32_t> initial_positions_;
    std::vector<int32_t> previous_positions_;
    std::vector<dynamixel::PortHandler *> portHandlers_;
    dynamixel::PacketHandler *packetHandler_;

    int kbhit();  // 追加：kbhitのプロトタイプ宣言
};

// タイムスタンプ取得関数
std::string getCurrentTimestamp();

#endif // PD_CONTROL_H
