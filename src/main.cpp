#include "pd_control.h"
#include <iostream>


int main() {
    std::vector<int> motor_ids = {1, 2}; // 2モータのIDを指定
    PDController controller("/dev/ttyUSB0", 57600, motor_ids);

    if (!controller.initialize()) {
        std::cerr << "Failed to initialize controller." << std::endl;
        return -1;
    }

    auto data_log = controller.runControl(5.0); // 5秒間の運転

    controller.saveData(data_log, "data/pd_control/output.csv");

    return 0;
}
