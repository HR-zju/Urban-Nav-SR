#pragma once

#include <fstream>
#include <string>
#include <memory>

namespace gnss_comm {

class KmlExporter {
public:
    KmlExporter(const std::string& directory);
    ~KmlExporter();

    // Prevent copying
    KmlExporter(const KmlExporter&) = delete;
    KmlExporter& operator=(const KmlExporter&) = delete;

    void add_point(double lat, double lon, double alt);

private:
    std::ofstream file_;
    std::string file_path_;
};

} // namespace gnss_comm
