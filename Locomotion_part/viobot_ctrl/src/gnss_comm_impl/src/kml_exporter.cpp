#include "gnss_comm_impl/kml_exporter.hpp"
#include <iomanip>
#include <chrono>
#include <iostream>

namespace gnss_comm {

KmlExporter::KmlExporter(const std::string& directory) {
    // Generate filename with timestamp
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    
    char time_str[100];
    std::strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", std::localtime(&now_c));
    file_path_ = directory + "/spp_trajectory_" + std::string(time_str) + ".kml";

    file_.open(file_path_, std::ios::out);
    if (file_.is_open()) {
        file_ << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        file_ << "<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n";
        file_ << "  <Document>\n";
        file_ << "    <name>GNSS Trajectory</name>\n";
        file_ << "    <Style id=\"track_style\">\n";
        file_ << "      <LineStyle>\n";
        file_ << "        <color>ff0000ff</color>\n"; // Red
        file_ << "        <width>4</width>\n";
        file_ << "      </LineStyle>\n";
        file_ << "    </Style>\n";
        file_ << "    <Placemark>\n";
        file_ << "      <name>Path</name>\n";
        file_ << "      <styleUrl>#track_style</styleUrl>\n";
        file_ << "      <LineString>\n";
        file_ << "        <tessellate>1</tessellate>\n";
        file_ << "        <coordinates>\n";
    } else {
        std::cerr << "Failed to open KML file: " << file_path_ << std::endl;
    }
}

KmlExporter::~KmlExporter() {
    if (file_.is_open()) {
        file_ << "        </coordinates>\n";
        file_ << "      </LineString>\n";
        file_ << "    </Placemark>\n";
        file_ << "  </Document>\n";
        file_ << "</kml>\n";
        file_.close();
        std::cout << "KML file saved to " << file_path_ << std::endl;
    }
}

void KmlExporter::add_point(double lat, double lon, double alt) {
    if (file_.is_open()) {
        // KML coordinates format is: lon,lat,alt
        file_ << std::fixed << std::setprecision(8) 
              << "          " << lon << "," << lat << "," << alt << "\n";
        file_.flush();
    }
}

} // namespace gnss_comm
