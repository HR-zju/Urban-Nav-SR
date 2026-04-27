import Foundation
import CoreLocation

enum CoordinateTransform {
    private static let earthA = 6378245.0
    private static let ee = 0.00669342162296594323

    static func wgs84ToGCJ02(_ coordinate: CLLocationCoordinate2D) -> CLLocationCoordinate2D {
        guard !isOutsideChina(coordinate) else {
            return coordinate
        }

        let delta = delta(for: coordinate)
        return CLLocationCoordinate2D(
            latitude: coordinate.latitude + delta.latitude,
            longitude: coordinate.longitude + delta.longitude
        )
    }

    private static func isOutsideChina(_ coordinate: CLLocationCoordinate2D) -> Bool {
        coordinate.longitude < 72.004 ||
        coordinate.longitude > 137.8347 ||
        coordinate.latitude < 0.8293 ||
        coordinate.latitude > 55.8271
    }

    private static func delta(for coordinate: CLLocationCoordinate2D) -> CLLocationCoordinate2D {
        var dLat = transformLat(x: coordinate.longitude - 105.0, y: coordinate.latitude - 35.0)
        var dLon = transformLon(x: coordinate.longitude - 105.0, y: coordinate.latitude - 35.0)
        let radLat = coordinate.latitude / 180.0 * Double.pi
        var magic = sin(radLat)
        magic = 1 - ee * magic * magic
        let sqrtMagic = sqrt(magic)
        dLat = (dLat * 180.0) / ((earthA * (1 - ee)) / (magic * sqrtMagic) * Double.pi)
        dLon = (dLon * 180.0) / (earthA / sqrtMagic * cos(radLat) * Double.pi)
        return CLLocationCoordinate2D(latitude: dLat, longitude: dLon)
    }

    private static func transformLat(x: Double, y: Double) -> Double {
        var value = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * sqrt(abs(x))
        value += (20.0 * sin(6.0 * x * Double.pi) + 20.0 * sin(2.0 * x * Double.pi)) * 2.0 / 3.0
        value += (20.0 * sin(y * Double.pi) + 40.0 * sin(y / 3.0 * Double.pi)) * 2.0 / 3.0
        value += (160.0 * sin(y / 12.0 * Double.pi) + 320.0 * sin(y * Double.pi / 30.0)) * 2.0 / 3.0
        return value
    }

    private static func transformLon(x: Double, y: Double) -> Double {
        var value = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * sqrt(abs(x))
        value += (20.0 * sin(6.0 * x * Double.pi) + 20.0 * sin(2.0 * x * Double.pi)) * 2.0 / 3.0
        value += (20.0 * sin(x * Double.pi) + 40.0 * sin(x / 3.0 * Double.pi)) * 2.0 / 3.0
        value += (150.0 * sin(x / 12.0 * Double.pi) + 300.0 * sin(x / 30.0 * Double.pi)) * 2.0 / 3.0
        return value
    }
}
