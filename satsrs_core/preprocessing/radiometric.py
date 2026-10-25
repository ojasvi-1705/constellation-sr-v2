"""Radiometric Calibration for Satellite Imagery"""
import numpy as np

class RadiometricCalibrator:
    """Calibrate satellite imagery from L1C (TOA) to L2A (BOA)"""
    
    def __init__(self, solar_zenith_angle=30.0, day_of_year=180):
        self.solar_zenith_angle = solar_zenith_angle
        self.day_of_year = day_of_year
        self.solar_distance_correction = self._calculate_solar_distance()
    
    def _calculate_solar_distance(self):
        """Calculate solar distance correction"""
        day_angle = (self.day_of_year - 1) * 2 * np.pi / 365.25
        return 1.0 + 0.033 * np.cos(day_angle)
    
    def toa_to_boa(self, toa_image, band_index):
        """Convert TOA to BOA reflectance"""
        # Simplified conversion
        boa = toa_image / self.solar_distance_correction
        return np.clip(boa, 0, 1)
    
    def calculate_ndvi(self, red_band, nir_band):
        """Calculate NDVI index"""
        ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
        return ndvi

def create_calibrator(solar_zenith_angle=30.0):
    """Create radiometric calibrator"""
    return RadiometricCalibrator(solar_zenith_angle=solar_zenith_angle)
