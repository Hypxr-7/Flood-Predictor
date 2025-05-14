var countries = ee.FeatureCollection("FAO/GAUL/2015/level1");

// -----------------------------------------
// 1. ROI: Balochistan
// -----------------------------------------
var balochistan = countries
  .filter(ee.Filter.eq('ADM1_NAME', 'Balochistan'))
  .filter(ee.Filter.eq('ADM0_NAME', 'Pakistan'))
  .geometry();

Map.centerObject(balochistan)

// -----------------------------------------
// 2. Set Time Range - Keep 5 years | Higher reaches in max memory limit
// -----------------------------------------
var startDate = ee.Date('2000-01-01');
var endDate = startDate.advance(5, 'year');
var dateRange = ee.DateRange(startDate, endDate);

// -----------------------------------------
// 3. Land Surface Temperature (MOD11A1)
// -----------------------------------------
var LST_Day = ee.ImageCollection('MODIS/061/MOD11A1')
  .filterDate(dateRange)
  .select('LST_Day_1km')
  .map(function(img) {
    return img
      .multiply(0.02) // scale factor
      .subtract(273.15) // convert K to °C
      .copyProperties(img, ['system:time_start']);
  });

var meanLST = LST_Day.mean().clip(balochistan);

// -----------------------------------------
// 4. Vegetation Index (NDVI - MOD13A2)
// -----------------------------------------
var NDVI = ee.ImageCollection('MODIS/061/MOD13A2')
  .filterDate(dateRange)
  .select('NDVI');

var meanNDVI = NDVI.mean().clip(balochistan);

// -----------------------------------------
// 5. Precipitation (CHIRPS)
// -----------------------------------------
var precipitation = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(dateRange)
  .select('precipitation');

var meanPrecipitation = precipitation.mean().clip(balochistan);

// -----------------------------------------
// 6. Ice Presence (NDSI - MOD09GA)
// -----------------------------------------
var NDSI = ee.ImageCollection('MODIS/MOD09GA_006_NDSI')
  .filterDate(dateRange)
  .select('NDSI');

var meanNDSI = NDSI.mean().clip(balochistan);

// -----------------------------------------
// 7. Visualization Parameters
// -----------------------------------------
var ndviVis = {
  min: 0,
  max: 9000,
  palette: [
    "FFFFFF", "CE7E45", "DF923D", "F1B555", "FCD163", "99B718", "74A901",
    "66A000", "529400", "3E8601", "207401", "056201", "004C00", "023B01",
    "012E01", "011D01", "011301"
  ]
};

var tempVis = {
  min: 20,
  max: 40,
  palette: ['blue', 'limegreen', 'yellow', 'darkorange', 'red']
};

var precipitationVis = {
  min: 1.0,
  max: 17.0,
  palette: ['001137', '0aab1e', 'e7eb05', 'ff4a2d', 'e90000']
};

var iceVis = {
  palette: ['000088', '0000FF', '8888FF', 'FFFFFF']
};

// -----------------------------------------
// 8. Add Layers to Map
// -----------------------------------------
Map.addLayer(meanNDVI, ndviVis, 'Mean NDVI (2000-2005)');
Map.addLayer(meanLST, tempVis, 'Mean LST °C (2000-2005)');
Map.addLayer(meanPrecipitation, precipitationVis, 'Mean Precipitation (2000-2005)');
Map.addLayer(meanNDSI, iceVis, 'Mean NDSI (Ice) (2000-2005)');

// -----------------------------------------
// 9. Time Series Charts
// -----------------------------------------

// Temperature Time Series
var lstChart = ui.Chart.image.series({
  imageCollection: LST_Day,
  region: balochistan,
  reducer: ee.Reducer.mean(),
  scale: 1000,
  xProperty: 'system:time_start'
}).setOptions({
  title: 'Temperature Time Series (°C)',
  vAxis: {title: 'LST (°C)'},
  lineWidth: 2,
  pointSize: 4
});
print(lstChart);

// NDVI Time Series
var ndviChart = ui.Chart.image.series({
  imageCollection: NDVI,
  region: balochistan,
  reducer: ee.Reducer.mean(),
  scale: 1000,
  xProperty: 'system:time_start'
}).setOptions({
  title: 'Vegetation Index (NDVI) Time Series',
  vAxis: {title: 'NDVI'},
  lineWidth: 2,
  pointSize: 4
});
print(ndviChart);

// Ice Presence Time Series
var iceChart = ui.Chart.image.series({
  imageCollection: NDSI,
  region: balochistan,
  reducer: ee.Reducer.mean(),
  scale: 500,
  xProperty: 'system:time_start'
}).setOptions({
  title: 'Ice Presence (NDSI) Time Series',
  vAxis: {title: 'NDSI'},
  lineWidth: 2,
  pointSize: 4
});
print(iceChart);

// Precipitation Time Series
var precipChart = ui.Chart.image.series({
  imageCollection: precipitation,
  region: balochistan,
  reducer: ee.Reducer.mean(),
  scale: 2500,
  xProperty: 'system:time_start'
}).setOptions({
  title: 'Precipitation Time Series (mm)',
  vAxis: {title: 'Precipitation (mm)'},
  lineWidth: 2,
  pointSize: 4
});
print(precipChart);
