var gaul = ee.FeatureCollection("FAO/GAUL/2015/level1");

// Filter only Pakistan's Level 1 regions
var pakistanRegions = gaul.filter(ee.Filter.eq('ADM0_NAME', 'Pakistan'));

// Print the names of each region
var names = pakistanRegions.aggregate_array('ADM1_NAME');
print('Administrative regions of Pakistan (GAUL Level 1):', names);
